import tensorflow as tf
import os
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from config import *
from utils.losses import SSDLoss
# from utils.pipelines import WiderFacePipeline
from utils.datasets import WiderFaceDataset
# from utils.callbacks import LearningRateScheduler
from utils.callbacks import AdvancedEarlyStopping
import models.ssd as ssd

# Configure environment
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
tf.config.threading.set_inter_op_parallelism_threads(6)
tf.config.threading.set_intra_op_parallelism_threads(6)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

start_epoch = START_EPOCH
epochs = EPOCHS
batch_size = BATCH_SIZE

model_name = MODEL_NAME

# Define filenames
if not os.path.exists(WORK_DIR):
    os.mkdir(WORK_DIR)
log_dir = WORK_DIR + '/' + model_name
output_model_file = WORK_DIR + '/' + model_name + '.h5'
weight_file = WORK_DIR + '/' + model_name + '_weights.h5'
checkpoint_path = WORK_DIR + '/checkpoint-' + model_name + '-{epoch:02d}-{val_loss:.2f}' + '.h5'

def create_datasets():
    train_dataset = WiderFaceDataset(
        SSD_CONFIG,
        txt_annos_path=TRAINING_ANNOS_PATH,
        image_root_dir=TRAINING_IMAGES_DIR,
        argument=True,
        batch_size=batch_size
    )

    val_dataset = WiderFaceDataset(
        SSD_CONFIG,
        txt_annos_path=VALIDATION_ANNOS_PATH,
        image_root_dir=VALIDATION_IMAGES_DIR,
        argument=False,
        batch_size=batch_size
    )

    _train_samples = train_dataset.generate_dataset_from_imagefile()
    _val_samples = val_dataset.generate_dataset_from_imagefile()
    return _train_samples, _val_samples

# Build data pipeline
if DATA_PIPELINE is not None:
    with tf.device(DATA_PIPELINE):
        train_samples, val_samples = create_datasets()
else:
    train_samples, val_samples = create_datasets()

# For multi-gpu training
if TRAINING_PIPELINE is not None:
    strategy = tf.distribute.MirroredStrategy(TRAINING_PIPELINE)
else:
    strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    if MODEL == 'ssd_resnet50':
        net = ssd.SSD_ResNet50(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            kernel_regularizer=l2(L2_REG),
        )
    elif MODEL == 'ssd_resnet101':
        net = ssd.SSD_ResNet101(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            kernel_regularizer=l2(L2_REG),
        )
    else:
        raise ValueError("Unknown model: `{}`.".format(MODEL))

    net.model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=MOMENTUM),
        loss=SSDLoss(),
    )

model = net.model

# Load weights by layer names
if PRE_TRAINED_WEIGHTS is not None:
    net.load_weights_by_layer(PRE_TRAINED_WEIGHTS)

# Load existing .h5 weights file
if os.path.exists(weight_file):
    model.load_weights(weight_file, by_name=True, skip_mismatch=True)
    print('Found weights file: {}, load weights.'.format(weight_file))
else:
    print('No weights file found. Skip loading weights.')


# Tensorboard
tensorboard = TensorBoard(log_dir=log_dir, write_images=False)

# Auto-save
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_weights_only=True,
    save_best_only=False,
    save_freq='epoch'
)

# Auto-decay learning rate
# lr_scheduler = LearningRateScheduler(SCHEDULE)
lr_scheduler = AdvancedEarlyStopping(patience=PATIENCE, filter_order=5, decay_rate=10, min_lr=1e-8, log_dir=log_dir)


# Start training
model.fit(
    x=train_samples,
    validation_data=val_samples,
    epochs=epochs,
    callbacks=[tensorboard, checkpoint, lr_scheduler],
    initial_epoch=start_epoch,
    shuffle=False,
    verbose=1,
)

model.save_weights(weight_file)
model.save(output_model_file)