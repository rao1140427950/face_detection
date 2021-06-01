# The ID of visible GPUs for training
GPU = '8, 9'

# Specify device for generating data pipeline
# If DATA_PIPELINE = 'CPU:0', training and validation data is generated on CPU (NOT recommanded).
# If DATA_PIPELINE = None, training and validation data is generated on default device.
DATA_PIPELINE = 'GPU:1'
# Specify device for training procedure
# It's better to generate data pipeline on a seperate device. i.e. using 2 or more GPUs for
# training, and using 1 independent device for data pipeline. If data pipeline and training are
# done on the same device (set DATA_PIPELINE = None), BATCH_SIZE should be set to a much smaller
# value to avoid OOM.
TRAINING_PIPELINE = ['GPU:0']

IMAGE_SIZE = 512

# MODEL = 'ssd_resnet50'
MODEL = 'ssdfpn_resnet_cbam'
REPETITIONS = (3, 3, 3, 3)

L2_REG = 0.0008

INIT_LR = 0.01
# At which epoch learning rate is decay by 10.
# SCHEDULE = [12, 20, 28]
MOMENTUM = 0.8
PATIENCE = 6

START_EPOCH = 0
EPOCHS = 512
BATCH_SIZE = 6
MODEL_NAME = 'ssdfpn_resnet3333_cbam'
WORK_DIR = '/home/raosj/checkpoints/face_detection'

# Dataset dir
DATA_ROOT_DIR = '/home/raosj/datasets/wider_face/'
TRAINING_ANNOS_PATH = DATA_ROOT_DIR + 'wider_face_split/wider_face_train_bbx_gt.txt'
TRAINING_IMAGES_DIR = DATA_ROOT_DIR + 'WIDER_train/images'
VALIDATION_ANNOS_PATH = DATA_ROOT_DIR + 'wider_face_split/wider_face_val_bbx_gt.txt'
VALIDATION_IMAGES_DIR = DATA_ROOT_DIR + 'WIDER_val/images'

PRE_TRAINED_WEIGHTS = '/home/raosj/pretrained-weights/weights-resnet50-imagenet'

# For inference
TEST_IMAGE_PATH = 'images/test_image_1.jpg'
TEST_MODEL_WEIGHTS = WORK_DIR + '/checkpoint-ssdfpn_resnet3333_cbam-313-4.06.h5'
CONF_THRESH = 0.5

# For evaluation
EVALUATION_RESULTS_DIR = 'evaluation_results'

SSD_CONFIG = {
        'img_height': IMAGE_SIZE,
        'img_width': IMAGE_SIZE,
        'image_size': (IMAGE_SIZE, IMAGE_SIZE),
        'offsets': 0.5,
        'n_classes': 1,
        'min_scale': None,
        'max_scale': None,
        # 'scales': [0.005, 0.008, 0.018, 0.035, 0.28, 0.60],
        'scales': [0.007, 0.03, 0.08, 0.28, 0.60],
        'aspect_ratios_per_layer': [[0.9, 1.8, 2.5],
                                    [0.9, 1.8, 2.5],
                                    [0.9, 1.8, 2.5],
                                    [0.9, 1.8, 2.5],],
        'two_boxes_for_ar1': True,
        'steps': [4, 8, 16, 32],
        'variances': [0.1, 0.2],
}

TRANS_CONFIG = {
    'brightness': 0.15,
    'hue': 0.05,
    'contrast': 0.15,
    'saturation': 0.15,
    'shrink': 0.7,
}