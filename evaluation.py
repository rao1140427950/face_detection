import tensorflow as tf
import models.ssd as ssd
import models.ssd_fpn as ssd_fpn
from utils.losses import SSDLoss
import os
import numpy as np
from config import *
from inference import inference_single_imagefile

os.environ['CUDA_VISIBLE_DEVICES'] = '9'

# Build model
# Build model
if MODEL == 'ssd_resnet50':
    net = ssd.SSD_ResNet50(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
elif MODEL == 'ssdfpn_resnet_cbam':
    net = ssd_fpn.SSDFPN_ResNet_CBAM(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        repetitions=REPETITIONS,
        config=SSD_CONFIG,
    )
else:
    raise ValueError("Unknown model: `{}`.".format(MODEL))

net.model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=MOMENTUM),
    loss=SSDLoss(),
)

# Load weights from checkpoints
# checkpoint_path = 'None'
# checkpoint_path = WORK_DIR + '/checkpoint-' + MODEL_NAME + '.h5'
# checkpoint_path = WORK_DIR + '/checkpoint-ssd_resnet50_v2-63-2.87.h5'
checkpoint_path = WORK_DIR + '/checkpoint.h5'
weight_file = TEST_MODEL_WEIGHTS
if os.path.exists(weight_file):
    net.load_weights(weight_file)
    print('Load {}.'.format(weight_file))
elif os.path.exists(checkpoint_path):
    net.load_weights(checkpoint_path)
    print('Load {}.'.format(checkpoint_path))
else:
    raise ValueError("Checkpoint and weights file not found.")

if not os.path.exists(EVALUATION_RESULTS_DIR):
    os.mkdir(EVALUATION_RESULTS_DIR)

if not os.path.exists(VALIDATION_IMAGES_DIR):
    raise ValueError("Dir: {} not found.".format(VALIDATION_IMAGES_DIR))
parent_dirs =  os.listdir(VALIDATION_IMAGES_DIR)
print("Found {} dirs in {}.".format(len(parent_dirs), VALIDATION_IMAGES_DIR))
for parent_dir in parent_dirs:
    results_parent_dir = os.path.join(EVALUATION_RESULTS_DIR, parent_dir)
    parent_dir = os.path.join(VALIDATION_IMAGES_DIR, parent_dir)
    if not os.path.exists(results_parent_dir):
        os.mkdir(results_parent_dir)
    image_files = os.listdir(parent_dir)
    print("Found {} images in {}.".format(len(image_files), parent_dir))
    for image_file in image_files:
        result_file_name = image_file.split('.')[0]
        result_file = os.path.join(results_parent_dir, result_file_name + '.txt')
        image_file = os.path.join(parent_dir, image_file)
        preds = inference_single_imagefile(net, image_file, _thresh=0)
        scores = preds[:, 1]
        boxes = preds[:, 2:6].astype(np.int)
        lines = [result_file_name + "\n", "%d\n" % len(preds)]
        # (xmin, ymin, xmax, ymax) to (xmin, ymin, w, h)
        boxes += 1
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        for box, score in zip(boxes, scores):
            lines.append("%d %d %d %d %.3f\n" % (box[0], box[1], box[2], box[3], score))
        f = open(result_file, 'w')
        f.writelines(lines)
        f.close()
        print("%s created." % result_file)
