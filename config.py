GPU = '9'

N_WORKERS = 4

IMAGE_SIZE = 512

MODEL = 'ssd_resnet50'

L2_REG = 0.0008

INIT_LR = 0.01
SCHEDULE = [4, 12, 20]
MOMENTUM = 0.8

EPOCHS = 26
BATCH_SIZE = 6
MODEL_NAME = 'ssd_resnet50'
WORK_DIR = '/home/raosj/checkpoints/face_detection'

PRE_TRAINED_WEIGHTS = '/home/raosj/pretrained-weights/weights-resnet50-imagenet'

TEST_IMAGE_PATH = 'images/test_image_1.jpg'

SSD_CONFIG = {
        'img_height': IMAGE_SIZE,
        'img_width': IMAGE_SIZE,
        'image_size': (IMAGE_SIZE, IMAGE_SIZE),
        'offsets': 0.5,
        'n_classes': 1,
        'min_scale': None,
        'max_scale': None,
        'scales': [0.04, 0.1, 0.26, 0.45, 0.58],
        'aspect_ratios_per_layer': [[1.0, 2.0, 0.5],
                                    [1.0, 2.0, 0.5],
                                    [1.0, 2.0, 0.5],
                                    [1.0, 2.0, 0.5]],
        'two_boxes_for_ar1': True,
        'steps': [4, 8, 16, 32],
        'variances': [0.1, 0.2],
}
