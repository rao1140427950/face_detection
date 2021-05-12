GPU = '7'

IMAGE_SIZE = 512

MODEL = 'ssd_resnet50'

L2_REG = 0.0008

INIT_LR = 0.01
SCHEDULE = [4, 8, 12]
MOMENTUM = 0.8

EPOCHS = 16
BATCH_SIZE = 5
MODEL_NAME = 'ssd_resnet50'
WORK_DIR = '/home/raosj/checkpoints/face_detection'

PRE_TRAINED_WEIGHTS = '/home/raosj/pretrained-weights/weights-resnet50-imagenet'