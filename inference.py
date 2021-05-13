import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from config import *
import models.ssd as ssd
from utils.losses import SSDLoss
from utils.layers import DecodeDetections


if MODEL == 'ssd_resnet50':
    net = ssd.SSD_ResNet50(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
else:
    raise ValueError("Unknown model: `{}`.".format(MODEL))

net.model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=MOMENTUM),
    loss=SSDLoss(),
)

checkpoint_path = WORK_DIR + '/checkpoint-' + MODEL_NAME + '.h5'
net.load_weights(checkpoint_path)

orig_image = cv.imread(TEST_IMAGE_PATH)
orig_image = cv.cvtColor(orig_image, cv.COLOR_BGR2RGB)
img_width = img_height = IMAGE_SIZE

input_image = cv.resize(orig_image, (512, 512), interpolation=cv.INTER_LINEAR)
input_image = np.array([input_image], dtype=np.float32)
# normalize image
input_image /= 255.0
input_image -= 0.5
input_image *= 2
y_pred = net.model.predict(input_image)

confidence_threshold = 0.3

decode = DecodeDetections(confidence_thresh=confidence_threshold,
                          iou_threshold=0.45,
                          top_k=500,
                          nms_max_output_size=1000,
                          coords='centroids',
                          normalize_coords=True,
                          img_height=img_height,
                          img_width=img_width,
                          name='decoded_predictions')

y_pred = decode.call(y_pred)

y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

# Display the image and draw the predicted boxes onto it.

# Set the colors for the bounding boxes
classes = ['background', 'face']

# plt.figure(figsize=(8, 8))
plt.imshow(orig_image)

current_axis = plt.gca()

for box in y_pred_thresh[0]:
    # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
    box = K.eval(box)
    xmin = box[2] * orig_image.shape[1] / img_width
    ymin = box[3] * orig_image.shape[0] / img_height
    xmax = box[4] * orig_image.shape[1] / img_width
    ymax = box[5] * orig_image.shape[0] / img_height
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(
        plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=(0., 0.9, 0.), fill=False, linewidth=2))
    # current_axis.text(xmin, ymin, label, size='x-small', color='white', bbox={'facecolor': (0., 0.9, 0.), 'alpha': 1.0})

plt.show()