import os
import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from config import *
import models.ssd as ssd
from utils.losses import SSDLoss
from utils.box_utils import compute_nms, decode
from utils.anchor_utils import generate_anchors
import utils.transforms as trans


def inference_single_imagefile(_net, _image_path, _print=False, _show=False, _thresh=None):
    orig_image = cv.imread(_image_path)
    orig_image = cv.cvtColor(orig_image, cv.COLOR_BGR2RGB)
    img_height, img_width, _ = np.shape(orig_image)

    # Preprocess image
    input_image = cv.resize(orig_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv.INTER_LINEAR)
    input_image = np.array([input_image], dtype=np.float32)
    # normalize image
    input_image = trans.normalize_image(input_image)

    # Get predictions
    y_pred = _net.model.predict(input_image)
    if _thresh is None:
        confidence_threshold = CONF_THRESH
    else:
        confidence_threshold = _thresh
    confs = y_pred[:, :, :-4]
    locs = y_pred[:, :, -4:]

    # Decode predictions
    confs = tf.squeeze(confs, 0)
    locs = tf.squeeze(locs, 0)

    # confs = tf.math.softmax(confs, axis=-1)
    # classes = tf.math.argmax(confs, axis=-1)
    # scores = tf.math.reduce_max(confs, axis=-1)

    anchors = generate_anchors(SSD_CONFIG)
    boxes = decode(anchors, locs)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, 2):
        cls_scores = confs[:, c]

        score_idx = cls_scores > confidence_threshold
        # cls_boxes = tf.boolean_mask(boxes, score_idx)
        # cls_scores = tf.boolean_mask(cls_scores, score_idx)
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 1000)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    # Get boxes for visualize
    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    boxes[:, 0] = boxes[:, 0] * img_width
    boxes[:, 2] = boxes[:, 2] * img_width
    boxes[:, 1] = boxes[:, 1] * img_height
    boxes[:, 3] = boxes[:, 3] * img_height
    classes = np.array(out_labels)
    classes = np.expand_dims(classes, axis=1)
    scores = out_scores.numpy()
    scores = np.expand_dims(scores, axis=1)

    y_pred = np.concatenate([classes, scores, boxes], axis=1)

    if _print:
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Find %d boxes:\n" % len(y_pred))
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred)

    if _show:
        # Display the image and draw the predicted boxes onto it.
        plt.imshow(orig_image)

        current_axis = plt.gca()

        for box in y_pred:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            box = K.eval(box)
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            # label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=(0., 0.9, 0.), fill=False, linewidth=2))
            # current_axis.text(xmin, ymin, label, size='x-small', color='white', bbox={'facecolor': (0., 0.9, 0.), 'alpha': 1.0})

        plt.show()

    return y_pred


if __name__ == '__main__':


    # Build model
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

    # Load weights from checkpoints
    # checkpoint_path = 'None'
    # checkpoint_path = WORK_DIR + '/checkpoint-' + MODEL_NAME + '.h5'
    checkpoint_path = TEST_MODEL_WEIGHTS
    if not os.path.exists(checkpoint_path):
        raise ValueError("Model weights file not found.")
    net.load_weights(checkpoint_path)

    inference_single_imagefile(net, TEST_IMAGE_PATH, _print=True, _show=True)
