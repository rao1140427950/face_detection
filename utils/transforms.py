import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def label_to_onehot(class_id, n_classes, dtype=tf.float32):
    class_id = tf.cast(class_id, tf.int64)
    class_id = tf.expand_dims(class_id, axis=1)
    _n = K.int_shape(class_id)[0]
    index = tf.cast(range(_n), tf.int64)
    index = tf.expand_dims(index, axis=1)
    indices = tf.concat([index, class_id], axis=1)
    values = tf.cast([1] * _n, dtype=dtype)
    dense_shape = (_n, n_classes)
    sparse_label = tf.sparse.SparseTensor(indices, values, dense_shape)
    return tf.sparse.to_dense(sparse_label)

def normalize_image(images):
    images /= 127.5
    images -= 1.
    return images

def reize_with_pad(_image, _boxes, target_height, target_width):
    h1, w1, c1 = K.int_shape(_image)
    _image = tf.image.resize_with_pad(_image, target_height, target_width)
    if h1 / w1 < target_height / target_width :
        _boxes[:, 1] = (_boxes[:, 1] - 0.5) / (target_height * w1) * (h1 * target_width) + 0.5
        _boxes[:, 3] = (_boxes[:, 3] - 0.5) / (target_height * w1) * (h1 * target_width) + 0.5
    elif h1 / w1 >= target_height / target_width :
        _boxes[:, 0] = (_boxes[:, 0] - 0.5) / (target_width * h1) * (w1 * target_height) + 0.5
        _boxes[:, 2] = (_boxes[:, 2] - 0.5) / (target_width * h1) * (w1 * target_height) + 0.5

    return _image, _boxes

def restore_normalized_image_to01(images):
    images /= 2.
    images += 0.5
    return images

def random_shrink(image_, boxes_, minval):
    if minval >= 1.:
        return image_, boxes_
    h1, w1, c1 = K.int_shape(image_)
    image_ = tf.image.resize(image_,
                             (tf.cast(h1 * tf.random.uniform(shape=(), minval=minval, maxval=1.), dtype=tf.int16),
                              tf.cast(w1 * tf.random.uniform(shape=(), minval=minval, maxval=1.), dtype=tf.int16)))
    h2, w2, c2 = K.int_shape(image_)
    h_off = tf.cast(tf.random.uniform(shape=(), minval=0, maxval=h1 - h2), dtype=tf.int32)
    w_off = tf.cast(tf.random.uniform(shape=(), minval=0, maxval=w1 - w2), dtype=tf.int32)
    image_ = tf.image.pad_to_bounding_box(image_, h_off, w_off, h1, w1)

    h1 = K.eval(h1)
    w1 = K.eval(w1)
    h2 = K.eval(h2)
    w2 = K.eval(w2)
    h_off = K.eval(h_off)
    w_off = K.eval(w_off)
    boxes_[:, 0] = (boxes_[:, 0] * w2 + w_off - 1) / w1
    boxes_[:, 1] = (boxes_[:, 1] * h2 + h_off - 1) / h1
    boxes_[:, 2] = (boxes_[:, 2] * w2 + w_off - 1) / w1
    boxes_[:, 3] = (boxes_[:, 3] * h2 + h_off - 1) / h1

    return image_, boxes_

def random_crop(image_, boxes_, minval, maxval=1.):
    h1, w1, c1 = K.int_shape(image_)
    _image = image_
    _boxes = boxes_.copy()
    h2 = tf.cast(tf.random.uniform(shape=(), minval=minval, maxval= maxval) * h1, dtype=tf.int32)
    w2 = tf.cast(tf.random.uniform(shape=(), minval=minval, maxval= maxval) * w1, dtype=tf.int32)
    h_off = tf.cast(tf.random.uniform(shape=(), minval=0., maxval= tf.cast(h1 - h2, tf.float32)), dtype=tf.int32)
    w_off = tf.cast(tf.random.uniform(shape=(), minval=0., maxval= tf.cast(w1 - w2, tf.float32)), dtype=tf.int32)
    image_ = tf.image.crop_to_bounding_box(image_, offset_height=h_off, offset_width=w_off, target_height=h2,
                                           target_width=w2)
    image_ = tf.image.resize(image_, (h1, w1))

    h1 = K.eval(h1)
    w1 = K.eval(w1)
    h2 = K.eval(h2)
    w2 = K.eval(w2)
    h_off = K.eval(h_off)
    w_off = K.eval(w_off)
    boxes_[:, 0] = (boxes_[:, 0] * w1 - w_off + 1) / w2
    boxes_[:, 1] = (boxes_[:, 1] * h1 - h_off + 1) / h2
    boxes_[:, 2] = (boxes_[:, 2] * w1 - w_off + 1) / w2
    boxes_[:, 3] = (boxes_[:, 3] * h1 - h_off + 1) / h2
    boxes_ = np.clip(boxes_, 0., 1.)
    index1 = boxes_[:, 0] == boxes_[:, 2]
    index2 = boxes_[:, 1] == boxes_[:, 3]
    index = np.bitwise_or(index1, index2)
    boxes_ = np.delete(boxes_, index, axis=0)

    if len(boxes_) == 0:
        image_ = _image
        boxes_ = _boxes

    return image_, boxes_


def random_shift(image_, boxes_, maxval):
    h1, w1, c1 = K.int_shape(image_)
    _image = image_
    _boxes = boxes_.copy()
    h_off = tf.cast(tf.random.uniform(shape=(), minval=-maxval, maxval=maxval * h1), dtype=tf.int32)
    w_off = tf.cast(tf.random.uniform(shape=(), minval=-maxval, maxval=maxval * w1), dtype=tf.int32)
    h_off = K.eval(h_off)
    w_off = K.eval(w_off)
    h1 = K.eval(h1)
    w1 = K.eval(w1)
    h_off = K.eval(h_off)
    w_off = K.eval(w_off)
    if h_off > 0 and w_off > 0 :
        image_ = tf.image.crop_to_bounding_box(image_, offset_height=h_off, offset_width=w_off, target_height=h1 - h_off,
                                               target_width=w1 - w_off)
        image_ = tf.image.pad_to_bounding_box(image_, 0, 0, h1, w1)

    elif h_off > 0 and w_off <= 0 :
        image_ = tf.image.crop_to_bounding_box(image_, offset_height=h_off, offset_width=0, target_height=h1 - h_off,
                                               target_width=w1 - w_off)
        image_ = tf.image.pad_to_bounding_box(image_, 0, w_off, h1, w1)
    elif h_off <= 0 and w_off > 0:
        image_ = tf.image.crop_to_bounding_box(image_, offset_height=0, offset_width=w_off, target_height=h1 - h_off,
                                               target_width=w1 - w_off)
        image_ = tf.image.pad_to_bounding_box(image_, h_off, 0, h1, w1)
    elif h_off <= 0 and w_off <= 0:
        image_ = tf.image.crop_to_bounding_box(image_, offset_height=0, offset_width=0, target_height=h1 - h_off,
                                               target_width=w1 - w_off)
        image_ = tf.image.pad_to_bounding_box(image_, h_off, w_off, h1, w1)

    boxes_[:, 0] = boxes_[:, 0] - w_off / w1
    boxes_[:, 1] = boxes_[:, 1] - h_off / h1
    boxes_[:, 2] = boxes_[:, 2] - w_off / w1
    boxes_[:, 3] = boxes_[:, 3] - h_off / h1
    boxes_ = np.clip(boxes_, 0., 1.)
    index1 = boxes_[:, 0] == boxes_[:, 2]
    index2 = boxes_[:, 1] == boxes_[:, 3]
    index = np.bitwise_or(index1, index2)
    boxes_ = np.delete(boxes_, index, axis=0)

    if len(boxes_) == 0:
        image_ = _image
        boxes_ = _boxes


    return image_, boxes_



def _show_image_with_bboxes(image_, boxes_):
    plt.imshow(image_)
    current_axis = plt.gca()
    _h, _w, _ = K.int_shape(image_)
    for box in boxes_:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        box = K.eval(box)
        xmin = box[0] * _w
        ymin = box[1] * _h
        xmax = box[2] * _w
        ymax = box[3] * _h
        # label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=(0., 0.9, 0.), fill=False, linewidth=2))

if __name__ == '__main__':
    # label = [0, 3, 2, 1]
    # one_hot = label_to_onehot(label, 4)
    # print(one_hot)
    # image = tf.io.read_file('images/test_image_1.jpg')
    # image = tf.image.decode_jpeg(image)
    image = cv.imread('../images/test_image_1.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    boxes = np.array([[70, 200, 150, 300], [480, 200, 570, 300]], dtype=np.float)
    h, w, _ = K.int_shape(image)
    boxes[:, 0] = boxes[:, 0] / w
    boxes[:, 2] = boxes[:, 2] / w
    boxes[:, 1] = boxes[:, 1] / h
    boxes[:, 3] = boxes[:, 3] / h
    plt.subplot(1, 2, 1)
    _show_image_with_bboxes(image.numpy(), boxes)
    image, boxes = random_shift(image, boxes, 0.5)
    plt.subplot(1, 2, 2)
    _show_image_with_bboxes(image.numpy(), boxes)
    plt.show()


