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

def restore_normalized_image_to01(images):
    images /= 2.
    images += 0.5
    return images


if __name__ == '__main__':
    label = [0, 3, 2, 1]
    one_hot = label_to_onehot(label, 4)
    print(one_hot)