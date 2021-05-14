import numpy as np
import tensorflow as tf
import warnings
import os, sys
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
sys.path.append('..')
from models.ssd import SSD_ResNet50
from utils.box_utils import compute_target
from utils.anchor_utils import generate_anchors
import utils.transforms as trans
from config import SSD_CONFIG

def _show(image, label=None):
    plt.figure()
    plt.imshow(image)
    if label is not  None:
        plt.title(str(label.numpy()))
    plt.axis('off')
    plt.show()

class WiderFaceDataset:

    def __init__(
            self,
            encoder_config,
            txt_annos_path=None,
            image_root_dir=None,
            tfrecord_path=None,
            transform_config=None,
            argument=True,
            batch_size=8,
    ):
        self._txt_annos_path = txt_annos_path
        self._image_root_dir = image_root_dir
        self._tfrecord_path = tfrecord_path
        self._annos = None
        self._n_samples = 0
        self._image_size = encoder_config['image_size']

        self._argument = argument
        self._num_parallel_calls = 4
        self._batch_size = batch_size
        self._prefetch_buffer_size = 6
        self._index_dataset = None
        self._image_dataset = None
        self._batch_dataset = None

        self._anchors = generate_anchors(encoder_config)

        self._transform_config = {
            'brightness': 0.15,
            'hue': 0.05,
            'contrast': 0.15,
            'saturation': 0.15,
        }

        if transform_config is not None:
            self._transform_config.update(transform_config)

        self._image_feature_description = {
            'filepath': tf.io.FixedLenFeature([], tf.string),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'bboxes': tf.io.VarLenFeature(tf.float32),
        }


    def _read_txt_annos(self):
        if self._annos is not None:
            return self._annos

        assert self._txt_annos_path is not None
        assert self._image_root_dir is not None

        f = open(self._txt_annos_path, 'r')
        lines = f.readlines()
        f.close()
        annos = []
        while len(lines) > 0:
            filename = lines.pop(0)[:-1]
            n_boxes = eval(lines.pop(0))
            boxes = []
            if n_boxes == 0:
                lines.pop(0)
                warnings.warn('Image `{}` has no bbox.'.format(filename))
                continue
            for n in range(n_boxes):
                bbox = lines.pop(0).split(' ')
                if eval(bbox[2]) > 0 and eval(bbox[3]) > 0:
                    boxes.append([eval(bbox[0]), eval(bbox[1]), eval(bbox[2]), eval(bbox[3])])
                else:
                    warnings.warn('File `{}` has bbox which has 0 width or height.'.format(filename))
            if len(boxes) == 0:
                warnings.warn('File `{}` has no valid bbox.'.format(filename))
                continue
            filename = os.path.join(self._image_root_dir, filename)
            if os.path.exists(filename):
                annos.append([filename, boxes])
            else:
                warnings.warn('File `{}` not found.'.format(filename))
        self._annos = annos
        self._n_samples = len(annos)

        return annos

    # Data argumentation
    def _transforms(self, image, boxes, class_id):
        image = tf.image.random_brightness(image, self._transform_config['brightness'])
        image = tf.image.random_hue(image, self._transform_config['hue'])
        image = tf.image.random_contrast(
            image, 1. - self._transform_config['contrast'], 1. + self._transform_config['contrast'])
        image = tf.image.random_saturation(
            image, 1. - self._transform_config['saturation'], 1. + self._transform_config['saturation']
        )
        return image, boxes, class_id

    @staticmethod
    def _encode_image(image):
        return tf.image.encode_jpeg(image)

    @staticmethod
    def _decode_image_raw(image_raw):
        return tf.image.decode_jpeg(image_raw)

    def _process_single_image_with_bboxes(self, image, boxes):
        h, w, c = K.int_shape(image)
        n_boxes = len(boxes)
        class_id = np.ones((n_boxes,), dtype=np.float32)
        boxes = np.array(boxes, dtype=np.float32)
        # (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes -= 1
        # normalize coords
        boxes[:, 0] = boxes[:, 0] / w
        boxes[:, 2] = boxes[:, 2] / w
        boxes[:, 1] = boxes[:, 1] / h
        boxes[:, 3] = boxes[:, 3] / h

        # convert image type
        image = tf.cast(image, tf.float32)
        # normalize image
        image = trans.normalize_image(image)

        if self._argument:
            image, boxes, class_id = self._transforms(image, boxes, class_id)

        image = tf.image.resize(image, self._image_size)
        class_id, boxes = compute_target(self._anchors, boxes, class_id)

        # To onehot
        class_id = tf.one_hot(tf.cast(class_id, tf.int32), 2, dtype=tf.float32)

        boxes = tf.cast(boxes, tf.float32)

        return image, tf.concat([class_id, boxes], axis=1)


    def _map_single_index(self, index):

        def py_map_single_index(p: tf.Tensor):
            p = K.eval(p)
            filepath, boxes = self._annos[p]
            image_raw = tf.io.read_file(filepath)
            image = self._decode_image_raw(image_raw)
            return self._process_single_image_with_bboxes(image, boxes)

        imag, bbox = tf.py_function(py_map_single_index, inp=[index], Tout=(tf.float32, tf.float32))
        return tf.cast(imag, tf.float32), tf.cast(bbox, tf.float32)

    def generate_dataset_from_imagefile(self):
        self._read_txt_annos()
        self._index_dataset = tf.data.Dataset.range(self._n_samples).shuffle(self._n_samples)
        self._image_dataset = self._index_dataset.map(self._map_single_index,
                                                      num_parallel_calls=self._num_parallel_calls)
        self._batch_dataset = self._image_dataset.batch(self._batch_size).prefetch(self._prefetch_buffer_size)
        return self._batch_dataset


if __name__ == '__main__':
    ssd = SSD_ResNet50()
    dataset = WiderFaceDataset(
        SSD_CONFIG,
        txt_annos_path='/home/raosj/datasets/wider_face/wider_face_split/wider_face_val_bbx_gt.txt',
        image_root_dir='/home/raosj/datasets/wider_face/WIDER_val/images',
        tfrecord_path='/home/raosj/datasets/wider_face/wider_val.tfrecords',
    )
    # dataset.read_txt_annos()
    samples = dataset.generate_dataset_from_imagefile()
    # dataset.imagefile_to_tfrecords()
    # samples = dataset.generate_dataset_from_tfrecords()
    cnt = 0
    for sample in samples:
        image0, label0 = sample
        print(cnt)
        cnt += 1
        for img, lb in zip(image0, label0):
            img = img.numpy()
            lb = lb.numpy()
            # print(img.min(), img.max())
            print(np.shape(lb))
            img = trans.restore_normalized_image_to01(img)
            # img += 0.5
            _show(img)
