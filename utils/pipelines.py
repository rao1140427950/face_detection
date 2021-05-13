import tensorflow as tf
import warnings
import os, sys
import numpy as np
import cv2 as cv
import math
from random import shuffle
import matplotlib.pyplot as plt
sys.path.append('..')
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

class WiderFacePipeline(tf.keras.utils.Sequence):

    def __init__(
            self,
            encoder_config,
            txt_annos_path,
            image_root_dir,
            transform_config=None,
            argument=True,
            batch_size=0,
    ):

        self._txt_annos_path = txt_annos_path
        self._image_root_dir = image_root_dir
        self._annos = []
        self._n_samples = 0
        self._image_size = encoder_config['image_size']

        self._argument = argument
        self._batch_size = batch_size

        self._transform_config = {
            'brightness': 0.15,
            'hue': 0.05,
            'contrast': 0.15,
            'saturation': 0.15,
        }

        if transform_config is not None:
            self._transform_config.update(transform_config)

        self._anchors = generate_anchors(encoder_config)
        self._read_txt_annos()
        shuffle(self._annos)


    def _read_txt_annos(self):
        if len(self._annos) > 0:
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
            filename = os.path.join(self._image_root_dir, filename)
            if os.path.exists(filename):
                annos.append([filename, boxes])
            else:
                warnings.warn('File `{}` not found.'.format(filename))
        self._annos = annos
        self._n_samples = len(annos)

        return annos

    def __len__(self):
        if self._batch_size > 0:
            return math.ceil(len(self._annos) / self._batch_size)
        else:
            return len(self._annos)

    def on_epoch_end(self):
        shuffle(self._annos)

    def __getitem__(self, idx):
        if self._batch_size > 0:
            batch_annos = self._annos[idx * self._batch_size:(idx + 1) * self._batch_size]
            batch_images = []
            batch_gts = []
            for anno in batch_annos:
                _image, _gt = self._get_single_item(anno)
                batch_images.append(tf.expand_dims(_image, axis=0))
                batch_gts.append(tf.expand_dims(_gt, axis=0))

            return tf.concat(batch_images, axis=0), tf.concat(batch_gts, axis=0)
        else:
            anno = self._annos[idx]
            return self._get_single_item(anno)

    def _transforms(self, image, boxes, class_id):
        image = tf.image.random_brightness(image, self._transform_config['brightness'])
        image = tf.image.random_hue(image, self._transform_config['hue'])
        image = tf.image.random_contrast(
            image, 1. - self._transform_config['contrast'], 1. + self._transform_config['contrast'])
        image = tf.image.random_saturation(
            image, 1. - self._transform_config['saturation'], 1. + self._transform_config['saturation']
        )
        return image, boxes, class_id

    def _get_single_item(self, anno):
        _filepath, _boxes = anno
        _image = cv.imread(_filepath)
        _image = cv.cvtColor(_image, cv.COLOR_BGR2RGB)

        h, w, c = np.shape(_image)
        n_boxes = len(_boxes)
        class_id = np.ones((n_boxes,), dtype=np.float32)
        boxes = np.array(_boxes, dtype=np.float32)
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
        _image = _image.astype(np.float32)
        # normalize image
        image = trans.normalize_image(_image)

        if self._argument:
            image, boxes, class_id = self._transforms(image, boxes, class_id)

        image = tf.image.resize(image, self._image_size)
        class_id, boxes = compute_target(self._anchors, boxes, class_id)

        # To onehot
        class_id = tf.one_hot(tf.cast(class_id, tf.int32), 2, dtype=tf.float32)

        boxes = tf.cast(boxes, tf.float32)

        return image, tf.concat([class_id, boxes], axis=1)


def generate_dataset(
        config,
        txt_annos_path,
        image_root_dir,
        argument=False,
        batch_size=8,
):
    generater = WiderFacePipeline(
        config,
        txt_annos_path=txt_annos_path,
        image_root_dir=image_root_dir,
        argument=argument
    )
    return tf.data.Dataset.from_generator(
        generater
    ).batch(batch_size).prefetch(8)


if __name__ == '__main__':
    # ssd = SSD_ResNet50()
    # samples = WiderFacePipeline(
    #     SSD_CONFIG,
    #     txt_annos_path='/home/raosj/datasets/wider_face/wider_face_split/wider_face_val_bbx_gt.txt',
    #     image_root_dir='/home/raosj/datasets/wider_face/WIDER_val/images',
    #     argument=False
    # )
    samples = generate_dataset(
        SSD_CONFIG,
        txt_annos_path='/home/raosj/datasets/wider_face/wider_face_split/wider_face_val_bbx_gt.txt',
        image_root_dir='/home/raosj/datasets/wider_face/WIDER_val/images',
        argument=False,
        batch_size=8,
    )

    for sample in samples:
        image0, label0 = samples
        for img, lb in zip(image0, label0):
            # print(img.min(), img.max())
            print(np.shape(lb))
            img /= 2.0
            img += 0.5
            _show(img)
