import tensorflow as tf
import warnings
import os, sys
import numpy as np
import cv2 as cv
import math
from random import shuffle
import matplotlib.pyplot as plt
sys.path.append('..')
from utils.encode_box_utils import SSDInputEncoder
from models.ssd import SSD_ResNet50


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
            txt_annos_path=None,
            image_root_dir=None,
            transform_config=None,
            argument=True,
            batch_size=8,
    ):

        self._txt_annos_path = txt_annos_path
        self._image_root_dir = image_root_dir
        self._annos = []
        self._n_samples = 0
        self._image_size = encoder_config['image_size']
        self._input_encoder = SSDInputEncoder(
            img_height=encoder_config['img_height'],
            img_width=encoder_config['img_width'],
            n_classes=encoder_config['n_classes'],
            predictor_sizes=encoder_config['predictor_sizes'],
            min_scale=encoder_config['min_scale'],
            max_scale=encoder_config['max_scale'],
            scales=encoder_config['scales'],
            aspect_ratios_per_layer=encoder_config['aspect_ratios_per_layer'],
            two_boxes_for_ar1=encoder_config['two_boxes_for_ar1'],
            steps=encoder_config['steps'],
            offsets=encoder_config['offsets'],
            clip_boxes=encoder_config['clip_boxes'],
            variances=encoder_config['variances'],
            coords=encoder_config['coords'],
            normalize_coords=encoder_config['normalize_coords'],
        )
        self._output_label_shape = encoder_config['output_sizes']

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
        return math.ceil(len(self._annos) / self._batch_size)

    def on_epoch_end(self):
        shuffle(self._annos)

    def __getitem__(self, idx):
        batch_annos = self._annos[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_images = []
        batch_gts = []
        for anno in batch_annos:
            _image, _gt = self._get_single_item(anno)
            batch_images.append(np.expand_dims(_image.astype(np.float32), axis=0))
            batch_gts.append(np.expand_dims(_gt.astype(np.float32), axis=0))

        return np.concatenate(batch_images, axis=0), np.concatenate(batch_gts, axis=0)

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
        class_id = np.ones((n_boxes, 1), dtype=np.float)
        boxes = np.array(_boxes, dtype=np.float)
        # (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        # normalize coords
        boxes[:, 0] = boxes[:, 0] / w
        boxes[:, 2] = boxes[:, 2] / w
        boxes[:, 1] = boxes[:, 1] / h
        boxes[:, 3] = boxes[:, 3] / h

        # convert image type
        _image = _image.astype(np.float)
        # normalize image
        _image /= 255.0
        _image -= 0.5
        _image *= 2

        if self._argument:
            _image, boxes, class_id = self._transforms(_image, boxes, class_id)

        _image = cv.resize(_image, self._image_size, interpolation=cv.INTER_LINEAR)

        # restore the coords
        h, w = self._image_size
        boxes[:, 0] = boxes[:, 0] * w
        boxes[:, 2] = boxes[:, 2] * w
        boxes[:, 1] = boxes[:, 1] * h
        boxes[:, 3] = boxes[:, 3] * h

        # (xmin, ymin, xmax, ymax) to (xmin, ymin, xmax, ymax)
        gt_labels = np.concatenate([class_id, boxes], axis=1)
        gt_encoded = self._input_encoder([gt_labels])[0]
        # for n in range(10):
        #     gt_encoded = self._input_encoder([gt_labels])[0]
        #     print(n)

        return _image, gt_encoded



if __name__ == '__main__':
    ssd = SSD_ResNet50()
    samples = WiderFacePipeline(
        ssd.get_config(),
        txt_annos_path='/home/raosj/datasets/wider_face/wider_face_split/wider_face_val_bbx_gt.txt',
        image_root_dir='/home/raosj/datasets/wider_face/WIDER_val/images',
        argument=False
    )

    image0, label0 = samples[0]
    for img, lb in zip(image0, label0):
        # print(img.min(), img.max())
        print(np.shape(lb))
        img /= 2.0
        img += 0.5
        _show(img)
