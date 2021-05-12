import numpy as np
# import tensorflow_io as tfio
import tensorflow as tf
import warnings
import os, sys
import tensorflow.keras.backend as K
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

class WiderFaceDataset:

    def __init__(
            self,
            txt_annos_path,
            image_root_dir,
            encoder_config,
            transform_config=None,
            argument=True,
            batch_size=8,
    ):
        self._txt_annos_path = txt_annos_path
        self._image_root_dir = image_root_dir
        self._annos = None
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
        self._num_parallel_calls = 4
        self._batch_size = batch_size
        self._prefetch_buffer_size = 4
        self._index_dataset = None
        self._image_dataset = None
        self._batch_dataset = None

        self._transform_config = {
            'brightness': 0.15,
            'hue': 0.05,
            'contrast': 0.15,
            'saturation': 0.15,
        }

        if transform_config is not None:
            self._transform_config.update(transform_config)

        self._read_txt_annos()

    def _read_txt_annos(self):
        if self._annos is not None:
            return self._annos
        f = open(self._txt_annos_path, 'r')
        lines = f.readlines()
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

    def _transforms(self, image, boxes):
        image = tf.image.random_brightness(image, self._transform_config['brightness'])
        image = tf.image.random_hue(image, self._transform_config['hue'])
        image = tf.image.random_contrast(
            image, 1. - self._transform_config['contrast'], 1. + self._transform_config['contrast'])
        image = tf.image.random_saturation(
            image, 1. - self._transform_config['saturation'], 1. + self._transform_config['saturation']
        )
        return image, boxes

    @staticmethod
    def _encode_image(image):
        return tf.image.encode_jpeg(image)

    @staticmethod
    def _decode_image_raw(image_raw):
        return tf.image.decode_jpeg(image_raw)

    def _map_single_index(self, index):

        def py_map_single_index(p: tf.Tensor):
            p = K.eval(p)
            filepath, boxes = self._annos[p]
            image_raw = tf.io.read_file(filepath)
            image = self._decode_image_raw(image_raw)
            h, w, c = K.int_shape(image)

            n_boxes = len(boxes)
            boxes = np.array(boxes, dtype=np.float)
            # (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            # normalize coords
            boxes[:, 0] = boxes[:, 0] / w
            boxes[:, 2] = boxes[:, 2] / w
            boxes[:, 1] = boxes[:, 1] / h
            boxes[:, 3] = boxes[:, 3] / h

            # convert image type
            image = tf.cast(image, tf.float32)
            # normalize image
            image /= 255.0
            image -= 0.5
            image *= 2

            if self._argument:
                image, boxes = self._transforms(image, boxes)
            image = tf.image.resize(image, self._image_size)

            # (cx, cy, w, h) to (class_id, cx, cy, w, h)
            class_id = np.ones((n_boxes, 1), dtype=np.float)
            gt_labels = np.concatenate([class_id, boxes], axis=1)
            gt_encoded = self._input_encoder([gt_labels])[0]

            return image, gt_encoded

        imag, gt = tf.py_function(py_map_single_index, inp=[index], Tout=(tf.float32, tf.float32))
        return tf.cast(imag, tf.float32), tf.cast(gt, tf.float32)


    def generate_dataset(self):
        self._index_dataset = tf.data.Dataset.range(self._n_samples).shuffle(self._n_samples)
        self._image_dataset = self._index_dataset.map(self._map_single_index,
                                                      num_parallel_calls=self._num_parallel_calls)
        self._batch_dataset = self._image_dataset.batch(self._batch_size).prefetch(self._prefetch_buffer_size)
        return self._batch_dataset




if __name__ == '__main__':
    ssd = SSD_ResNet50()
    dataset = WiderFaceDataset(
        '/home/raosj/datasets/wider_face/wider_face_split/wider_face_train_bbx_gt.txt',
        '/home/raosj/datasets/wider_face/WIDER_train/images',
        ssd.get_config(),
    )
    # dataset.read_txt_annos()
    samples = dataset.generate_dataset()
    for sample in samples:
        image0, label0 = sample
        for img, lb in zip(image0, label0):
            img = img.numpy()
            lb = lb.numpy()
            # print(img.min(), img.max())
            print(np.shape(lb))
            img /= 2.0
            img += 0.5
            _show(img)
