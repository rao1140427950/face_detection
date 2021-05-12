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
        self._read_buffer_size = 256 * (1024 * 1024)
        self._tfrecords_shuffle_buffer_size = 512
        self._batch_size = batch_size
        self._prefetch_buffer_size = 16
        self._index_dataset = None
        self._image_dataset = None
        self._batch_dataset = None
        self._raw_image_dataset = None
        self._shuffle_image_dataset = None
        self._parsed_image_dataset = None
        self._decoded_image_dataset = None

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
            filename = os.path.join(self._image_root_dir, filename)
            if os.path.exists(filename):
                annos.append([filename, boxes])
            else:
                warnings.warn('File `{}` not found.'.format(filename))
        self._annos = annos
        self._n_samples = len(annos)

        return annos

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

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _int64_list_feature(list_value):
        """Returns an int64_list from a bool / enum / int / uint list."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list_value))

    @staticmethod
    def _float_list_feature(list_value):
        """Returns an float_list from a bool / enum / int / uint list."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=list_value))

    def _process_single_image_with_bboxes(self, image, boxes):
        h, w, c = K.int_shape(image)
        n_boxes = len(boxes)
        class_id = np.ones((n_boxes, 1), dtype=np.float)
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
            image, boxes, class_id = self._transforms(image, boxes, class_id)

        image = tf.image.resize(image, self._image_size)

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

        return image, gt_encoded

    def _map_single_index(self, index):

        def py_map_single_index(p: tf.Tensor):
            p = K.eval(p)
            filepath, boxes = self._annos[p]
            image_raw = tf.io.read_file(filepath)
            image = self._decode_image_raw(image_raw)
            return self._process_single_image_with_bboxes(image, boxes)

        imag, gt = tf.py_function(py_map_single_index, inp=[index], Tout=(tf.float32, tf.float32))
        return tf.cast(imag, tf.float32), tf.cast(gt, tf.float32)

    def _map_single_example(self, image_features):
        image_raw = image_features['image_raw']
        boxes = image_features['bboxes'].values
        image = self._decode_image_raw(image_raw)

        def py_map_single_example(_image, _boxes):
            _boxes = K.eval(_boxes).reshape((-1, 4))
            return self._process_single_image_with_bboxes(_image, _boxes)

        imag, gt = tf.py_function(py_map_single_example, inp=[image, boxes], Tout=(tf.float32, tf.float32))
        return tf.cast(imag, tf.float32), tf.cast(gt, tf.float32)

    def _image_example(self, image_bytes, bboxes, filepath_bytes):
        feature = {
            'filepath': self._bytes_feature(filepath_bytes),
            'image_raw': self._bytes_feature(image_bytes),
            'bboxes': self._float_list_feature(bboxes),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _parse_image_function(self, example_proto):
        return tf.io.parse_single_example(example_proto, self._image_feature_description)

    def imagefile_to_tfrecords(self):
        self._read_txt_annos()
        assert self._tfrecord_path is not None
        with tf.io.TFRecordWriter(self._tfrecord_path) as writer:
            for anno in self._annos:
                filepath, boxes  = anno
                image_string = open(filepath, 'rb').read()
                bboxes = np.array(boxes, dtype=np.float32).reshape((-1))
                tf_example = self._image_example(image_string, bboxes, filepath.encode())
                writer.write(tf_example.SerializeToString())
                print('Image', filepath, 'processed.')


    def generate_dataset_from_tfrecords(self):
        if self._tfrecord_path is None:
            raise ValueError('TFRecord path can not be None.')
        elif isinstance(self._tfrecord_path, list):
            for p in self._tfrecord_path:
               if not os.path.exists(p):
                   raise ValueError('TFRecord path `%s` does not exist.' % p)
            tpath = self._tfrecord_path
        else:
            if not os.path.exists(self._tfrecord_path):
                raise ValueError('TFRecord path `%s` does not exist.' % self._tfrecord_path)
            tpath = [self._tfrecord_path]

        self._raw_image_dataset = tf.data.TFRecordDataset(tpath, buffer_size=self._read_buffer_size)
        self._shuffle_image_dataset = self._raw_image_dataset.shuffle(self._tfrecords_shuffle_buffer_size)
        self._parsed_image_dataset = self._shuffle_image_dataset.map(self._parse_image_function)
        self._decoded_image_dataset = self._parsed_image_dataset.map(self._map_single_example,
                                                                   num_parallel_calls=self._num_parallel_calls)
        self._batch_dataset = self._decoded_image_dataset.batch(self._batch_size).prefetch(self._prefetch_buffer_size)
        return self._batch_dataset



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
        ssd.get_config(),
        txt_annos_path='/home/raosj/datasets/wider_face/wider_face_split/wider_face_val_bbx_gt.txt',
        image_root_dir='/home/raosj/datasets/wider_face/WIDER_val/images',
        tfrecord_path='/home/raosj/datasets/wider_face/wider_val.tfrecords',
    )
    # dataset.read_txt_annos()
    # samples = dataset.generate_dataset_from_imagefile()
    # dataset.imagefile_to_tfrecords()
    samples = dataset.generate_dataset_from_tfrecords()
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
            img /= 2.0
            img += 0.5
            # _show(img)
