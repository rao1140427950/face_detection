import abc
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class Model(metaclass=abc.ABCMeta):
    """
    A simple abstract class.
    Implement some useful functions.
    """

    def __init__(self, basemodel_names=None, model=None, basemodel=None, kernel_regularizer=None):
        if basemodel_names is None:
            basemodel_names = []
        self.model = model
        self.kernel_regularizer = kernel_regularizer
        self.basemodel_names = basemodel_names
        self.basemodel = basemodel
        self.ch_att_num = 0
        self.sp_att_num = 0

    def __call__(self, x):
        return self.model(x)

    def pool(self, inputs, method='max', pool_size=3, strides=1, padding='same', name=None):
        if method == 'max':
            return self.maxpool_layer(inputs, pool_size=pool_size, strides=strides, padding=padding, name=name)
        elif method == 'avg':
            return self.avgpool_layer(inputs, pool_size=pool_size, strides=strides, padding=padding, name=name)
        else:
            raise ValueError('Pooling method should be `avg` or `max` but get `{}`.'.format(method))

    def channel_attention(self, inputs, r=8, name_id=None):
        if name_id is None:
            name_id = self.ch_att_num
            self.ch_att_num += 1

        c = inputs.shape[3]
        shared_fc1 = tf.keras.layers.Dense(
            c // r,
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer='he_normal',
            activation='relu',
            name='chatt_shared_fc1_o_{}'.format(name_id)
        )
        shared_fc2 = tf.keras.layers.Dense(
            c,
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer='he_normal',
            name='chatt_shared_fc2_o_{}'.format(name_id)
        )

        avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='chatt_gap_o_{}'.format(name_id))(inputs)
        avg_pool = tf.keras.layers.Reshape((1, 1, c), name='chatt_gapreshape_o_{}'.format(name_id))(avg_pool)
        avg_pool = shared_fc1(avg_pool)
        avg_pool = shared_fc2(avg_pool)

        max_pool = tf.keras.layers.GlobalMaxPooling2D(name='chatt_gmp_o_{}'.format(name_id))(inputs)
        max_pool = tf.keras.layers.Reshape((1, 1, c), name='chatt_gmpreshape_o_{}'.format(name_id))(max_pool)
        max_pool = shared_fc1(max_pool)
        max_pool = shared_fc2(max_pool)

        outputs = tf.keras.layers.Add(name='chatt_add_o_{}'.format(name_id))([avg_pool, max_pool])
        outputs = tf.keras.layers.Activation('hard_sigmoid', name='chatt_act_o_{}'.format(name_id))(outputs)
        outputs = tf.keras.layers.Multiply(name='chatt_mul_o_{}'.format(name_id))([inputs, outputs])

        return outputs

    def spatial_attention(self, inputs, ksize=7, name_id=None):
        if name_id is None:
            name_id = self.sp_att_num
            self.sp_att_num += 1

        avg_pool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True),
                                          name='spatt_apool_o_{}'.format(name_id))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True),
                                          name='spatt_mpool_o_{}'.format(name_id))(inputs)
        concat = tf.keras.layers.Concatenate(axis=3, name='spatt_concat_o_{}'.format(name_id))([avg_pool, max_pool])
        outputs = tf.keras.layers.Conv2D(
            1,
            ksize,
            activation='hard_sigmoid',
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            name='spatt_conv_o_{}'.format(name_id)
        )(concat)
        outputs = tf.keras.layers.Multiply(name='spatt_mul_o_{}'.format(name_id))([inputs, outputs])

        return outputs

    def cbam_block(self, inputs, r=8, ksize=7, name_id=None):
        outputs = self.channel_attention(inputs, r, name_id)
        outputs = self.spatial_attention(outputs, ksize, name_id)

        return outputs

    def conv_layer(self, inputs, filters, kernel_size, strides=1, padding='same', activation='relu', name=None, bn=True):
        conv_name = None
        bn_name = None
        act_name = None
        if name is not None:
            conv_name = name + '_conv'
            bn_name = name + '_bn'
            act_name = name + '_' + activation
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            use_bias=False,
            kernel_regularizer=self.kernel_regularizer,
            name=conv_name
        )(inputs)
        if bn:
            x = tf.keras.layers.BatchNormalization(scale=False, name=bn_name)(x)
        if activation:
            x = tf.keras.layers.Activation(activation, name=act_name)(x)
        return x

    @staticmethod
    def maxpool_layer(inputs, pool_size=3, strides=1, padding='same', name=None):
        return tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)(inputs)

    @staticmethod
    def avgpool_layer(inputs, pool_size=3, strides=1, padding='same', name=None):
        return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)(inputs)

    def freeze_layers(self, layer_name):
        check = False
        for layer in self.model.layers:
            if check:
                layer.trainable = True
            else:
                layer.trainable = False
            if layer.name == layer_name:
                check = True
        return

    def freeze_base_layers(self):
        if self.basemodel is not None:
            for layer in self.basemodel.layers:
                layer.trainable = False
            print('Freeze base layers by basemodel.')
        else:
            for layer in self.model.layers:
                if layer.name in self.basemodel_names:
                    layer.trainable = False
                else:
                    layer.trainable = True
            print('Freeze base layers by basemodel_names.')
        return

    def release_all_layers(self):
        for layer in self.model.layers:
            layer.trainable = True
        return

    def summary(self):
        for layer in self.model.layers:
            print(layer.name, layer.trainable)
        return

    def save_weights_by_layer(self, weights_dir):
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        for layer in self.model.layers:
            w = layer.get_weights()
            n = layer.name
            p = os.path.join(weights_dir, n + '.npy')
            np.save(p, w)
            print('save', n + '.npy')
        return

    def load_weights_by_layer(self, weights_dir):
        for layer in self.model.layers:
            n = layer.name
            p = os.path.join(weights_dir, n + '.npy')
            if os.path.exists(p):
                w = np.load(p, allow_pickle=True)
                try:
                    layer.set_weights(w)
                except ValueError:
                    print('Failed to load ' + n + '.npy: ValueError occurred.')
                else:
                    print('load', n + '.npy')
            else:
                print('Failed to load ' + n + '.npy: Weights not found.')
        return

    def load_weights(self, weights_file, **kwargs):
        self.model.load_weights(weights_file, **kwargs)
        return

    def save_weights(self, weights_file, **kwargs):
        self.model.save_weights(weights_file, **kwargs)
        return

    def plot_model(self, filename):
        tf.keras.utils.plot_model(
            self.model,
            to_file = filename,
            show_shapes=True,
            show_layer_names=True
        )
        return

    def add_kernel_regularizer(self, kernel_regularizer=None):
        if kernel_regularizer is not None:
            self.kernel_regularizer = kernel_regularizer
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = self.kernel_regularizer
        return

    def print_json(self):
        print(self.model.to_json())
        return

    def save_json(self, filename):
        with open(filename, 'w') as f:
            f.write(self.model.to_json())
            f.close()
        return

    def print_layer_names(self):
        names = []
        for layer in self.model.layers:
            names.append(layer.name)
        print(names)
        return

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        return

