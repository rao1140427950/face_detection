import abc
import os
import numpy as np
import tensorflow as tf


class Model(metaclass=abc.ABCMeta):
    """
    A simple abstract class.
    Implement some useful functions.
    """

    def __init__(self, basemodel_names=None, model=None, basemodel=None, kernel_regularizer=None, r=4):
        if basemodel_names is None:
            basemodel_names = []
        self.model = model
        self.kernel_regularizer = kernel_regularizer
        self.basemodel_names = basemodel_names
        self.basemodel = basemodel
        self.r = r

    def __call__(self, x):
        return self.model(x)

    def pool(self, inputs, method='max', pool_size=3, strides=1, padding='same', name=None):
        if method == 'max':
            return self.maxpool_layer(inputs, pool_size=pool_size, strides=strides, padding=padding, name=name)
        elif method == 'avg':
            return self.avgpool_layer(inputs, pool_size=pool_size, strides=strides, padding=padding, name=name)
        else:
            raise ValueError('Pooling method should be `avg` or `max` but get `{}`.'.format(method))

    def se_block(self, inputs, name_id):
        r = self.r
        c = inputs.shape[3]
        x = tf.keras.layers.GlobalAveragePooling2D(name='se_gap_o_' + name_id)(inputs)
        x = tf.keras.layers.Dense(
            int(c/r),
            activation='relu',
            kernel_regularizer=self.kernel_regularizer,
            name='se_fc1_o_' + name_id
        )(x)
        x = tf.keras.layers.Dense(
            c,
            activation='sigmoid',
            kernel_regularizer=self.kernel_regularizer,
            name='se_fc2_o_' + name_id
        )(x)

        return tf.keras.layers.Multiply(name='se_multi_o_' + name_id)([inputs, x])

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

