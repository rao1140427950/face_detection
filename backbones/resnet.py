import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, ZeroPadding2D, Add
import tensorflow.keras.backend as K
import sys
sys.path.append('..')
from utils.model import Model


class ResNet(Model):

    @staticmethod
    def identity_block(input_tensor, kernel_size, filters, block):

        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        x = Conv2D(filters1, (1, 1), name=block + '_1_conv', kernel_initializer='he_normal')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=block + '_1_bn')(x)
        x = Activation('relu', name=block + '_1_relu')(x)

        x = Conv2D(filters2, kernel_size, kernel_initializer='he_normal',
                   padding='same', name=block + '_2_conv')(x)
        x = BatchNormalization(axis=bn_axis, name=block + '_2_bn')(x)
        x = Activation('relu', name=block + '_2_relu')(x)

        x = Conv2D(filters3, (1, 1), name=block + '_3_conv', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=bn_axis, name=block + '_3_bn')(x)

        x = Add(name=block + '_add')([x, input_tensor])
        x = Activation('relu', name=block + '_out')(x)
        return x

    @staticmethod
    def conv_block(input_tensor, kernel_size, filters, block, strides=(2, 2)):

        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal',
                   name=block + '_1_conv')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=block + '_1_bn')(x)
        x = Activation('relu', name=block + '_1_relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal',
                   name=block + '_2_conv')(x)
        x = BatchNormalization(axis=bn_axis, name=block + '_2_bn')(x)
        x = Activation('relu', name=block + '_2_relu')(x)

        x = Conv2D(filters3, (1, 1), name=block + '_3_conv', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=bn_axis, name=block + '_3_bn')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal',
                          name=block + '_0_conv')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=block + '_0_bn')(shortcut)

        x = Add(name=block + '_add')([x, shortcut])
        x = Activation('relu', name=block + '_out')(x)
        return x


    def __init__(self,
                 input_shape=(224, 224, 3),
                 kernel_regularizer=l2(0.0008),
                 repetitions=(3, 4, 6, 3),
                 out_layer_names='auto'
                 ):

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        if out_layer_names is None:
            out_layer_names = 'auto'

        in_layer = Input(shape=input_shape, name='input_image')

        features = []

        x = ZeroPadding2D((3, 3), name='conv1_pad')(in_layer)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1_conv', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=bn_axis, name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)
        x = ZeroPadding2D((1, 1), name='pool1_pad')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_pool')(x)

        x = self.conv_block(x, 3, [64, 64, 256], block='conv2_block1', strides=(1, 1))
        for r in range(repetitions[0] - 1):
            x = self.identity_block(x, 3, [64, 64, 256], block='conv2_block' + str(r + 2))

        if out_layer_names == 'auto':
            features.append(x)

        x = self.conv_block(x, 3, [128, 128, 512], block='conv3_block1')
        for r in range(repetitions[1] - 1):
            x = self.identity_block(x, 3, [128, 128, 512], block='conv3_block' + str(r + 2))

        if out_layer_names == 'auto':
            features.append(x)

        x = self.conv_block(x, 3, [256, 256, 1024], block='conv4_block1')
        for r in range(repetitions[2] - 1):
            x = self.identity_block(x, 3, [256, 256, 1024], block='conv4_block' + str(r + 2))

        if out_layer_names == 'auto':
            features.append(x)

        x = self.conv_block(x, 3, [512, 512, 2048], block='conv5_block1')
        for r in range(repetitions[3] - 1):
            x = self.identity_block(x, 3, [512, 512, 2048], block='conv5_block' + str(r + 2))

        if out_layer_names == 'auto':
            features.append(x)

        model = tf.keras.models.Model(inputs=in_layer, outputs=x, name='resnet')
        if out_layer_names == 'auto':
            model = tf.keras.models.Model(inputs=in_layer, outputs=features, name='resnet_backbone')
        else:
            for name in out_layer_names:
                features.append(model.get_layer(name=name).output)
            model = tf.keras.models.Model(inputs=in_layer, outputs=features, name='resnet_backbone')

        super().__init__(basemodel_names=[], model=model, kernel_regularizer=kernel_regularizer)
        # self.add_kernel_regularizer(kernel_regularizer)


if __name__ == '__main__':
    resnet = ResNet()
    # resnet.model.summary()
    resnet.plot_model('resnet50_backbone.png')