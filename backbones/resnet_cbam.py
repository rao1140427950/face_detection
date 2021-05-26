import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, ZeroPadding2D, Add
import tensorflow.keras.backend as K
import sys
sys.path.append('..')
from utils.model import Model


class ResNet_CBAM(Model):
    """
    model name | repetitions
    resnet34   | (2, 2, 2, 2)
    resnet50   | (3, 4, 6, 3)
    resnet101  | (3, 4, 23, 3)
    resnet152  | (3, 8, 36, 3)

    The implementation is compatible of tf.keras.applications.resnet
    Layer names are the same
    """

    def __init__(self,
                 input_shape=(224, 224, 3),
                 kernel_regularizer=l2(0.0008),
                 repetitions=(3, 4, 6, 3),
                 ):

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        self.ch_att_num = 0
        self.sp_att_num = 0
        self.kernel_regularizer = kernel_regularizer

        in_layer = Input(shape=input_shape, name='input_image')

        features = []

        x = ZeroPadding2D((3, 3), name='conv1_pad')(in_layer)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1_conv', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=bn_axis, name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)

        # feature = self.conv_layer(x, 64, 3)
        # feature = self.conv_layer(feature, 64, 3)
        # features.append(feature)

        x = ZeroPadding2D((1, 1), name='pool1_pad')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_pool')(x)

        x = self.conv_block(x, 3, [64, 64, 256], block='conv2_block1', strides=(1, 1))
        for r in range(repetitions[0] - 2):
            x = self.identity_block(x, 3, [64, 64, 256], block='conv2_block' + str(r + 2))
        x = self.identity_block(x, 3, [64, 64, 256], block='conv2_block' + str(repetitions[0]), cbma=False)

        features.append(x)

        x = self.conv_block(x, 3, [128, 128, 512], block='conv3_block1')
        for r in range(repetitions[1] - 2):
            x = self.identity_block(x, 3, [128, 128, 512], block='conv3_block' + str(r + 2))
        x = self.identity_block(x, 3, [128, 128, 512], block='conv3_block' + str(repetitions[1]), cbma=False)

        features.append(x)

        x = self.conv_block(x, 3, [256, 256, 1024], block='conv4_block1')
        for r in range(repetitions[2] - 2):
            x = self.identity_block(x, 3, [256, 256, 1024], block='conv4_block' + str(r + 2))
        x = self.identity_block(x, 3, [256, 256, 1024], block='conv4_block' + str(repetitions[2]), cbma=False)

        features.append(x)

        x = self.conv_block(x, 3, [512, 512, 2048], block='conv5_block1')
        for r in range(repetitions[3] - 2):
            x = self.identity_block(x, 3, [512, 512, 2048], block='conv5_block' + str(r + 2))
        x = self.identity_block(x, 3, [512, 512, 2048], block='conv5_block' + str(repetitions[3]), cbma=False)

        features.append(x)

        model = tf.keras.models.Model(inputs=in_layer, outputs=features, name='resnet_backbone')

        super().__init__(basemodel_names=[], model=model, kernel_regularizer=kernel_regularizer)
        # self.add_kernel_regularizer(kernel_regularizer)

    def identity_block(self, input_tensor, kernel_size, filters, block, cbma=True):

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
        if cbma:
            x = self.cbam_block(x)

        x = Add(name=block + '_add')([x, input_tensor])
        x = Activation('relu', name=block + '_out')(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, block, strides=(2, 2), cmba=True):

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
        if cmba:
            x = self.cbam_block(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal',
                          name=block + '_0_conv')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=block + '_0_bn')(shortcut)


        x = Add(name=block + '_add')([x, shortcut])
        x = Activation('relu', name=block + '_out')(x)
        return x



if __name__ == '__main__':
    resnet = ResNet_CBAM()
    # resnet.model.summary()
    # resnet.plot_model('resnet50_cbam_backbone.png')