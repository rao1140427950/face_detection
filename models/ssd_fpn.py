import tensorflow as tf
import sys
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Reshape, Concatenate, Activation, UpSampling2D, Add
from tensorflow.keras.backend import int_shape
sys.path.append('..')
from utils.model import Model
from backbones.resnet_cbam import ResNet_CBAM
from models.ssd import SSDHead


class FPN(Model):

    def __init__(self,
                 backbone,
                 n_dims=128,
                 kernel_regularizer=None):

        self.kernel_regularizer = kernel_regularizer
        input_x = backbone.model.input
        predict_layers = backbone.model.outputs
        n_predictor_layers = len(predict_layers)

        for n in range(n_predictor_layers):
            predict_layers[n] = self.conv_layer(predict_layers[n], n_dims, 1)

        predict_layers.reverse()
        for n in range(n_predictor_layers - 1):
            x = UpSampling2D()(predict_layers[n])
            predict_layers[n + 1] = Add()([predict_layers[n + 1], x])
        predict_layers.reverse()

        for n in range(n_predictor_layers):
            predict_layers[n] = self.conv_layer(predict_layers[n], n_dims, 3)

        model = tf.keras.models.Model(inputs=input_x, outputs=predict_layers, name='fpn_head')
        super().__init__(basemodel=backbone.model, model=model, kernel_regularizer=kernel_regularizer)

class SSDFPN_ResNet_CBAM(SSDHead):

    def __init__(
            self,
            input_shape=(512, 512, 3),
            repetitions=(3, 4, 6, 3),
            kernel_regularizer=l2(0.0008),
            config=None,
    ):
        aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
                                   [1.0, 2.0, 0.5],
                                   [1.0, 2.0, 0.5],
                                   [1.0, 2.0, 0.5],
                                   [1.0, 2.0, 0.5]]
        resnet = ResNet_CBAM(
            input_shape=input_shape,
            kernel_regularizer=kernel_regularizer,
            repetitions=repetitions,
        )
        fpn = FPN(
            backbone=resnet,
            n_dims=128,
            kernel_regularizer=kernel_regularizer,
        )
        super().__init__(
            backbone=fpn,
            n_classes=1,
            aspect_ratios_per_layer=aspect_ratios_per_layer,
            l2_regularization=0.0005,
            two_boxes_for_ar1=True,
            config=config,
        )

class SSDFPN_ResNet50_CBAM(SSDFPN_ResNet_CBAM):

    def __init__(
            self,
            input_shape=(512, 512, 3),
            kernel_regularizer=l2(0.0008),
            config=None,
    ):
        super().__init__(
            input_shape=input_shape,
            repetitions=(3, 4, 6, 3),
            kernel_regularizer=kernel_regularizer,
            config=config
        )



class SSDFPN_ResNet101_CBAM(SSDFPN_ResNet_CBAM):

    def __init__(
            self,
            input_shape=(512, 512, 3),
            kernel_regularizer=l2(0.0008),
            config=None,
    ):
        super().__init__(
            input_shape=input_shape,
            repetitions=(3, 4, 23, 3),
            kernel_regularizer=kernel_regularizer,
            config=config
        )


if __name__ == '__main__':
    # ssd = SSDFPN_ResNet50_CBAM()
    ssd = SSDFPN_ResNet50_CBAM()
    print(len(ssd.basemodel.outputs))
    for _ in ssd.basemodel.outputs:
        print(_)
    # ssd.plot_model('ssd_resnet50.png')



