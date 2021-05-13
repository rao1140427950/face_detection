import tensorflow as tf
import sys
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Reshape, Concatenate, Activation
from tensorflow.keras.backend import int_shape
sys.path.append('..')
from utils.model import Model
from backbones.resnet import ResNet


class SSDHead(Model):

    def __init__(self,
                 backbone,
                 n_classes=1,
                 l2_regularization=0.0005,
                 config=None,
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 ):

        ### Initialize and check variables
        if aspect_ratios_per_layer is None:
            aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
                                       [1.0, 2.0, 0.5],
                                       [1.0, 2.0, 0.5],
                                       [1.0, 2.0, 0.5]]
        if config is not None:
            if 'aspect_ratios_per_layer' in config:
                aspect_ratios_per_layer = config['aspect_ratios_per_layer']
            if 'two_boxes_for_ar1' in config:
                two_boxes_for_ar1 = config['two_boxes_for_ar1']

        input_x = backbone.model.input

        predict_layers = backbone.model.outputs
        n_predictor_layers = len(predict_layers)

        n_classes += 1 # background class.
        l2_reg = l2_regularization
        self.kernel_regularizer = l2(l2_reg)
        _, img_height, img_width, img_channels = int_shape(input_x)

        assert n_predictor_layers == len(aspect_ratios_per_layer)

        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))

        ### Build the convolutional predictor layers on top of the base network
        # We precidt `n_classes` confidence values for each box,
        # hence the confidence predictors have depth `n_boxes * n_classes`
        # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
        bbox_confs = []
        cnt = 0
        for _features, _n_boxes in zip(predict_layers, n_boxes):
            bbox_confs.append(self._clean_conv_layer(_features, _n_boxes * n_classes, name='bbox_conf_{}'.format(cnt)))
            cnt += 1
        # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
        # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
        bbox_locs = []
        cnt = 0
        for _features, _n_boxes in zip(predict_layers, n_boxes):
            bbox_locs.append(self._clean_conv_layer(_features, _n_boxes * 4, name='bbox_loc_{}'.format(cnt)))
            cnt += 1

        # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
        # We want the classes isolated in the last axis to perform softmax on them
        bbox_confs_reshape = []
        cnt = 0
        for _bbox_conf in bbox_confs:
            bbox_confs_reshape.append(Reshape((-1, n_classes), name='bbox_conf_reshape_{}'.format(cnt))(_bbox_conf))
            cnt += 1
        # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
        # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
        bbox_locs_reshape = []
        cnt = 0
        for _bbox_loc in bbox_locs:
            bbox_locs_reshape.append(Reshape((-1, 4), name='bbox_loc_reshape_{}'.format(cnt))(_bbox_loc))
            cnt += 1

        ### Concatenate the predictions from the different layers
        # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
        # so we want to concatenate along axis 1, the number of boxes per layer
        # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
        conf_all = Concatenate(axis=1, name='bbox_conf')(bbox_confs_reshape)
        # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
        loc_all = Concatenate(axis=1, name='bbox_loc')(bbox_locs_reshape)

        # The box coordinate predictions will go into the loss function just the way they are,
        # but for the class predictions, we'll apply a softmax activation layer first
        conf_all_softmax = Activation('softmax', name='bbox_conf_softmax')(conf_all)

        # Concatenate the class and box predictions and the anchors to one large predictions vector
        # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4)
        predictions = Concatenate(axis=2, name='predictions')([conf_all_softmax, loc_all])

        model = tf.keras.models.Model(inputs=input_x, outputs=predictions)

        super().__init__(basemodel=backbone.model, model=model, kernel_regularizer=l2(l2_reg))


    def _clean_conv_layer(self, inputs, filters, kernel_size=3, name=None):
        return Conv2D(
            filters,
            kernel_size,
            padding='same',
            activation='linear',
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer='he_normal',
            name=name
        )(inputs)


class SSD_ResNet50(SSDHead):

    def __init__(
            self,
            input_shape=(512, 512, 3),
            kernel_regularizer=l2(0.0008),
            out_layer_names='auto',
            config=None,
    ):
        repetitions = (3, 4, 6, 3)
        resnet = ResNet(
            input_shape=input_shape,
            kernel_regularizer=kernel_regularizer,
            repetitions=repetitions,
            out_layer_names=out_layer_names,
        )
        super().__init__(
            backbone=resnet,
            n_classes=1,
            aspect_ratios_per_layer=None,
            l2_regularization=0.0005,
            two_boxes_for_ar1=True,
            config=config,
        )


class SSD_ResNet101(SSDHead):

    def __init__(
            self,
            input_shape=(512, 512, 3),
            kernel_regularizer=l2(0.0008),
            out_layer_names='auto',
            config=None,
    ):
        repetitions = (3, 4, 23, 3)
        resnet = ResNet(
            input_shape=input_shape,
            kernel_regularizer=kernel_regularizer,
            repetitions=repetitions,
            out_layer_names=out_layer_names,
        )
        super().__init__(
            backbone=resnet,
            n_classes=1,
            aspect_ratios_per_layer=None,
            l2_regularization=0.0005,
            two_boxes_for_ar1=True,
            config=config,
        )


if __name__ == '__main__':
    ssd = SSD_ResNet50()
    ssd.plot_model('ssd_resnet50.png')



