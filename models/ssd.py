import tensorflow as tf
import sys
import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Reshape, Concatenate, Activation
from tensorflow.keras.backend import int_shape
sys.path.append('..')
from utils.model import Model
from utils.layers import AnchorBoxes, DecodeDetections, DecodeDetectionsFast
from backbones.resnet import ResNet


class SSDHead(Model):

    def __init__(self,
                 backbone,
                 n_classes=1,
                 mode='training',
                 aspect_ratios_per_layer=None,
                 steps=None,
                 variances=None,
                 l2_regularization=0.0005,
                 scales=None,
                 offsets=None,
                 two_boxes_for_ar1=True,
                 coords='centroids',
                 clip_boxes=False,
                 normalize_coords=True,
                 confidence_thresh=0.01,
                 iou_threshold=0.45,
                 top_k=500,
                 nms_max_output_size=1000,
                 ):

        ### Initialize and check variables

        if aspect_ratios_per_layer is None:
            aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
                                       [1.0, 2.0, 0.5],
                                       [1.0, 2.0, 0.5],
                                       [1.0, 2.0, 0.5]]
        if steps is None:
            steps = [4, 8, 16, 32]
        if variances is None:
            variances = [0.1, 0.1, 0.2, 0.2]
        if scales is None:
            scales = [0.04, 0.1, 0.26, 0.45, 0.58]
        if offsets is None:
            offsets = [0.5, 0.5, 0.5, 0.5]

        input_x = backbone.model.input

        predict_layers = backbone.model.outputs
        n_predictor_layers = len(predict_layers)

        n_classes += 1 # background class.
        l2_reg = l2_regularization
        self.kernel_regularizer = l2(l2_reg)
        _, img_height, img_width, img_channels = int_shape(input_x)

        assert n_predictor_layers == len(aspect_ratios_per_layer)
        assert n_predictor_layers == len(steps)
        assert n_predictor_layers + 1 == len(scales)
        assert len(variances) == 4
        assert np.any(np.array(variances) > 0)
        assert n_predictor_layers == len(offsets)

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

        ### Generate the anchor boxes
        # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
        anchors = []
        cnt = 0
        this_scales = scales[:-1]
        next_scales = scales[1:]
        for _this_scale, _next_scale, _ratio, _step, _offset, _bbox_loc in zip(
            this_scales, next_scales, aspect_ratios_per_layer, steps, offsets, bbox_locs
        ):
            anchors.append(
                AnchorBoxes(
                    img_height, img_width,
                    this_scale=_this_scale, next_scale=_next_scale,
                    aspect_ratios=_ratio,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    this_steps=_step,
                    this_offsets=_offset,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    coords=coords,
                    normalize_coords=normalize_coords,
                    name='bbox_anchor_{}'.format(cnt)
                )(_bbox_loc)
            )
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
        # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
        anchors_reshape = []
        cnt = 0
        for _anchor in anchors:
            anchors_reshape.append(Reshape((-1, 8), name='anchors_reshape_{}'.format(cnt))(_anchor))
            cnt += 1

        ### Concatenate the predictions from the different layers
        # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
        # so we want to concatenate along axis 1, the number of boxes per layer
        # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
        conf_all = Concatenate(axis=1, name='bbox_conf')(bbox_confs_reshape)
        # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
        loc_all = Concatenate(axis=1, name='bbox_loc')(bbox_locs_reshape)
        # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
        anchor_all = Concatenate(axis=1, name='bbox_anchor')(anchors_reshape)

        # The box coordinate predictions will go into the loss function just the way they are,
        # but for the class predictions, we'll apply a softmax activation layer first
        conf_all_softmax = Activation('softmax', name='bbox_conf_softmax')(conf_all)

        # Concatenate the class and box predictions and the anchors to one large predictions vector
        # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
        predictions = Concatenate(axis=2, name='predictions')([conf_all_softmax, loc_all, anchor_all])

        if mode == 'training':
            model = tf.keras.models.Model(inputs=input_x, outputs=predictions)
        elif mode == 'inference':
            decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
            model = tf.keras.models.Model(inputs=input_x, outputs=decoded_predictions)
        elif mode == 'inference_fast':
            decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                       iou_threshold=iou_threshold,
                                                       top_k=top_k,
                                                       nms_max_output_size=nms_max_output_size,
                                                       coords=coords,
                                                       normalize_coords=normalize_coords,
                                                       img_height=img_height,
                                                       img_width=img_width,
                                                       name='decoded_predictions')(predictions)
            model = tf.keras.models.Model(inputs=input_x, outputs=decoded_predictions)
        else:
            raise ValueError(
                "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

        super().__init__(basemodel=backbone.model, model=model, kernel_regularizer=l2(l2_reg))

        predictor_sizes = []
        for _bbox_conf in bbox_confs:
            predictor_sizes.append(int_shape(_bbox_conf)[1:3])
        self._predictor_sizes = np.array(predictor_sizes)

        config = {'img_height': img_height,
                  'img_width': img_width,
                  'image_size': (img_height, img_width),
                  'n_classes': n_classes - 1,
                  'predictor_sizes': self._predictor_sizes,
                  'min_scale': None,
                  'max_scale': None,
                  'scales': scales,
                  'aspect_ratios_per_layer': aspect_ratios_per_layer,
                  'two_boxes_for_ar1': two_boxes_for_ar1,
                  'steps': steps,
                  'offsets': offsets,
                  'clip_boxes': clip_boxes,
                  'variances': variances,
                  'coords': coords,
                  'normalize_coords': normalize_coords,
                  'output_sizes': int_shape(predictions)[1:3]}
        self._config = config

    def get_predictor_sizes(self):
        return self._predictor_sizes

    def get_config(self):
        return self._config

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
    ):
        repetitions = (3, 4, 6, 3)
        resnet = ResNet(
            input_shape=input_shape,
            kernel_regularizer=kernel_regularizer,
            repetitions=repetitions,
            out_layer_names=out_layer_names
        )
        super().__init__(
            backbone=resnet,
            n_classes=1,
            mode='training',
            aspect_ratios_per_layer=None,
            steps=None,
            variances=None,
            l2_regularization=0.0005,
            scales=None,
            offsets=None,
            two_boxes_for_ar1=True,
            coords='centroids',
            clip_boxes=False,
            normalize_coords=True,
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=500,
            nms_max_output_size=1000,
        )


class SSD_ResNet101(SSDHead):

    def __init__(
            self,
            input_shape=(512, 512, 3),
            kernel_regularizer=l2(0.0008),
            out_layer_names='auto',
    ):
        repetitions = (3, 4, 23, 3)
        resnet = ResNet(
            input_shape=input_shape,
            kernel_regularizer=kernel_regularizer,
            repetitions=repetitions,
            out_layer_names=out_layer_names
        )
        super().__init__(
            backbone=resnet,
            n_classes=1,
            mode='training',
            aspect_ratios_per_layer=None,
            steps=None,
            variances=None,
            l2_regularization=0.0005,
            scales=None,
            offsets=None,
            two_boxes_for_ar1=True,
            coords='centroids',
            clip_boxes=False,
            normalize_coords=True,
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=500,
            nms_max_output_size=1000,
        )


if __name__ == '__main__':
    ssd = SSD_ResNet50()
    ssd.plot_model('ssd_resnet50.png')



