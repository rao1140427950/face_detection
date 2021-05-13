import itertools
import math
import tensorflow as tf


def generate_anchors(config):
    """ Generate default boxes for all feature maps

    Args:
        config: information of feature maps
            scales: boxes' size relative to image's size
            fm_sizes: sizes of feature maps
            ratios: box ratios used in each feature maps

    Returns:
        anchors: tensor of shape (num_default, 4)
                       with format (cx, cy, w, h)
    """
    anchors = []
    scales = config['scales']
    steps = config['steps']
    two_boxes_for_ar1 = config['two_boxes_for_ar1']
    ratios = config['aspect_ratios_per_layer']
    img_height = config['img_height']
    img_width = config['img_width']
    offsets = config['offsets']

    for m, step in enumerate(steps):
        fm_size_w = img_width // step
        fm_size_h = img_height // step
        for i, j in itertools.product(range(fm_size_h), range(fm_size_w)):
            cx = (j + offsets) / fm_size_w
            cy = (i + offsets) / fm_size_h

            for ratio in ratios[m]:
                if ratio == 1.0:
                    anchors.append([
                        cx,
                        cy,
                        scales[m],
                        scales[m]
                    ])
                    if two_boxes_for_ar1:
                        anchors.append([
                            cx,
                            cy,
                            math.sqrt(scales[m] * scales[m + 1]),
                            math.sqrt(scales[m] * scales[m + 1])
                        ])
                else:
                    r = math.sqrt(ratio)
                    anchors.append([
                        cx,
                        cy,
                        scales[m] * r,
                        scales[m] / r
                    ])

    anchors = tf.constant(anchors)
    anchors = tf.clip_by_value(anchors, 0.0, 1.0)

    return anchors

