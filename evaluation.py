import tensorflow as tf
import models.ssd as ssd
import models.ssd_fpn as ssd_fpn
from utils.losses import SSDLoss
import os
import numpy as np
from config import *
from utils.datasets import WiderFaceDataset
from inference import inference_single_imagefile
from utils.box_utils import *
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '9'

# Build model
if MODEL == 'ssd_resnet50':
    net = ssd.SSD_ResNet50(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
elif MODEL == 'ssdfpn_resnet_cbam':
    net = ssd_fpn.SSDFPN_ResNet_CBAM(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        repetitions=REPETITIONS,
        config=SSD_CONFIG,
    )
else:
    raise ValueError("Unknown model: `{}`.".format(MODEL))

net.model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=MOMENTUM),
    loss=SSDLoss(),
)

# Load weights from checkpoints
# checkpoint_path = 'None'
# checkpoint_path = WORK_DIR + '/checkpoint-' + MODEL_NAME + '.h5'
checkpoint_path = WORK_DIR + 'checkpoint-ssdfpn_resnet3333_cbam-313-4.06.h5'
weight_file = WORK_DIR + '/' + MODEL_NAME + '_weights.h5'
if os.path.exists(weight_file):
    net.load_weights(weight_file)
    print('Load {}.'.format(weight_file))
elif os.path.exists(checkpoint_path):
    net.load_weights(checkpoint_path)
    print('Load {}.'.format(checkpoint_path))
else:
    raise ValueError("Checkpoint and weights file not found.")

if not os.path.exists(EVALUATION_RESULTS_DIR):
    os.mkdir(EVALUATION_RESULTS_DIR)

if not os.path.exists(VALIDATION_IMAGES_DIR):
    raise ValueError("Dir: {} not found.".format(VALIDATION_IMAGES_DIR))
parent_dirs =  os.listdir(VALIDATION_IMAGES_DIR)
print("Found {} dirs in {}.".format(len(parent_dirs), VALIDATION_IMAGES_DIR))

# read ground truth
val_data = WiderFaceDataset(
    SSD_CONFIG,
    txt_annos_path=VALIDATION_ANNOS_PATH,
    image_root_dir=VALIDATION_IMAGES_DIR,
    argument=False
)

# annos_dict = val_data._read_txt_annos()
# np.save('annos_dict.npy', annos_dict)

annos_dict = np.load('annos_dict.npy', allow_pickle=True).item()
def cal_score(boxes_gt, boxes_pre, iou_thresh):
    print(len(boxes_pre), len(boxes_gt))
    boxes_pre = np.array(boxes_pre)
    boxes_gt = np.array(boxes_gt)
    boxes_pre[:,2] += boxes_pre[:,0]
    boxes_pre[:,3] += boxes_pre[:,1]
    boxes_gt[:,2] += boxes_gt[:,0]
    boxes_gt[:,3] += boxes_gt[:,1]
    boxes_pre = boxes_pre.tolist()
    boxes_gt = boxes_gt.tolist()
    pre_true = 0
    pre_false = 0
    for box_pre in boxes_pre:
        pre_flag = False
        for box_gt in boxes_gt:
            box_pre_shaped = np.reshape(np.array(box_pre, np.float), (1, -1))
            box_gt_shaped = np.reshape(np.array(box_gt, np.float), (1, -1))
            iou = tf.squeeze(compute_iou(box_pre_shaped, box_gt_shaped))
            print(iou)
            if iou > iou_thresh:
                pre_flag = True
        if pre_flag:
            pre_true += 1
        else:
            pre_false += 1
    TP = pre_true / len(boxes_gt)
    # FP = pre_false / len(boxes_pre)
    FN = pre_false / len(boxes_gt)
    P = pre_true / len(boxes_pre)
    return TP, P

def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    print(mpre)
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
 
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    print(mpre)
  return ap

pre_dict = dict()

for parent_dir in tqdm(parent_dirs):
    parent_dir_origin = parent_dir
    results_parent_dir = os.path.join(EVALUATION_RESULTS_DIR, parent_dir)
    parent_dir = os.path.join(VALIDATION_IMAGES_DIR, parent_dir)
    # if not os.path.exists(results_parent_dir):
    #     os.mkdir(results_parent_dir)
    image_files = os.listdir(parent_dir)
    print("Found {} images in {}.".format(len(image_files), parent_dir))
    for image_file in tqdm(image_files):
        result_file_name = image_file.split('.')[0]
        # print(result_file_name)
        key = parent_dir_origin + '/' + result_file_name + '.jpg'
        result_file = os.path.join(results_parent_dir, result_file_name + '.txt')
        image_file = os.path.join(parent_dir, image_file)
        preds = inference_single_imagefile(net, image_file, _thresh=0)
        scores = preds[:, 1]
        boxes = preds[:, 2:6].astype(np.int)
        # lines = [result_file_name + "\n", "%d\n" % len(preds)]
        # (xmin, ymin, xmax, ymax) to (xmin, ymin, w, h)
        boxes += 1
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        box_pre = []
        for box, score in zip(boxes, scores):
            # lines.append("%d %d %d %d %.3f\n" % (box[0], box[1], box[2], box[3], score))
            box_pre.append([score, box[0:4].tolist()])
        pre_dict[key] = box_pre
        # print(box_pre)
        # print(annos_dict[key])
        # print(box_pre)
        # print(cal_score(annos_dict[key], box_pre))
        # f = open(result_file, 'w')
        # f.writelines(lines)
        # f.close()
        # print("%s created." % result_file)
np.save('pre_dict.npy', pre_dict)

pre_dict = np.load('pre_dict.npy', allow_pickle=True).item()

def cal_tp_p(pre_dict, annos_dict, iou_th):
    TP_mean = []
    P_mean = []
    for conf_th_int in range(1,10):
        conf_th = conf_th_int*0.1
        TP = []
        P = []
        for k in pre_dict.keys():
            if pre_dict[k][0] > conf_th:
                tp, p = cal_score(annos_dict[k], pre_dict[k][1], iou_th)
                TP.append(tp)
                P.append(p)
        tp_m, p_m = np.mean(TP), np.mean(P)
        TP_mean.append(tp_m)
        P_mean.append(p_m)
    return TP_mean, P_mean
TP, P = cal_tp_p(pre_dict, annos_dict, 0.5, 0.5)
print(voc_ap(TP, P))
