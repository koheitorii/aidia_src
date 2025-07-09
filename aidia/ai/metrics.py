import numpy as np
import torch
import torch.nn as nn

from aidia import image


def mask_iou(pred, gt):
    pred_list = image.mask2rect(pred)
    gt_list = image.mask2rect(gt)
    
    ious = []
    for pred_rect in pred_list:
        best_iou = 0.0
        for gt_rect in gt_list:
           iou = calc_iou(pred_rect, gt_rect)
           if iou > best_iou:
               best_iou = iou
        ious.append(best_iou)
    return ious


def calc_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max((inter_x2 - inter_x1), 0) * max((inter_y2 - inter_y1), 0)
    area_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area_1 + area_2 - inter_area

    iou = inter_area / union_area
    return iou


def eval_on_iou(y_true, y_pred):
    tp = 0
    fp = 0
    num_gt = 0
    for i in range(y_true.shape[0]):
        pred_mask = y_pred[i]
        gt_mask = y_true[i]
        iou_list = mask_iou(pred_mask, gt_mask)
        for iou in iou_list:
            if iou >= 0.5:
                tp += 1
            else:
                fp += 1
        num_gt += len(image.mask2rect(gt_mask))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (num_gt + 1e-12)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    return [precision, recall, f1]


def common_metrics(tp, tn, fp, fn):
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    return [acc, precision, recall, specificity, f1]


def mIoU(c_matrix) -> float:
    intersection = np.diag(c_matrix)
    union = np.sum(c_matrix, axis=0) + np.sum(c_matrix, axis=1) - intersection
    iou = intersection / union
    miou = np.mean(iou)
    return miou

def multi_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes))

    for yt, yp in zip(y_true, y_pred):
        matrix[yt, yp] += 1

    return matrix

def iou(box1, box2):
    """Calculates IoU of box1 and box2.

    Parameters
    ----------
    box1: 1D vector [y1, x1, y2, x2]
    box2: 1D vector [y1, x1, y2, x2]

    Returns
    -------
    iou: float
        Intersection Over Union value.
    """
    # Calculate intersection areas.
    y1 = max(box1[0], box2[0])
    y2 = min(box1[2], box2[2])
    x1 = max(box1[1], box2[1])
    x2 = min(box1[3], box2[3])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)

    # Compute IoU.
    box1_area = max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0)
    box2_area = max(box2[2] - box2[0], 0) * max(box2[3] - box2[1], 0)
    union = box1_area + box2_area - intersection
    iou = intersection / union
    return iou


class MultiMetrics(nn.Module):
    def __init__(self, threshold=0.5, class_id=None, name='MultiMetrics', **kwargs):
        super().__init__()
        self.true_positives = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.true_negatives = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        # self.false_positives = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        # self.false_negatives = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.sum_ytrue = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.sum_ypred = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.sum_inv_ytrue = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        # self.sum_inv_ypred = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.threshold = threshold
        self.class_id = class_id
        self.name = name

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.class_id is not None:
            y_true = y_true[..., self.class_id]
            y_pred = y_pred[..., self.class_id]
        
        y_true = y_true.bool()
        inv_y_true = ~y_true
        y_pred = y_pred >= self.threshold
        inv_y_pred = ~y_pred

        # TP
        values = y_true & y_pred
        values = values.float()
        self.true_positives.data += torch.sum(values)
        self.true_positives.data += 1e-6

        # TN
        values = inv_y_true & inv_y_pred
        values = values.float()
        self.true_negatives.data += torch.sum(values)
        self.true_negatives.data += 1e-6

        # FP
        # values = inv_y_true & y_pred
        # values = values.float()
        # self.false_positives.data += torch.sum(values)

        # FN
        # values = y_true & inv_y_pred
        # values = values.float()
        # self.false_negatives.data += torch.sum(values)

        # TP + FN
        y_true = y_true.float()
        self.sum_ytrue.data += torch.sum(y_true)
        self.sum_ytrue.data += 1e-6

        # TP + FP
        y_pred = y_pred.float()
        self.sum_ypred.data += torch.sum(y_pred)
        self.sum_ypred.data += 1e-6

        # TN + FP
        inv_y_true = inv_y_true.float()
        self.sum_inv_ytrue.data += torch.sum(inv_y_true)
        self.sum_inv_ytrue.data += 1e-6

        # TN + FN
        # inv_y_pred = inv_y_pred.float()
        # self.sum_inv_ypred.data += torch.sum(inv_y_pred)
    
    def result(self):
        precision = self.true_positives / self.sum_ypred
        recall = self.true_positives / self.sum_ytrue
        specificity = self.true_negatives / self.sum_inv_ytrue
        tpr = recall
        fpr = 1.0 - specificity
        f1 = (2 * precision * recall) / (precision + recall)
        return [precision, recall, specificity, tpr, fpr, f1]
    
    def reset_states(self):
        """メトリクスの状態をリセットする"""
        self.true_positives.data.zero_()
        self.true_negatives.data.zero_()
        self.sum_ytrue.data.zero_()
        self.sum_ypred.data.zero_()
        self.sum_inv_ytrue.data.zero_()

        
