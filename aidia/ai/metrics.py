import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import product

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

def binary_confusion_matrix(y_true, y_pred):
    """Calculates binary confusion matrix.

    Parameters
    ----------
    y_true: 1D vector
        True labels.
    y_pred: 1D vector
        Predicted labels.

    Returns
    -------
    matrix: 2D array
        Confusion matrix.
    """
    matrix = np.zeros((2, 2))

    for yt, yp in zip(y_true, y_pred):
        matrix[yt, yp] += 1

    return np.array(matrix, dtype=int)

def multi_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes))

    for yt, yp in zip(y_true, y_pred):
        matrix[yt, yp] += 1

    return matrix

def _validate_style_kwargs(default_style_kwargs, user_style_kwargs):
    invalid_to_valid_kw = {
        "ls": "linestyle",
        "c": "color",
        "ec": "edgecolor",
        "fc": "facecolor",
        "lw": "linewidth",
        "mec": "markeredgecolor",
        "mfcalt": "markerfacecoloralt",
        "ms": "markersize",
        "mew": "markeredgewidth",
        "mfc": "markerfacecolor",
        "aa": "antialiased",
        "ds": "drawstyle",
        "font": "fontproperties",
        "family": "fontfamily",
        "name": "fontname",
        "size": "fontsize",
        "stretch": "fontstretch",
        "style": "fontstyle",
        "variant": "fontvariant",
        "weight": "fontweight",
        "ha": "horizontalalignment",
        "va": "verticalalignment",
        "ma": "multialignment",
    }
    for invalid_key, valid_key in invalid_to_valid_kw.items():
        if invalid_key in user_style_kwargs and valid_key in user_style_kwargs:
            raise TypeError(
                f"Got both {invalid_key} and {valid_key}, which are aliases of one "
                "another"
            )
    valid_style_kwargs = default_style_kwargs.copy()

    for key in user_style_kwargs.keys():
        if key in invalid_to_valid_kw:
            valid_style_kwargs[invalid_to_valid_kw[key]] = user_style_kwargs[key]
        else:
            valid_style_kwargs[key] = user_style_kwargs[key]

    return valid_style_kwargs

def confusion_matrix_display(
        fig,
        ax,
        confusion_matrix,
        display_labels=None,
        include_values=True,
        cmap="viridis",
        xticks_rotation="horizontal",
        values_format=None,
        colorbar=True,
        im_kw=None,
        text_kw=None,
    ):
    cm = confusion_matrix
    n_classes = cm.shape[0]

    default_im_kw = dict(interpolation="nearest", cmap=cmap)
    im_kw = im_kw or {}
    im_kw = _validate_style_kwargs(default_im_kw, im_kw)
    text_kw = text_kw or {}

    im_ = ax.imshow(cm, **im_kw)
    text_ = None
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0

        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min

            if values_format is None:
                text_cm = format(cm[i, j], ".2g")
                if cm.dtype.kind != "f":
                    text_d = format(cm[i, j], "d")
                    if len(text_d) < len(text_cm):
                        text_cm = text_d
            else:
                text_cm = format(cm[i, j], values_format)

            default_text_kwargs = dict(ha="center", va="center", color=color)
            text_kwargs = _validate_style_kwargs(default_text_kwargs, text_kw)

            text_[i, j] = ax.text(j, i, text_cm, **text_kwargs)

    if display_labels is None:
        display_labels = np.arange(n_classes)
    if colorbar:
        fig.colorbar(im_, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    return

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

        
