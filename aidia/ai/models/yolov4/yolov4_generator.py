import random
import re
import cv2
import numpy as np
import tensorflow as tf
import imgaug
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from aidia import aidia_logger
from aidia.ai.dataset import Dataset
from aidia.ai.config import AIConfig
from aidia.ai.models.yolov4.yolov4_config import YOLO_Config
from aidia.ai.models.yolov4.yolov4_utils import image_preprocess, bbox_iou

class YOLODataGenerator(object):
    """Data generator for object detection with YOLO.
    
    Parameters
    ----------
    dataset: Dataset
    config: AIConfig
    mode: str
        "train", "val" and "test"
    """
    def __init__(self, dataset:Dataset, config:AIConfig, mode="train") -> None:
        assert mode in ["train", "val", "test"]

        self.dataset = dataset
        self.config = config
        self.mode = mode

        self.num_classes = self.dataset.num_classes
        self.augseq = config.get_augseq()

        if mode == "train":
            self.image_ids = self.dataset.train_ids
            self.augmentation = True
        elif self.mode == "val":
            self.image_ids = self.dataset.val_ids
            self.augmentation = False
        elif self.mode == "test":
            self.image_ids = self.dataset.test_ids
            self.augmentation = False
        else:
            raise NotImplementedError
        self.num_images = len(self.image_ids)
        np.random.shuffle(self.image_ids)

        self.yolo_config = YOLO_Config()
        self.is_tiny = True if self.config.MODEL.split("-")[-1] == "tiny" else False
        version = int(re.sub(r'[^0-9]', '', self.config.MODEL))
        self.strides, self.anchors, _ = self.yolo_config.get_yolo_params(self.is_tiny, version)
        self.train_output_sizes = self.config.INPUT_SIZE // self.strides

        self.image_count = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        with tf.device("/cpu:0"):
            # initialize variables
            self.batch_image = np.zeros((
                    self.config.total_batchsize,
                    self.config.INPUT_SIZE,
                    self.config.INPUT_SIZE, 3), dtype=np.float32)
            if self.is_tiny:
                self.batch_label_mbbox = np.zeros((
                        self.config.total_batchsize,
                        self.train_output_sizes[0],
                        self.train_output_sizes[0],
                        self.yolo_config.ANCHOR_PER_SCALE,
                        5 + self.config.num_classes), dtype=np.float32)
                self.batch_label_lbbox = np.zeros((
                        self.config.total_batchsize,
                        self.train_output_sizes[1],
                        self.train_output_sizes[1],
                        self.yolo_config.ANCHOR_PER_SCALE,
                        5 + self.config.num_classes), dtype=np.float32)
            else:
                self.batch_label_sbbox = np.zeros((
                    self.config.total_batchsize,
                    self.train_output_sizes[0],
                    self.train_output_sizes[0],
                    self.yolo_config.ANCHOR_PER_SCALE,
                    5 + self.config.num_classes), dtype=np.float32)
                self.batch_label_mbbox = np.zeros((
                    self.config.total_batchsize,
                    self.train_output_sizes[1],
                    self.train_output_sizes[1],
                    self.yolo_config.ANCHOR_PER_SCALE,
                    5 + self.config.num_classes), dtype=np.float32)
                self.batch_label_lbbox = np.zeros((
                    self.config.total_batchsize,
                    self.train_output_sizes[2],
                    self.train_output_sizes[2],
                    self.yolo_config.ANCHOR_PER_SCALE,
                    5 + self.config.num_classes), dtype=np.float32)
            
            if not self.is_tiny:
                self.batch_sbboxes = np.zeros(
                    (self.config.total_batchsize, self.yolo_config.MAX_BBOX_PER_SCALE, 4), dtype=np.float32)
            self.batch_mbboxes = np.zeros(
                (self.config.total_batchsize, self.yolo_config.MAX_BBOX_PER_SCALE, 4), dtype=np.float32)
            self.batch_lbboxes = np.zeros(
                (self.config.total_batchsize, self.yolo_config.MAX_BBOX_PER_SCALE, 4), dtype=np.float32)

            # return iteration
            b = 0
            while b < self.config.total_batchsize:
                image_id = self.image_ids[self.image_count]
                self.image_count += 1
                if self.image_count >= self.num_images:
                    self.image_count = 0
                    np.random.shuffle(self.image_ids)

                img = self.dataset.load_image(image_id, is_resize=False)
                bboxes = self.dataset.get_yolo_bboxes(image_id)

                if self.augmentation:
                    bbs = BoundingBoxesOnImage([
                            BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=self.dataset.class_names[label_idx])
                            for xmin, ymin, xmax, ymax, label_idx in bboxes
                        ], shape=img.shape)
                    img, bbs = self.augseq(image=img, bounding_boxes=bbs)
                    bbs = bbs.remove_out_of_image().clip_out_of_image()
                    _bboxes = []
                    for i in range(len(bbs.bounding_boxes)):
                        after = bbs.bounding_boxes[i]
                        x1 = after.x1_int
                        y1 = after.y1_int
                        x2 = after.x2_int
                        y2 = after.y2_int
                        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:  # exclude boxes have tiny area
                            _bboxes.append([x1, y1, x2, y2, self.dataset.class_names.index(after.label)])
                    if len(_bboxes) > 0:
                        bboxes = np.array(_bboxes, int)
                    else:
                        bboxes = None

                    # def pad(image, by):
                    #     image_border1 = imgaug.pad(image, top=1, right=1, bottom=1, left=1,
                    #                         mode="constant", cval=255)
                    #     image_border2 = imgaug.pad(image_border1, top=by-1, right=by-1,
                    #                         bottom=by-1, left=by-1,
                    #                         mode="constant", cval=0)
                    #     return image_border2


                    # def draw_bbs(image, bbs, border):
                    #     image_border = pad(image, border)
                    #     for bb in bbs.bounding_boxes:
                    #         if bb.is_fully_within_image(image.shape):
                    #             color = [255, 0, 0]
                    #         elif bb.is_partly_within_image(image.shape):
                    #             color = [255,255,0]
                    #         else:
                    #             color = [0, 255, 255]
                    #         image_border = bb.shift(left=border, top=border)\
                    #                         .draw_on_image(image_border, size=2, color=color)

                    #     return image_border
                    
                    # image_after = draw_bbs(img, bbs, 100)
                    # import cv2
                    # cv2.imwrite("test.png", image_after)
                    # if self.config.RANDOM_HFLIP:
                    #     img, bboxes = self.random_horizontal_flip(img, bboxes)
                    # if self.config.RANDOM_VFLIP:
                    #     img, bboxes = self.random_vertical_flip(img, bboxes)
                    # if self.config.RANDOM_BRIGHTNESS > 0:
                    #     img = self.random_brightness(img, self.config.RANDOM_BRIGHTNESS)
                    # if self.config.RANDOM_CONTRAST > 0:
                    #     img = self.random_contrast(img, self.config.RANDOM_CONTRAST)
                    # img, bboxes = self.random_crop(img, bboxes)
                    # img, bboxes = self.random_translate(img, bboxes)
                
                if bboxes is None:
                    continue

                img, bboxes = image_preprocess(img, self.config.image_size, bboxes)

                #TODO: handling data augmentation error
                try:
                    if self.is_tiny:
                        (label_mbbox, label_lbbox, mbboxes, lbboxes) = self.preprocess_true_boxes(bboxes)
                    else:
                        (label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) = self.preprocess_true_boxes(bboxes)
                except Exception as e:
                    aidia_logger.error(e, exc_info=True)
                    continue
                
                self.batch_image[b, :, :, :] = img
                if not self.is_tiny:
                    self.batch_label_sbbox[b, :, :, :, :] = label_sbbox
                    self.batch_sbboxes[b, :, :] = sbboxes
                self.batch_label_mbbox[b, :, :, :, :] = label_mbbox
                self.batch_label_lbbox[b, :, :, :, :] = label_lbbox
                self.batch_mbboxes[b, :, :] = mbboxes
                self.batch_lbboxes[b, :, :] = lbboxes

                b += 1

            if not self.is_tiny:
                batch_smaller_target = self.batch_label_sbbox, self.batch_sbboxes
            batch_medium_target = self.batch_label_mbbox, self.batch_mbboxes
            batch_larger_target = self.batch_label_lbbox, self.batch_lbboxes
            if self.is_tiny:
                return (self.batch_image, (batch_medium_target, batch_larger_target))
            else:
                return (self.batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target))
                
    def preprocess_true_boxes(self, bboxes):
        num_output = 2 if self.is_tiny else 3
        label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.yolo_config.ANCHOR_PER_SCALE,
                    5 + self.num_classes,
                )
            )
            for i in range(num_output)
        ]
        bboxes_xywh = [np.zeros((self.yolo_config.MAX_BBOX_PER_SCALE, 4)) for _ in range(num_output)]
        bbox_count = np.zeros((num_output,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )
            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(num_output):
                anchors_xywh = np.zeros((self.yolo_config.ANCHOR_PER_SCALE, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.yolo_config.MAX_BBOX_PER_SCALE)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.yolo_config.ANCHOR_PER_SCALE)
                best_anchor = int(best_anchor_ind % self.yolo_config.ANCHOR_PER_SCALE)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(
                    bbox_count[best_detect] % self.yolo_config.MAX_BBOX_PER_SCALE
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        
        if num_output == 3:
            label_sbbox, label_mbbox, label_lbbox = label
            sbboxes, mbboxes, lbboxes = bboxes_xywh
            return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
        elif num_output == 2:
            label_mbbox, label_lbbox = label
            mbboxes, lbboxes = bboxes_xywh
            return label_mbbox, label_lbbox, mbboxes, lbboxes
        else:
            raise NotImplementedError

    def random_horizontal_flip(self, img, bboxes):
        if random.random() < 0.5:
            _, w, _ = img.shape
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return img, bboxes

    def random_vertical_flip(self, img, bboxes):
        if random.random() < 0.5:
            h, _, _ = img.shape
            img = img[::-1, :, :]
            bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
        return img, bboxes
    
    def random_brightness(self, img, range=40):
        """Return an image applied brightness adjustment
        by beta sampled from [- range,  + range]"""
        beta = random.uniform(-range, range)
        ret = img.astype(float) + beta
        return ret.clip(0, 255).astype(np.uint8)
    
    def random_contrast(self, img, range=0.2):
        """Return an image applied contrast adjustment
        by alpha sampled from [1 - range, 1 + range]"""
        alpha = random.uniform(1.0 - range, 1.0 + range)
        ret = img.astype(float) * alpha
        return ret.clip(0, 255).astype(np.uint8)

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            )
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            )
            crop_xmax = max(
                w, int(max_bbox[2] + random.uniform(0, max_r_trans))
            )
            crop_ymax = max(
                h, int(max_bbox[3] + random.uniform(0, max_d_trans))
            )

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes
