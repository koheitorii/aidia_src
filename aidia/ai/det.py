import os
import keras
import numpy as np
import glob
import random
import torch

from aidia import ModelTypes
from aidia.ai.dataset import Dataset
from aidia.ai.config import AIConfig
from aidia.image import det2merge
# from aidia.ai.models.yolov4.yolov4 import YOLO
# from aidia.ai.models.yolov4.yolov4_generator import YOLODataGenerator
from aidia import utils

from ultralytics import YOLO


class DetectionModel(object):
    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self.dataset = None
        self.model = None
    
    def set_config(self, config):
        self.config = config

    def build_dataset(self):
        self.dataset = Dataset(self.config)
        self.dataset.write_dataset_for_yolo()

    def load_dataset(self):
        self.dataset = Dataset(self.config, load=True)
    
    def build_model(self, mode, weights_path=None):
        """Build YOLO model."""
        assert mode in ["train", "test"]

        if mode == "train":
            if self.config.MODEL.split('_')[1] == "YOLO11n":
                self.model = YOLO("yolo11n.pt", task="detect", verbose=False)
            else:
                raise ValueError("Unsupported model for training.")
        elif mode == "test":
            self.model = YOLO(weights_path, task="detect", verbose=False)
        else:
            raise ValueError("Mode must be 'train' or 'test'.")

    def train(self, custom_callbacks=None):
        """Train YOLO model."""
        assert self.dataset is not None, "Dataset must be built or loaded before training."
        assert self.model is not None, "Model must be built before training."

        if custom_callbacks is not None:
            # add custom callbacks to the model
            self.model.add_callback("on_train_batch_end", custom_callbacks[0])
            self.model.add_callback("on_val_end", custom_callbacks[1])

        self.model.train(
            data=self.dataset.path_yaml,
            epochs=self.config.EPOCHS,
            imgsz=self.config.max_input_size,
            batch=self.config.BATCH_SIZE,
            lr0=self.config.LEARNING_RATE,
            project=self.config.log_dir,
            name=self.config.MODEL,
            rect=self.config.KEEP_ASPECT_RATIO,
            fliplr=0.5 if self.config.RANDOM_HFLIP else 0.0,
            flipud=0.5 if self.config.RANDOM_VFLIP else 0.0,
            degrees=self.config.RANDOM_ROTATE * 180.0,
            scale=self.config.RANDOM_SCALE,
            translate=self.config.RANDOM_SHIFT,
            shear=self.config.RANDOM_SHEAR,
            hsv_v=self.config.RANDOM_BRIGHTNESS,
            perspective=0.0,
            mosaic=0.0,
            erasing=0.0,
            auto_augment=None,
            verbose=False,
        )

    def predict(self, images, conf=0.25, iou=0.45):
        """Predict bounding boxes for a given image."""
        assert self.model is not None, "Model must be built before prediction."

        # run inference
        results = self.model.predict(source=images, conf=conf, iou=iou)
        
    # def predict_by_id(self, image_id, thresh=0.5):
    #     # load image and annotation
    #     org_img = self.dataset.load_image(image_id, is_resize=False)

    #     # TODO: ground truth visualization
    #     # anno_gt = self.dataset.get_yolo_bboxes(image_id)
    #     # if len(anno_gt) == 0:
    #     #     bboxes_gt = []
    #     #     classes_gt = []
    #     # else:
    #     #     bboxes_gt, classes_gt = anno_gt[:, :4], anno_gt[:, 4]

    #     # prediction
    #     pred_bboxes = self.model.predict(org_img)
        
    #     bbox_dict_pred = []
    #     for bbox_pred in pred_bboxes:
    #         # xmin, ymin, xmax, ymax = list(map(str, map(int, bbox[:4])))
    #         bbox = list(map(float, bbox_pred[:4]))
    #         score = bbox_pred[4]
    #         class_id = int(bbox_pred[5])
    #         class_name = self.dataset.class_names[class_id]
    #         score = '%.4f' % score
    #         bbox_dict_pred.append({"class_id": class_id,
    #                                 "class_name": class_name,
    #                                 "confidence": score,
    #                                 "bbox": bbox})
    #     bbox_dict_pred.sort(key=lambda x:float(x['confidence']), reverse=True)

    #     merge = det2merge(org_img, bbox_dict_pred)
    #     return merge
    
    def convert2onnx(self):
        pass

    @staticmethod
    def voc_ap(rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]
        """
        This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab:  for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #   range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #   range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        """
        This part creates a list of indexes where the recall changes
            matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        """
        The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre
    