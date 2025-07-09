import os
import keras
import numpy as np
import glob
import random

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
        self.dataset.write_dataset_for_ultra()
    

    def load_dataset(self):
        self.dataset = Dataset(self.config, load=True)
    

    def build_model(self, mode, weights_path=None):
        """Build YOLO model."""
        assert mode in ["train", "test"]

        if mode == "train":
            self.model = YOLO(None, task="detect")
        elif mode == "test":
            self.model = YOLO(weights_path, task="detect")
        else:
            raise ValueError("Mode must be 'train' or 'test'.")


    def train(self, custom_callbacks=None):
        checkpoint_dir = utils.get_dirpath_with_mkdir(self.config.log_dir, 'weights')

        self.model.train(
            data=self.dataset.yaml_path,
            epochs=self.config.EPOCHS,
            imgsz=self.config.INPUT_SIZE,
            batch=self.config.BATCH_SIZE,
            device=0 if self.config.gpu_num >= 0 else "cpu",
            project=self.config.log_dir,
            name=self.config.NAME,
            # save_period=1 if self.config.SAVE_BEST else 20,
            # save_best=self.config.SAVE_BEST,
        )
        # if self.config.SAVE_BEST:
        #     checkpoint_path = os.path.join(checkpoint_dir, "best_model.h5")
        # else:
        #     checkpoint_path = os.path.join(checkpoint_dir, "{epoch:04d}.h5")

        # callbacks = [
        #     keras.callbacks.ModelCheckpoint(
        #         checkpoint_path,
        #         monitor='val_loss',
        #         save_best_only=self.config.SAVE_BEST,
        #         save_weights_only=True,
        #         period=1 if self.config.SAVE_BEST else 20,
        #     ),
        # ]
        # if custom_callbacks:
        #     for c in custom_callbacks:
        #         callbacks.append(c)

        # train_generator = YOLODataGenerator(self.dataset, self.config, mode="train")
        # val_generator = YOLODataGenerator(self.dataset, self.config, mode="val")

        # self.model.fit(
        #     train_generator,
        #     steps_per_epoch=self.dataset.train_steps,
        #     epochs=self.config.EPOCHS,
        #     verbose=0,
        #     validation_data=val_generator,
        #     validation_steps=self.dataset.val_steps,
        #     callbacks=callbacks
        # )

        # save last model
        # if not self.config.SAVE_BEST:
        #     checkpoint_path = os.path.join(checkpoint_dir, "last_model.h5")
        #     self.model.save_weights(checkpoint_path)


    def stop_training(self):
        self.model.stop_training = True


    def evaluate(self, cb_widget=None):
        sum_AP = 0.0
        nc = self.dataset.num_classes
        _i = 0
        
        # test data predictions
        preds = []
        for image_id in self.dataset.test_ids:
            # update progress
            if cb_widget is not None:
                cb_widget.notifyMessage.emit(f"{_i+1} / {self.dataset.num_test}")
                cb_widget.progressValue.emit(int((_i+1) / (self.dataset.num_test * 100)))
                _i += 1
            org_img = self.dataset.load_image(image_id, is_resize=False)
            pred_bboxes = self.model.predict(org_img)
            preds.append(pred_bboxes)

        # calculate AP each class
        if cb_widget is not None:
            cb_widget.notifyMessage.emit(f"Calculating...")
        for class_id in range(nc):
            gt_count = 0.0
            tp_list = []
            fp_list = []
            for pred_idx, image_id in enumerate(self.dataset.test_ids):
                # load image and annotation
                anno_gt = self.dataset.get_yolo_bboxes(image_id)
                if len(anno_gt) == 0:
                    bboxes_gt = []
                    classes_gt = []
                else:
                    bboxes_gt, classes_gt = anno_gt[:, :4], anno_gt[:, 4]

                # ground truths
                bbox_dict_gt = []
                for i in range(len(bboxes_gt)):
                    class_id = classes_gt[i]
                    class_name = self.dataset.class_names[class_id]
                    bbox = list(map(float, bboxes_gt[i]))
                    bbox_dict_gt.append({"class_name": class_name,
                                        "bbox": bbox,
                                        "used": False})
                    gt_count += 1

                # prediction
                pred_bboxes = preds[pred_idx]
                
                bbox_dict_pred = []
                for bbox_pred in pred_bboxes:
                    # xmin, ymin, xmax, ymax = list(map(str, map(int, bbox[:4])))
                    bbox = list(map(float, bbox_pred[:4]))
                    score = bbox_pred[4]
                    class_id = int(bbox_pred[5])
                    class_name = self.dataset.class_names[class_id]
                    score = '%.4f' % score
                    bbox_dict_pred.append({"class_id": class_id,
                                        "class_name": class_name,
                                        "confidence": score,
                                        "bbox": bbox})
                bbox_dict_pred.sort(key=lambda x:float(x['confidence']), reverse=True)

                # count True Positive and False Positive
                for pred in bbox_dict_pred:
                    overlap_max = -1
                    gt_match = -1

                    cls_id = pred["class_id"]
                    cls_pred = pred["class_name"]
                    bb_pred = pred["bbox"]

                    for gt in bbox_dict_gt:
                        cls_gt = gt["class_name"]
                        if cls_gt != cls_pred:
                            continue
                        bb_gt = gt["bbox"]
                        bi = [max(bb_pred[0], bb_gt[0]),
                            max(bb_pred[1], bb_gt[1]),
                            min(bb_pred[2], bb_gt[2]),
                            min(bb_pred[3], bb_gt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            area_pred = (bb_pred[2] - bb_pred[0] + 1) * (bb_pred[3] - bb_pred[1] + 1)
                            area_gt = (bb_gt[2] - bb_gt[0] + 1) * (bb_gt[3] - bb_gt[1] + 1)
                            ua = area_pred + area_gt - iw * ih
                            ov = (iw * ih) / ua
                            if ov > overlap_max:
                                overlap_max = ov
                                gt_match = gt
                
                    # set minimum overlap (AP50)
                    iou_threshold = 0.5  # TODO:user configuration param
                    if overlap_max >= iou_threshold:
                        if not bool(gt_match["used"]):
                            gt_match["used"] = True
                            tp_list.append(1.0)
                            fp_list.append(0.0)
                        else:
                            tp_list.append(0.0)
                            fp_list.append(1.0)
                    else:
                        tp_list.append(0.0)
                        fp_list.append(1.0)
            
            cumsum = 0
            for idx, val in enumerate(fp_list):
                fp_list[idx] += cumsum
                cumsum += val

            cumsum = 0
            for idx, val in enumerate(tp_list):
                tp_list[idx] += cumsum
                cumsum += val

            tp = tp_list[-1]
            fp = fp_list[-1]

            recall = tp_list[:]
            for idx, val in enumerate(tp_list):
                recall[idx] = tp_list[idx] / gt_count

            precision = tp_list[:]
            for idx, val in enumerate(tp_list):
                precision[idx] = tp_list[idx] / (fp_list[idx] + tp_list[idx])

            ap, mrec, mprec = self.voc_ap(recall, precision)
            sum_AP += ap

        pre = tp / (tp + fp)
        rec = tp / gt_count
        mAP = sum_AP / nc

        res = {
            "Precision": pre,
            "Recall": rec,
            "mAP50": mAP,
        }

        # save dict
        eval_dir = utils.get_dirpath_with_mkdir(self.config.log_dir, 'evaluation')
        save_dict = {
                "Metrics": list(res.keys()),
                "Values": list(res.values()),
            }
        utils.save_dict_to_excel(save_dict, os.path.join(eval_dir, "scores.xlsx"))
        return res
    
    def predict_by_id(self, image_id, thresh=0.5):
        # load image and annotation
        org_img = self.dataset.load_image(image_id, is_resize=False)

        # TODO: ground truth visualization
        # anno_gt = self.dataset.get_yolo_bboxes(image_id)
        # if len(anno_gt) == 0:
        #     bboxes_gt = []
        #     classes_gt = []
        # else:
        #     bboxes_gt, classes_gt = anno_gt[:, :4], anno_gt[:, 4]

        # prediction
        pred_bboxes = self.model.predict(org_img)
        
        bbox_dict_pred = []
        for bbox_pred in pred_bboxes:
            # xmin, ymin, xmax, ymax = list(map(str, map(int, bbox[:4])))
            bbox = list(map(float, bbox_pred[:4]))
            score = bbox_pred[4]
            class_id = int(bbox_pred[5])
            class_name = self.dataset.class_names[class_id]
            score = '%.4f' % score
            bbox_dict_pred.append({"class_id": class_id,
                                    "class_name": class_name,
                                    "confidence": score,
                                    "bbox": bbox})
        bbox_dict_pred.sort(key=lambda x:float(x['confidence']), reverse=True)

        merge = det2merge(org_img, bbox_dict_pred)
        return merge
    
    def convert2onnx(self):
        onnx_path = os.path.join(self.config.log_dir, "model.onnx")
        if os.path.exists(onnx_path):
            return
        # tf2onnx.convert.from_keras(self.model, opset=11, output_path=onnx_path)

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
    