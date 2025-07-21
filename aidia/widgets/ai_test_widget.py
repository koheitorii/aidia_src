import os
import cv2
import numpy as np
import json

from onnxruntime import InferenceSession
from qtpy import QtWidgets

from aidia import CLS, DET, SEG, CONFIG_JSON, DrawMode
from aidia.image import convert_dtype, preprocessing, imread

from ultralytics import YOLO

class AITestWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
     
    def generate_shapes(self, img, log_dir, epsilon, area_limit):
        h, w = img.shape[:2]
        if img.dtype == np.uint16:
            img = convert_dtype(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        json_path = os.path.join(log_dir, CONFIG_JSON)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} is not found.")
        try:
            with open(json_path, encoding="utf-8") as f:
                dic = json.load(f)
        except Exception as e:
            try:    #  not UTF-8 json file handling
                with open(json_path) as f:
                    dic = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load config.json: {e}")
        
        img_size = (dic["INPUT_SIZE"], dic["INPUT_SIZE"])
        task = dic["TASK"]
        labels = dic["LABELS"]

        onnx_path = os.path.join(log_dir, "model.onnx")

        if task == SEG:
            model = InferenceSession(onnx_path)
            img = cv2.resize(img, img_size)
            inputs = preprocessing(img, is_tensor=True, channel_first=True)
            input_name = model.get_inputs()[0].name
            result = model.run([], {input_name: inputs})[0]
            masks = np.where(result[0] >= 0.5, 255, 0)
            masks = masks.astype(np.uint8)
            masks = cv2.resize(masks, (w, h), cv2.INTER_NEAREST)
            shapes = self.mask2polygon(masks, labels, epsilon, area_limit)
            return shapes
        elif task == DET:
            model = YOLO(onnx_path, task="detect")
            result = model.predict(img, device='cpu')[0]
            boxes = result.boxes.xyxy.numpy()
            class_id = result.boxes.cls.numpy()
            shapes = []
            for i in range(len(boxes)):
                box = boxes[i]
                shape = {
                    "label": labels[int(class_id[i])],
                    "points": [[box[0], box[1]], [box[2], box[3]]],
                    "shape_type": DrawMode.RECTANGLE
                }
                shapes.append(shape)
            return shapes
        else:
            raise NotImplementedError("Not implemented error.")

    @staticmethod
    def mask2polygon(masks, labels, approx_epsilon=0.003, area_limit=50):
        """ Convert masks to polygon shapes.
        Args:
            masks (np.ndarray): Binary masks of shape (H, W, N) or (N, H, W).
            labels (list): List of label names.
            approx_epsilon (float): Approximation epsilon for polygon approximation.
            area_limit (int): Minimum area for a contour to be considered.
        Returns:
            list: List of shapes, each shape is a dictionary with label, points, and shape_type.
        """
        shapes = []
        for i in range(len(labels) - 1):
            # binary = masks[:, :, i + 1]
            binary = masks[i + 1]
            binary = cv2.dilate(binary, (9, 9))
            # binary = np.array(np.where(masks[:, :, i + 1], 255, 0), dtype=np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not len(contours):
                continue
            for cnt in contours:
                # skip figures have small area.
                area = cv2.contourArea(cnt)
                if area < area_limit:
                    continue
                # Detect points of a polygon.
                approx = cv2.approxPolyDP(
                    curve=cnt,
                    epsilon=approx_epsilon * cv2.arcLength(cnt, True),
                    closed=True)
                # Skip polygons have less than 3 points.
                if len(approx) < 3:
                    continue
                approx = approx.astype(int).reshape((-1, 2)).tolist()

                shape = {}
                shape["label"] = labels[i - 1]
                shape["points"] = approx
                shape["shape_type"] = DrawMode.POLYGON
                shapes.append(shape)
        return shapes
