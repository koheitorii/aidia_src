import os
import cv2
import numpy as np
import json

from onnxruntime import InferenceSession
from qtpy import QtWidgets

from aidia import CLS, DET, SEG, CONFIG_JSON, DrawMode
from aidia import image
from aidia.image import convert_dtype, preprocessing, imread
from aidia.ai.config import AIConfig

from ultralytics import YOLO

class AITestWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
     
    def generate_shapes(self, img_path, log_dir, epsilon, area_limit):
        """ Generate shapes from an image using a trained model.
        Args:
            img_path (str): Path to the input image.
            log_dir (str): Directory containing the trained model and config.
            epsilon (float): Approximation epsilon for polygon approximation.
            area_limit (int): Minimum area for a contour to be considered.
        Returns:
            list: List of shapes, each shape is a dictionary with label, points, and shape_type.
        """
        img = image.read_image(img_path)
     
        src_h, src_w = img.shape[:2]
        
        json_path = os.path.join(log_dir, CONFIG_JSON)
        config = AIConfig()
        config.load(json_path)
        
        task = config.TASK
        labels = config.LABELS

        onnx_path = os.path.join(log_dir, "model.onnx")

        if task == SEG and not config.is_ultralytics():
            model = InferenceSession(onnx_path)
            img = cv2.resize(img, config.image_size)
            if config.is_need_padding():
                img = image.pad_image_to_target_size(img, config.max_input_size)
            inputs = preprocessing(img, is_tensor=True, channel_first=True)
            input_name = model.get_inputs()[0].name
            result = model.run([], {input_name: inputs})[0][0]
            shapes = self.result2polygon(result, labels, (src_w, src_h), epsilon, area_limit)
            return shapes
        elif task == DET and config.is_ultralytics():
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
    def result2polygon(result, labels, img_size, approx_epsilon=0.003, area_limit=50):
        """ Convert segmentation result to polygon shapes.
        Args:
            result (np.ndarray): Segmentation result from the model.
            labels (list): List of class labels.
            img_size (tuple): Size of the original image (width, height).
            approx_epsilon (float): Approximation epsilon for polygon approximation.
            area_limit (int): Minimum area for a contour to be considered.
        Returns:
            list: List of shapes, each shape is a dictionary with label, points, and shape_type.
        """
        masks = np.where(result >= 0.5, 255, 0)
        masks = masks.astype(np.uint8)

        shapes = []
        for i in range(len(labels)):
            # binary = masks[:, :, i + 1]
            binary = masks[i + 1]
            binary = cv2.resize(binary, img_size)
            binary = cv2.dilate(binary, (9, 9))
            # binary = np.array(np.where(masks[i + 1], 255, 0), dtype=np.uint8)
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
                shape["label"] = labels[i]
                shape["points"] = approx
                shape["shape_type"] = DrawMode.POLYGON
                shapes.append(shape)
        return shapes
