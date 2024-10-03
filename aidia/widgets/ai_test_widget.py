import os
import cv2
import numpy as np
import json

from onnxruntime import InferenceSession
from qtpy import QtWidgets

from aidia import CLS, DET, SEG
from aidia.image import convert_dtype, mask2polygon, preprocessing

class AITestWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
     
    def generate_shapes(self, img, log_dir, epsilon, area_limit):
        h, w = img.shape[:2]
        if img.dtype == np.uint16:
            img = convert_dtype(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        json_path = os.path.join(log_dir, "config.json")
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
        model = InferenceSession(onnx_path)

        if task == SEG:
            img = cv2.resize(img, img_size)
            inputs = preprocessing(img, is_tensor=True)
            input_name = model.get_inputs()[0].name
            result = model.run([], {input_name: inputs})[0]
            masks = np.where(result[0] >= 0.5, 255, 0)
            masks = masks.astype(np.uint8)
            masks = cv2.resize(masks, (w, h), cv2.INTER_NEAREST)
            shapes = mask2polygon(masks, labels, epsilon, area_limit)
            return shapes
        else:
            raise NotImplementedError("Not implemented error.")
