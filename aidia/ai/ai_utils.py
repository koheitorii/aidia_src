import os
from ultralytics import YOLO
from onnxruntime import InferenceSession
from aidia.ai.config import AIConfig
from aidia import DET, SEG
from aidia import image
import cv2

class InferenceModel(object):
    """Class for handling ONNX model inference."""
    def __init__(self, onnx_path: str, config: AIConfig = None):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file {onnx_path} does not exist.")
        self.session = InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.config = config

    def run(self, file_path: str, save_path: str = None):
        """Run inference and return the first result."""
        try:
            img = image.read_image(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read image {file_path}: {e}")
        img = cv2.resize(img, self.config.image_size)
        if self.config.is_need_padding():
            img = image.pad_image_to_target_size(img, self.config.max_input_size)
        inputs = image.preprocessing(img, is_tensor=True, channel_first=True)
        result = self.session.run([], {self.input_name: inputs})[0][0]
        if save_path is not None:
            if self.config.TASK == SEG:
                result_img = image.mask2merge(img, result, self.config.LABELS, show_labels=self.config.SHOW_LABELS, show_conf=self.config.SHOW_CONF)
            elif self.config.TASK == DET:
                raise NotImplementedError("Saving results for DET task is not implemented yet.")
            image.imwrite(result_img, save_path)
        return result
    

class InferenceModel_Ultralytics(object):
    """Class for handling Ultralytics model inference."""
    def __init__(self, onnx_path: str, config: AIConfig = None):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Model file {onnx_path} does not exist.")
        self.config = config

        if config.TASK == DET:
            task = 'detect'
        elif config.TASK == SEG:
            task = 'segment'
        else:
            raise ValueError(f"Unsupported task: {config.TASK}")
        self.model = YOLO(onnx_path, task=task)

    def run(self, file_path: str, save_path: str = None):
        """Run inference and return the first result."""
        try:
            img = image.read_image(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read image {file_path}: {e}")
        img = image.read_image(file_path)
        img = cv2.resize(img, self.config.image_size)
        result = self.model.predict(img, device='cpu')[0]
        if save_path is not None:
            result_img = result.plot(labels=self.config.SHOW_LABELS, conf=self.config.SHOW_CONF, line_width=1)
            image.imwrite(result_img, save_path)
        return result


def write_onnx_u(filepath: str):
    """Convert YOLO model to ONNX format."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} does not exist.")
    model = YOLO(filepath)
    return model.export(format="onnx", device='cpu')
