import os
from ultralytics import YOLO


def write_onnx(filepath: str):
    """Convert YOLO model to ONNX format."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} does not exist.")
    model = YOLO(filepath)
    return model.export(format="onnx", device='cpu')