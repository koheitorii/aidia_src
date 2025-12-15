<p align="center" style="font-size:36px; font-weight:bold;">
  <img src="aidia/icons/icon.png" width="50%"><br>
  Aidia
</p>

<h4 align="center">
  AI Development and Image Annotation
</h4>

## Description
Aidia is a medical image annotation tool with AI development utilities.
Pre-built packages (Windows) of Aidia and documents are available at [TRK_Library](https://trklibrary.com/) (Japanese only).

## Features
- Building, training, evaluation and inference deep learning models for object detection and segmentation without coding.
- Image annotation for polygon, rectangle, polyline, line and point.
- Simply labeling by customized GUI buttons.
- DICOM format (.dcm) support including DICOM files which have no extention.
- Adjustment of brightness and contrast by mouse dragging like a DICOM viewer.

## Launch App
You need install Python 3.12 and run below:
```bash
python3 -m venv env # if not existed
source env/bin/activate
pip install -r requirements.txt # if not installed
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 # if not installed
python -m aidia
```

## GPU Support

You must require NVIDIA GPU and the driver.

## Acknowledgement

This project is inspired from [wkentaro/labelme](https://github.com/wkentaro/labelme),
