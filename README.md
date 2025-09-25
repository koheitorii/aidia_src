<h1 align="center">
  <img src="aidia/icons/icon.png" width="30%"><br>
  Aidia
</h1>

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
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
python -m aidia
```

## GPU Support

You must require NVIDIA GPU and the driver.

## Acknowledgement

This project is strongly inspired from [wkentaro/labelme](https://github.com/wkentaro/labelme), and thanks to [Ultralytics](https://www.ultralytics.com/).

