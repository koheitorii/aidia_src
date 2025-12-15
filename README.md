# Aidia
Aidia is a medical image annotation tool with AI development utilities.
Pre-built Windows packages of Aidia and documents are available at [TRK_Library](https://trklibrary.com/) (Japanese only).

## Features
- Building, training, evaluation and inference deep learning models for object detection and segmentation without coding.
- Image annotation for polygon, rectangle, polyline, line and point.
- Simply labeling by customized GUI buttons.
- DICOM format (.dcm) support including DICOM files which have no extention.
- Adjustment of brightness and contrast by mouse dragging like a DICOM viewer.

if you want to enable GPU computing, you need to install NVIDIA GPU and the suitable driver.

## Launch App
You need install Python 3.12 and run below:
```bash
python3 -m venv env # if not existed
source env/bin/activate
pip install -r requirements.txt # if not installed
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 # if not installed
python -m aidia
```

## Cite as
K. Torii. Report on the Open Access of Aidia : A Software for Supporting Medical Image AI Development. Journal of the Center for Community Engagement and Lifelong Learning, 34, 19-31, 2025.

K. Torii, R. Nishimura, E. Honda. Segmentation of Mandibular Canal on Dental Cone Beam CT Images with AI Development Support Software for Medical Images. Dental Radiology, 64(1), 11-19, 2024.

## Acknowledgement
This project is inspired from [wkentaro/labelme](https://github.com/wkentaro/labelme).
