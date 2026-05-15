import os
import shutil
import time
import random
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from onnxruntime import InferenceSession
from ultralytics import YOLO

from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Qt

from aidia import CLS, DET, SEG, TEST, CLEAR, ERROR, TASK_LIST, HOME_DIR
from aidia import LOCAL_DATA_DIR_NAME, CONFIG_JSON, DATASET_JSON
from aidia import ModelTypes
from aidia import aidia_logger
from aidia import qt
from aidia import utils
from aidia import errors
from aidia import image
from aidia.image import fig2img
from aidia.ai.config import AIConfig
from aidia.ai.dataset import Dataset
from aidia.ai.test import TestModel
from aidia.ai.task.detection import DetectionModel
from aidia.ai.task.segmentation import SegmentationModel
from aidia.ai.ai_utils import InferenceModel, InferenceModel_Ultralytics
from aidia.widgets import AIAugmentDialog
from aidia.widgets import AILabelReplaceDialog
from aidia.widgets import CopyDataDialog

import torch

# Set random seeds for reproducibility
seed = AIConfig().SEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set PyTorch to use the CPU or GPU
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def clear_session():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ParamComponent(object):
    """Base class for AI parameter components."""

    def __init__(self, type, tag, tips, validate_func=None, items=None, maximum_width=None):
        super().__init__()

        minimum_width = 200

        if type == "text":
            self.input_field = QtWidgets.QLineEdit()
            self.input_field.setPlaceholderText(tips)
            self.input_field.setToolTip(tips)
            self.input_field.setMinimumWidth(minimum_width)
            self.input_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if validate_func is not None:
                self.input_field.textChanged.connect(validate_func)
        elif type == "textbox":
            self.input_field = QtWidgets.QTextEdit()
            self.input_field.setPlaceholderText(tips)
            self.input_field.setToolTip(tips)
            self.input_field.setMinimumWidth(minimum_width)
            if maximum_width is not None:
                self.input_field.setMaximumWidth(maximum_width)
            if validate_func is not None:
                self.input_field.textChanged.connect(validate_func)
        elif type == "combo":
            self.input_field = QtWidgets.QComboBox()
            self.input_field.setToolTip(tips)
            self.input_field.setMinimumWidth(minimum_width)
            if items is not None:
                self.input_field.addItems(items)
            if validate_func is not None:
                self.input_field.currentIndexChanged.connect(validate_func)
        elif type == "checkbox":
            self.input_field = QtWidgets.QCheckBox(tag)
            self.input_field.setToolTip(tips)
            if validate_func is not None:
                self.input_field.stateChanged.connect(validate_func)
        else:
            return None

        self.tag = QtWidgets.QLabel(tag)
        self.state = CLEAR


class AITrainDialog(QtWidgets.QDialog):

    aiRunning = QtCore.Signal(bool)

    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowFlags(Qt.Window
                            | Qt.CustomizeWindowHint
                            | Qt.WindowTitleHint
                            | Qt.WindowCloseButtonHint
                            | Qt.WindowMaximizeButtonHint
                            )
        self.setWindowTitle("AI Workspace")

        self.setMinimumSize(QtCore.QSize(1200, 600))
        self.setWindowState(Qt.WindowState.WindowMaximized)

        self.dataset_dir = None
        self.target_logdir = None
        self.prev_log_name = None
        self.prev_dir = None
        self.start_time = 0
        self.start_epoch_time = 0
        self.end_epoch_time = 0
        self.epoch = []
        self.loss = []
        self.val_loss = []
        self.val_acc = []
        self.train_steps = 0
        self.val_steps = 0

        # Inference display settings
        self.show_labels = True
        self.show_conf = False

        # Create figures
        self.fig_loss, self.ax_loss = plt.subplots(figsize=(6, 6))
        self.fig_loss.patch.set_alpha(0.0)
        self.ax_loss.axis("off")

        self.fig_acc, self.ax_acc = plt.subplots(figsize=(6, 6))
        self.fig_acc.patch.set_alpha(0.0)
        self.ax_acc.axis("off")
        
        self.fig_pie, self.ax_pie = plt.subplots(figsize=(6, 6))
        self.fig_pie.patch.set_alpha(0.0)
        self.ax_pie.axis("off")

        self.param_objects = {}

        self.left_row_count = 0
        self.right_row_count = 0

        # Create main layout
        self._layout = QtWidgets.QGridLayout()

        title_main = qt.head_text(self.tr("Training Settings"))
        title_main.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        title_main.setMaximumHeight(30)
        self._layout.addWidget(title_main, 0, 1, 1, 4)
        self.left_row_count += 1
        self.right_row_count += 1

        self.label_current_mode = QtWidgets.QLabel()
        self.label_current_mode.setStyleSheet("QLabel{ font-size: 16px; }")
        self.label_current_mode.setMaximumHeight(30)
        self._layout.addWidget(self.label_current_mode, 1, 1, 1, 4, Qt.AlignmentFlag.AlignHCenter)
        self.left_row_count += 1
        self.right_row_count += 1

        # Experiment directory name
        def _validate(text):
            text = text.strip().replace(" ", "_")
            if text == "":
                self.set_error(self.param_name)
            else:
                self.set_ok(self.param_name)
                self.config.NAME = text
                self.config.build_params()

        self.param_name = ParamComponent(
            type="text",
            tag=self.tr("Experiment Directory Name"),
            tips=self.tr("Set the name of the experiment directory."),
            validate_func=_validate,
        )
        self.add_param_component(self.param_name)

        # Select experiment directory
        def _validate_logdir(idx):
            idx = int(idx)
            if idx < 0:
                return
            name = self.param_select_logdir.input_field.itemText(idx)
            self.target_logdir = os.path.join(self.dataset_dir, 'aidia_data', name)
            if name != self.config.NAME:
                self.config.NAME = name
                self.param_name.input_field.setText(name)
            self.load_experiment()

        self.param_select_logdir = ParamComponent(
            type="combo",
            tag=self.tr("Select Existing Directory"),
            tips=self.tr("Select an existing directory."),
            validate_func=_validate_logdir,
        )
        self.add_param_component(self.param_select_logdir)

        # Open Log Directory button
        self.button_open_logdir = QtWidgets.QPushButton(self.tr("Open Log Directory"))
        self.button_open_logdir.setToolTip(self.tr("Open the selected log directory."))
        self.button_open_logdir.setAutoDefault(False)
        self.button_open_logdir.setMaximumWidth(200)
        self.button_open_logdir.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.target_logdir)))
        self._layout.addWidget(self.button_open_logdir, self.left_row_count, 2, 1, 1)
        self.left_row_count += 1

        # Add hline
        hline = QtWidgets.QFrame()
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self._layout.addWidget(hline, self.left_row_count, 1, 1, 4)
        self.left_row_count += 1

        # Task selection
        def _validate(idx):
            idx = int(idx)
            self.config.TASK = TASK_LIST[idx]
            self.enable_params_by_task(self.config.TASK)
        self.param_task = ParamComponent(
            type="combo",
            tag=self.tr("Task"),
            tips=self.tr("""Select the task.
Detection uses YOLO and Segmentation uses U-Net.
If Performance Test are selected, the training test using MNIST dataset are performed and you can check the calculation performance."""),
            validate_func=_validate,
            items=[
                self.tr('Detection'),
                self.tr('Segmentation'),
                self.tr('Performance Test'),
            ]
        )
        self.add_param_component(self.param_task)

        # Model selection
        def _validate(idx):
            idx = int(idx)
            if self.config.TASK == DET:
                self.config.MODEL = ModelTypes.DET_MODEL[idx]
            elif self.config.TASK == SEG:
                self.config.MODEL = ModelTypes.SEG_MODEL[idx]
            else:
                self.config.MODEL = ''
            # Update augmentation parameters availability based on model
            self.update_augment_availability()
        self.param_model = ParamComponent(
            type="combo",
            tag=self.tr("Model"),
            tips=self.tr("""Select the model architecture."""),
            validate_func=_validate
        )
        self.add_param_component(self.param_model)

#         # dataset idx
#         def _validate(idx):
#             self.config.DATASET_NUM = int(idx+1)
#         self.param_dataset = ParamComponent(
#             type="combo",
#             tag=self.tr("Dataset"),
#             tips=self.tr("""Select the dataset pattern.
# Aidia splits the data into a 4:1 ratio (train:test) depend on the selected pattern.
# You can use this function for 5-fold cross-validation."""),
#             validate_func=_validate,
#             items=[
#                 self.tr("Pattern 1"),
#                 self.tr("Pattern 2"),
#                 self.tr("Pattern 3"),
#                 self.tr("Pattern 4"),
#                 self.tr("Pattern 5"),
#             ]
#         )
#         self.add_param_component(self.param_dataset)

        # input size
        def _validate(idx):
            self.config.INPUT_SIZE = int(self.param_input_size.input_field.itemText(idx))
        self.param_input_size = ParamComponent(
            type="combo",
            tag=self.tr("Input Size"),
            tips=self.tr("""Set the width and height of input images."""),
            validate_func=_validate,
            items=[
                "128", "160", "192", "224", "256",
                "320", "384", "448", "512", "576", "640", "704",
                "768", "832", "896", "960", "1024",
            ]
        )
        self.add_param_component(self.param_input_size)

        # epochs
        def _validate(idx):
            self.config.EPOCHS = int(self.param_epochs.input_field.itemText(idx))
        self.param_epochs = ParamComponent(
            type="combo",
            tag=self.tr("Epochs"),
            tips=self.tr("""Set the epochs.
If you set 100, all data are trained 100 times."""),
            validate_func=_validate,
            items=["10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000"]
        )
        self.add_param_component(self.param_epochs)

        # batch size
        def _validate(idx):
            self.config.BATCH_SIZE = int(self.param_batchsize.input_field.itemText(idx))
        self.param_batchsize = ParamComponent(
            type="combo",
            tag=self.tr("Batch Size"),
            tips=self.tr("""Set the batch size.
If you set 8, 8 samples are trained per step."""),
            validate_func=_validate,
            items=["1", "2", "4", "8", "16", "32", "64", "128", "256"]
        )
        self.add_param_component(self.param_batchsize)

        # learning rate
        def _validate(idx):
            self.config.LEARNING_RATE = float(self.param_lr.input_field.itemText(idx))
        self.param_lr = ParamComponent(
            type="combo",
            tag=self.tr("Learning Rate"),
            tips=self.tr("Set the initial learning rate."),
            validate_func=_validate,
            items=["0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]
        )
        self.add_param_component(self.param_lr)

        # Label definition
        def _validate():
            text = self.param_labels.input_field.toPlainText()
            text = text.strip().replace(" ", "")
            if len(text) == 0:
                self.set_error(self.param_labels)
                return
            parsed = text.split("\n")
            res = [p for p in parsed if p != ""]
            res = list(dict.fromkeys(res))   # delete duplicates
            if utils.is_full_width(text):  # error if the text includes 2-bytes codes.
                self.set_error(self.param_labels)
            else:
                self.set_ok(self.param_labels)
                self.config.LABELS = res
        self.param_labels = ParamComponent(
            type="textbox",
            tag=self.tr("Target Labels"),
            tips=self.tr("""Set target labels.
The labels are separated with line breaks."""),
            maximum_width=200,
            validate_func=_validate,
        )
        self.add_param_component(self.param_labels, right=True, custom_size=(2, 1))

        # Label replacement button
        button_replace_label = QtWidgets.QPushButton(self.tr("Label Replacement Setting"))
        button_replace_label.setAutoDefault(False)
        button_replace_label.setFixedWidth(200)
        button_replace_label.clicked.connect(self.open_label_replacement_dialog)
        self.button_replace_label = button_replace_label

        self._layout.addWidget(button_replace_label, self.right_row_count, 4, 1, 1, Qt.AlignmentFlag.AlignLeft)
        self.right_row_count += 1 + 1  # add space after the button for hline

        # Train target select
        def _validate(state): # check:2, empty:0
            if state == 2:
                self.config.DIR_SPLIT = True
            else:
                self.config.DIR_SPLIT = False
        self.param_is_dir_split = ParamComponent(
            type="checkbox",
            tag=self.tr("Separate Data by Directory"),
            tips=self.tr("""Separate data by directories when training."""),
            validate_func=_validate
        )
        self.add_param_component(self.param_is_dir_split, right=True)

        ### Add augment params ###
        self._augment_layout = QtWidgets.QVBoxLayout()
        self._augment_widget = QtWidgets.QWidget()
        self._augment_widget.setMaximumWidth(300)

        title_augment = qt.head_text(self.tr("Data Augmentation"))
        title_augment.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        title_augment.setMaximumHeight(30)
        self._augment_layout.addWidget(title_augment)

        # Vertical flip
        def _validate_vflip(state): # check:2, empty:0
            if state == 2:
                self.config.RANDOM_VFLIP = True
            else:
                self.config.RANDOM_VFLIP = False
        self.param_vflip = ParamComponent(
            type="checkbox",
            tag=self.tr("Vertical Flip"),
            tips=self.tr("Enable random vertical flip."),
            validate_func=_validate_vflip
        )
        self.add_augment_param(self.param_vflip)

        # Horizontal flip
        def _validate_hflip(state): # check:2, empty:0
            if state == 2:
                self.config.RANDOM_HFLIP = True
            else:
                self.config.RANDOM_HFLIP = False
        self.param_hflip = ParamComponent(
            type="checkbox",
            tag=self.tr("Horizontal Flip"),
            tips=self.tr("Enable random horizontal flip."),
            validate_func=_validate_hflip
        )
        self.add_augment_param(self.param_hflip)

        # Rotation
        def _validate_rotate(state): # check:2, empty:0
            if state == 2:
                if self.config.RANDOM_ROTATE == 0.0:
                    self.config.RANDOM_ROTATE = 0.1  # degrees
            else:
                self.config.RANDOM_ROTATE = 0
        self.param_rotate = ParamComponent(
            type="checkbox",
            tag=self.tr("Rotation"),
            tips=self.tr("Enable random rotation."),
            validate_func=_validate_rotate
        )
        self.add_augment_param(self.param_rotate)

        # Scale
        def _validate_scale(state): # check:2, empty:0
            if state == 2:
                if self.config.RANDOM_SCALE == 0.0:
                    self.config.RANDOM_SCALE = 0.1  # scale factor
            else:
                self.config.RANDOM_SCALE = 0.0
        self.param_scale = ParamComponent(
            type="checkbox",
            tag=self.tr("Scale"),
            tips=self.tr("Enable random scale variation."),
            validate_func=_validate_scale
        )
        self.add_augment_param(self.param_scale)

        # Shift
        def _validate_shift(state): # check:2, empty:0
            if state == 2:
                if self.config.RANDOM_SHIFT == 0.0:
                    self.config.RANDOM_SHIFT = 0.1  # pixels
            else:
                self.config.RANDOM_SHIFT = 0.0
        self.param_shift = ParamComponent(
            type="checkbox",
            tag=self.tr("Shift"),
            tips=self.tr("Enable random shift."),
            validate_func=_validate_shift
        )
        self.add_augment_param(self.param_shift)

        # Shear
        def _validate_shear(state): # check:2, empty:0
            if state == 2:
                if self.config.RANDOM_SHEAR == 0.0:
                    self.config.RANDOM_SHEAR = 0.1  # degrees
            else:
                self.config.RANDOM_SHEAR = 0.0
        self.param_shear = ParamComponent(
            type="checkbox",
            tag=self.tr("Shear"),
            tips=self.tr("Enable random shear."),
            validate_func=_validate_shear
        )
        self.add_augment_param(self.param_shear)

        # Brightness
        def _validate_brightness(state): # check:2, empty:0
            if state == 2:
                if self.config.RANDOM_BRIGHTNESS == 0.0:
                    self.config.RANDOM_BRIGHTNESS = 0.1  # brightness range
            else:
                self.config.RANDOM_BRIGHTNESS = 0.0
        self.param_brightness = ParamComponent(
            type="checkbox",
            tag=self.tr("Brightness"),
            tips=self.tr("Enable random brightness adjustment."),
            validate_func=_validate_brightness
        )
        self.add_augment_param(self.param_brightness)

        # Contrast
        def _validate_contrast(state): # check:2, empty:0
            if state == 2:
                if self.config.RANDOM_CONTRAST == 0.0:
                    self.config.RANDOM_CONTRAST = 0.1
            else:
                self.config.RANDOM_CONTRAST = 0.0
        self.param_contrast = ParamComponent(
            type="checkbox",
            tag=self.tr("Contrast"),
            tips=self.tr("Enable random contrast variation."),
            validate_func=_validate_contrast
        )
        self.add_augment_param(self.param_contrast)

        # Blur
        def _validate_blur(state): # check:2, empty:0
            if state == 2:
                if self.config.RANDOM_BLUR == 0.0:
                    self.config.RANDOM_BLUR = 0.1  # sigma
            else:
                self.config.RANDOM_BLUR = 0.0
        self.param_blur = ParamComponent(
            type="checkbox",
            tag=self.tr("Blur"),
            tips=self.tr("Enable random blur."),
            validate_func=_validate_blur
        )
        self.add_augment_param(self.param_blur)

        # Noise
        def _validate_noise(state): # check:2, empty:0
            if state == 2:
                if self.config.RANDOM_NOISE == 0.0:
                    self.config.RANDOM_NOISE = 0.1  # stddev
            else:
                self.config.RANDOM_NOISE = 0.0
        self.param_noise = ParamComponent(
            type="checkbox",
            tag=self.tr("Noise"),
            tips=self.tr("Enable random noise."),
            validate_func=_validate_noise
        )
        self.add_augment_param(self.param_noise)

        # Advanced settings
        button_advanced = QtWidgets.QPushButton(self.tr("Advanced Settings"))
        button_advanced.setToolTip(self.tr("Open advanced data augmentation settings."))
        button_advanced.setAutoDefault(False)
        button_advanced.clicked.connect(self.augment_setting_popup)
        self._augment_layout.addWidget(button_advanced)
        self.button_advanced = button_advanced

        # Update lowest row
        row_count = max(self.left_row_count, self.right_row_count)

        # Train button
        self.button_train = QtWidgets.QPushButton(self.tr("🚀 Train"))
        self.button_train.setAutoDefault(False)
        self.button_train.setMinimumHeight(80)
        self.button_train.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #4CAF50, stop:1 #45a049);
                border: 3px solid #3d8b40;
                border-radius: 12px;
                padding: 10px;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #5CBF60, stop:1 #55b059);
                border: 3px solid #4d9b50;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #3d8b40, stop:1 #357a35);
                padding-top: 12px;
                padding-bottom: 8px;
            }
            QPushButton:disabled {
                background: #cccccc;
                border: 3px solid #999999;
                color: #666666;
            }
        """)
        self.button_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button_train.clicked.connect(self.train)
        self._layout.addWidget(self.button_train, row_count, 1, 1, 4)
        row_count += 1

        # Loss graph
        self.image_widget_loss = qt.ImageWidget(self)
        self.image_widget_loss.setMinimumHeight(300)
        self._layout.addWidget(self.image_widget_loss, row_count, 1, 1, 2)

        # Accuracy graph
        self.image_widget_acc = qt.ImageWidget(self)
        self.image_widget_acc.setMinimumHeight(300)
        self._layout.addWidget(self.image_widget_acc, row_count, 3, 1, 2)
        row_count += 1

        # Progress bar
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 20px;
            }
        """)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self._layout.addWidget(self.progress, row_count, 1, 1, 4)
        row_count += 1

        # Status text
        self.text_status = QtWidgets.QLabel()
        self.text_status.setMaximumHeight(32)
        self._layout.addWidget(self.text_status, row_count, 1, 1, 4)

        ### Dataset information ###
        self._dataset_layout = QtWidgets.QVBoxLayout()
        self._dataset_widget = QtWidgets.QWidget()
        self._dataset_widget.setMaximumWidth(300)

        title_dataset = qt.head_text(self.tr("Dataset Information"))
        title_dataset.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        title_dataset.setMaximumHeight(30)
        self._dataset_layout.addWidget(title_dataset)

        # Dataset information
        self.text_dataset = QtWidgets.QLabel()
        self.text_dataset.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.text_dataset.setTextFormat(Qt.TextFormat.RichText)
        self.text_dataset.setWordWrap(True)
        self.text_dataset.setStyleSheet("QLabel { font-size: 12px; line-height: 150%; }")
        self._dataset_layout.addWidget(self.text_dataset)

        # Class information table
        self.table_classes = QtWidgets.QTableWidget()
        self.table_classes.setColumnCount(6)
        self.table_classes.setHorizontalHeaderLabels([
            "ID", 
            self.tr("Label"), 
            self.tr("All"), 
            self.tr("Train"), 
            self.tr("Val"), 
            self.tr("Test")
        ])
        self.table_classes.horizontalHeader().setStretchLastSection(False)
        self.table_classes.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_classes.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_classes.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_classes.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_classes.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_classes.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_classes.verticalHeader().setVisible(False)
        self.table_classes.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_classes.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_classes.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table_classes.setAlternatingRowColors(True)
        self.table_classes.setMaximumHeight(200)
        self._dataset_layout.addWidget(self.table_classes)

        self.image_widget_pie = qt.ImageWidget(self)
        self.image_widget_pie.setMinimumHeight(300)
        self._dataset_layout.addWidget(self.image_widget_pie)

        # utility layout
        self._utility_layout = QtWidgets.QVBoxLayout()
        self._utility_widget = QtWidgets.QWidget()
        self._utility_widget.setMaximumWidth(300)

        title_utility = qt.head_text(self.tr("Utilities"))
        title_utility.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self._utility_layout.addWidget(title_utility)

        # Prediction settings group box
        self.prediction_group = QtWidgets.QGroupBox(self.tr("Prediction"))
        self.prediction_layout = QtWidgets.QVBoxLayout(self.prediction_group)
        
        # Prediction display settings
        self.checkbox_show_labels = QtWidgets.QCheckBox(self.tr("Show Labels"))
        self.checkbox_show_labels.setToolTip(self.tr("Show labels in prediction results."))
        self.checkbox_show_labels.setChecked(True)
        def _on_show_labels_changed(state):
            if state == 2:
                self.show_labels = True
                self.checkbox_show_conf.setEnabled(True)
            else:
                self.show_labels = False
                self.checkbox_show_conf.setChecked(False)
                self.checkbox_show_conf.setEnabled(False)
                self.show_conf = False
        self.checkbox_show_labels.stateChanged.connect(_on_show_labels_changed)
        self.prediction_layout.addWidget(self.checkbox_show_labels)

        self.checkbox_show_conf = QtWidgets.QCheckBox(self.tr("Show Confidence"))
        self.checkbox_show_conf.setToolTip(self.tr("Show confidence scores in prediction results."))
        self.checkbox_show_conf.setChecked(True)
        def _on_show_conf_changed(state):
            if self.show_labels is False:
                self.checkbox_show_conf.setChecked(False)
                self.checkbox_show_conf.setEnabled(False)
                self.show_conf = False
                return
            if state == 2:
                self.show_conf = True
            else:
                self.show_conf = False
        self.checkbox_show_conf.stateChanged.connect(_on_show_conf_changed)
        self.prediction_layout.addWidget(self.checkbox_show_conf)

        self.button_pred = QtWidgets.QPushButton(self.tr("Select Image"))
        self.button_pred.setToolTip(self.tr("Predict a single image."))
        self.button_pred.setAutoDefault(False)
        self.button_pred.clicked.connect(self.predict_image)
        self.prediction_layout.addWidget(self.button_pred)

        self.button_pred_dir = QtWidgets.QPushButton(self.tr("Select Directory"))
        self.button_pred_dir.setToolTip(self.tr("Predict images in the directory you selected."))
        self.button_pred_dir.setAutoDefault(False)
        self.button_pred_dir.clicked.connect(self.predict_images_from_directory)
        self.prediction_layout.addWidget(self.button_pred_dir)
        
        # Add prediction group to utility layout
        self._utility_layout.addWidget(self.prediction_group)

        # Export model button
        # self.button_export_model = QtWidgets.QPushButton(self.tr("Export ONNX"))
        # self.button_export_model.setAutoDefault(False)
        # self.button_export_model.clicked.connect(self.export_model)
        # self._utility_layout.addWidget(self.button_export_model)

        # Export model to pretrained button
        self.button_export_model_to_pretrained = QtWidgets.QPushButton(self.tr("Export for Auto Annotation"))
        self.button_export_model_to_pretrained.setAutoDefault(False)
        self.button_export_model_to_pretrained.clicked.connect(self.export_model_to_pretrained)
        self._utility_layout.addWidget(self.button_export_model_to_pretrained)

        # Connect AI prediction thread
        self.ai_pred = AIPredThread(self)
        self.ai_pred.notifyMessage.connect(self.update_pred_status)
        self.ai_pred.progressValue.connect(self.update_pred_progress)
        self.ai_pred.finished.connect(self.ai_pred_finished)

        ### set layouts ###
        self._dataset_widget.setLayout(self._dataset_layout)
        self._layout.addWidget(self._dataset_widget, 0, 0, row_count + 1, 1, Qt.AlignmentFlag.AlignTop)

        self._augment_widget.setLayout(self._augment_layout)
        self._layout.addWidget(self._augment_widget, 0, 5, row_count - 2, 1, Qt.AlignmentFlag.AlignTop)

        self._utility_widget.setLayout(self._utility_layout)
        self._layout.addWidget(self._utility_widget, row_count - 2, 5, 3, 1, Qt.AlignmentFlag.AlignBottom)
        
        self.setLayout(self._layout)

        # connect AI thread
        self.ai = AITrainThread(self)
        # self.ai.fitStarted.connect(self.callback_fit_started)
        self.ai.notifyMessage.connect(self.update_status)
        self.ai.errorMessage.connect(self.popup_error)
        self.ai.datasetInfo.connect(self.update_dataset)
        self.ai.epochLogList.connect(self.update_logs)
        self.ai.batchLogList.connect(self.update_batch)
        self.ai.finished.connect(self.ai_finished)

        self.text_status.setText("Ready")

    def popup(self, dataset_dir, is_submode=False, data_labels=None):
        """Popup train window and set config parameters to input fields.
        
        Args:
            dataset_dir (str): Dataset directory path.
            is_submode (bool): If True, the training data is searched under the parent directory of the dataset directory. Otherwise, the training data is searched under the dataset directory.
            data_labels (list): List of labels in the dataset. This is used to set default label definition.
        """
        self.dataset_dir = dataset_dir

        # create data directory
        data_dirpath = utils.get_dirpath_with_mkdir(dataset_dir, 'aidia_data')

        # load config parameters
        self.config = AIConfig(dataset_dir)
        config_path = os.path.join(data_dirpath, 'config.json')
        if os.path.exists(config_path):
            try:
                self.config.load(config_path)
            except Exception as e:
                aidia_logger.error(e)

        self.config.SUBMODE = is_submode
        if is_submode:
            self.label_current_mode.setText(self.tr('Search data under <span style="color: red;"><b>PARENT</b></span> directory'))
        else:
            self.label_current_mode.setText(self.tr('Search data of <span style="color: green;"><b>CURRENT</b></span> directory'))

        # Store previous log directory for later use (e.g., to delete or move if the name is changed)
        self.prev_log_name = self.config.NAME

        # Basic params
        self.param_task.input_field.setCurrentIndex(TASK_LIST.index(self.config.TASK))
        self.enable_params_by_task(self.config.TASK)
        self.param_model.input_field.setCurrentText(self.config.MODEL)
        self.param_name.input_field.setText(self.config.NAME)
        # self.param_dataset.input_field.setCurrentIndex(int(self.config.DATASET_NUM) - 1)

        # Input size - use combo box
        if self.config.INPUT_SIZE in [int(self.param_input_size.input_field.itemText(i)) for i in range(self.param_input_size.input_field.count())]:
            self.param_input_size.input_field.setCurrentText(str(self.config.INPUT_SIZE))
        else:
            self.param_input_size.input_field.setCurrentIndex(0)  # Reset to first item if not found

        # Epochs - use combo box
        if str(self.config.EPOCHS) in [self.param_epochs.input_field.itemText(i) for i in range(self.param_epochs.input_field.count())]:
            self.param_epochs.input_field.setCurrentText(str(self.config.EPOCHS))
        else:
            self.param_epochs.input_field.setCurrentIndex(5)  # Default to index 5 (value "100")
        
        # Batch size - use combo box
        if str(self.config.BATCH_SIZE) in [self.param_batchsize.input_field.itemText(i) for i in range(self.param_batchsize.input_field.count())]:
            self.param_batchsize.input_field.setCurrentText(str(self.config.BATCH_SIZE))
        else:
            self.param_batchsize.input_field.setCurrentIndex(3)  # Default to index 3 (value "8")
        
        # Learning rate - use combo box
        if str(self.config.LEARNING_RATE) in [self.param_lr.input_field.itemText(i) for i in range(self.param_lr.input_field.count())]:
            self.param_lr.input_field.setCurrentText(str(self.config.LEARNING_RATE))
        else:
            self.param_lr.input_field.setCurrentIndex(2)  # Default to index 2 (value "0.001")

        # Label definition
        if len(self.config.LABELS) > 0:
            self.param_labels.input_field.setText("\n".join(self.config.LABELS))
        else:
            self.param_labels.input_field.setText("\n".join(data_labels))

        # Data separation
        if not self.config.SUBMODE:
            self.param_is_dir_split.input_field.setEnabled(False)
        self.param_is_dir_split.input_field.setChecked(self.config.DIR_SPLIT)

        # Inference display settings
        self.checkbox_show_labels.setChecked(self.config.SHOW_LABELS)
        if not self.config.SHOW_LABELS:
            self.checkbox_show_conf.setChecked(False)
            self.checkbox_show_conf.setEnabled(False)
            self.show_labels = False
            self.show_conf = False
        else:
            self.checkbox_show_conf.setChecked(self.config.SHOW_CONF)

        self.update_augment_checkboxes()
        self.update_augment_availability()
        self.update_logdir_list()

        # Reset status and progress
        self.load_experiment()

        self.exec_()
        if os.path.exists(os.path.join(dataset_dir, 'aidia_data')):
            self.config.save(config_path)
    
    def load_experiment(self):
        # Load dataset information
        dataset_json_path = os.path.join(self.config.log_dir, 'dataset.json')
        if os.path.exists(dataset_json_path):
            try:
                dataset_info = json.load(open(dataset_json_path, "r"))
                self.update_dataset(dataset_info)
            except Exception as e:
                aidia_logger.error(e)

        # Load loss graph
        self.image_widget_loss.clear()
        loss_graph_path = os.path.join(self.config.log_dir, "loss.png")
        if os.path.exists(loss_graph_path):
            self.image_widget_loss.set_image(loss_graph_path, alpha=True)

        # Load accuracy graph
        self.image_widget_acc.clear()
        acc_graph_path = os.path.join(self.config.log_dir, "acc.png")
        if os.path.exists(acc_graph_path):
            self.image_widget_acc.set_image(acc_graph_path, alpha=True)
    
    ### Callbacks ###
    def ai_finished(self):
        """Call back function when AI thread finished."""
        self.enable_params_by_task(self.config.TASK)

        # Set log name to previous if the current log directory does not exist (e.g., when the log directory is deleted due to name change during training)
        self.prev_log_name = self.config.NAME

        # Clear cuda cache
        clear_session()
        
        # Raise error handle
        config_path = os.path.join(self.config.log_dir, 'config.json')
        dataset_path = os.path.join(self.config.log_dir, 'dataset.json')
        if not os.path.exists(config_path) or not os.path.exists(dataset_path):
            # self.text_status.setText(self.tr("Training was failed."))
            self.reset_state()
            self.aiRunning.emit(False)
            self.text_status.setText(self.tr("Terminated training."))
            return
        
        # Display elapsed time
        now = time.time()
        etime = now - self.start_time
        h = int(etime // 3600)
        m = int(etime // 60 % 60)
        s = int(etime % 60)
        self.text_status.setText(self.tr("Done -- Elapsed time: {}h {}m {}s").format(h, m, s))

        # Save metrics
        df_dic = {
            "epoch": self.epoch,
            "loss": self.loss,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
        }
        utils.save_dict_to_excel(df_dic, os.path.join(self.config.log_dir, "loss.xlsx"))

        # Save figure
        self.fig_loss.savefig(os.path.join(self.config.log_dir, "loss.png"))
        self.fig_acc.savefig(os.path.join(self.config.log_dir, "acc.png"))

        # convet YOLO model to ONNX
        if self.config.is_ultralytics():
            from aidia.ai.ai_utils import write_onnx_u
            try:
                model_path = os.path.join(self.config.log_dir, self.config.MODEL, "weights", "best.pt")
                onnx_path = write_onnx_u(model_path)
            except Exception as e:
                aidia_logger.error(e)
                self.text_status.setText(self.tr("Failed to convert to ONNX model."))
                self.popup_error(self.tr("Failed to convert to ONNX model. Please check the model and try again."))
            else:
                shutil.move(onnx_path, os.path.join(self.config.log_dir, "model.onnx"))

        self.aiRunning.emit(False)
        
        self.update_logdir_list()
        self.switch_utility()

        self.start_epoch_time = 0
        self.end_epoch_time = 0

        # open log directory
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.config.log_dir))

    def ai_pred_finished(self):
        self.enable_params_by_task(self.config.TASK)
        self.aiRunning.emit(False)

        # open prediction result directory
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._result_dir))

    def enable_params_by_task(self, task):
        """Switch enabled state of parameters by task."""
        if task == CLS:
            raise NotImplementedError
        
        elif task in [DET, SEG]:
            self.param_model.input_field.clear()
            if task == DET:
                self.param_model.input_field.addItems(ModelTypes.DET_MODEL)
            elif task == SEG:
                self.param_model.input_field.addItems(ModelTypes.SEG_MODEL)
            self._enable_params()
        elif task == TEST:
            self.param_model.input_field.clear()
            self.disable_params()
            self.switch_enabled([
                self.param_name,
                self.param_batchsize,
                self.param_epochs,
                self.param_lr,
                self.param_task], True)
        else:
            raise ValueError

        # global setting
        self.switch_global_params()
        # Update augmentation parameter availability
        self.update_augment_availability()
        self.button_train.setEnabled(True)

        self.button_replace_label.setEnabled(True)
        self.button_advanced.setEnabled(True)
        # self.button_stop.setEnabled(False)

        self.switch_utility()

    def switch_utility(self):
        """Switch enabled state of utility components."""
        if self.param_select_logdir.input_field.count() == 0:
            self.disable_utility()
        else:
            self.enable_utility()

    def switch_enabled(self, targets: list[ParamComponent], enabled:bool):
        for obj in targets:
            if enabled:
                obj.tag.setStyleSheet(qt.LabelColor.get_style("default"))
            else:
                obj.tag.setStyleSheet(qt.LabelColor.get_style("disabled"))
            obj.input_field.setEnabled(enabled)
        if enabled and not self.config.SUBMODE:
            self.param_is_dir_split.tag.setStyleSheet(qt.LabelColor.get_style("disabled"))
            self.param_is_dir_split.input_field.setEnabled(False)

    def switch_global_params(self):
        if not self.config.SUBMODE or self.config.TASK in [TEST]:
            self.param_is_dir_split.tag.setStyleSheet(qt.LabelColor.get_style("disabled"))
            self.param_is_dir_split.input_field.setEnabled(False)
        else:
            self.param_is_dir_split.tag.setStyleSheet(qt.LabelColor.get_style("default"))
            self.param_is_dir_split.input_field.setEnabled(True)
    
    def _enable_params(self):
        """Enable all components."""
        for obj in self.param_objects.values():
            obj.input_field.setEnabled(True)
            obj.tag.setStyleSheet(qt.LabelColor.get_style("default"))
            if obj.state == ERROR:
                obj.tag.setStyleSheet(qt.LabelColor.get_style("error"))

    def enable_utility(self):
        """Enable utility components."""
        self.param_select_logdir.input_field.setEnabled(True)
        self.button_open_logdir.setEnabled(True)
        self.checkbox_show_labels.setEnabled(True)
        if not self.show_labels:
            self.checkbox_show_conf.setChecked(False)
            self.checkbox_show_conf.setEnabled(False)
            self.show_conf = False
        else:
            self.checkbox_show_conf.setEnabled(True)
        self.button_pred.setEnabled(True)
        self.button_pred_dir.setEnabled(True)
        # self.button_export_model.setEnabled(True)
        self.button_export_model_to_pretrained.setEnabled(True)
    
    def disable_utility(self):
        """Disable utility components."""
        self.param_select_logdir.input_field.setEnabled(False)
        self.button_open_logdir.setEnabled(False)
        self.checkbox_show_labels.setEnabled(False)
        self.checkbox_show_conf.setEnabled(False)
        self.button_pred.setEnabled(False)
        self.button_pred_dir.setEnabled(False)
        # self.button_export_model.setEnabled(False)
        self.button_export_model_to_pretrained.setEnabled(False)
    
    def disable_params(self):
        """Disable all components."""
        for obj in self.param_objects.values():
            obj.input_field.setEnabled(False)
            obj.tag.setStyleSheet(qt.LabelColor.get_style("disabled"))
        self.button_replace_label.setEnabled(False)
        self.button_advanced.setEnabled(False)
        self.button_train.setEnabled(False)
        
        # Disable utility components
        self.param_select_logdir.input_field.setEnabled(False)
        self.button_open_logdir.setEnabled(False)
        self.checkbox_show_labels.setEnabled(False)
        self.checkbox_show_conf.setEnabled(False)
        self.button_pred.setEnabled(False)
        self.button_pred_dir.setEnabled(False)
        # self.button_export_model.setEnabled(False)
        self.button_export_model_to_pretrained.setEnabled(False)

    def closeEvent(self, event):
        """Handle close event."""
        pass
        
    def showEvent(self, event):
        """Handle show event."""
        if self.ai.isRunning():
            self.disable_params()
            # self.button_stop.setEnabled(True)
        else:
            # self.reset_state()
            self.enable_params_by_task(self.config.TASK)
    
    def open_label_replacement_dialog(self):
        """Open label replacement dialog."""
        dialog = AILabelReplaceDialog(self)
        result = dialog.popup(self.config.REPLACE_DICT)
  
    def add_param_component(self, obj:ParamComponent, right=False, custom_size=None):
        """Add a parameter component to the layout."""
        self.param_objects[obj.tag.text()] = obj
        row = self.left_row_count
        pos = [1, 2]
        h, w = (1, 1)
        if right:
            row = self.right_row_count
            pos = [3, 4]
        if custom_size:
            h = custom_size[0]
            w = custom_size[1]
        if isinstance(obj.input_field, QtWidgets.QCheckBox):
            # checkbox is larger than other components
            self._layout.addWidget(obj.input_field, row, pos[0], h, w+1)
        else:
            self._layout.addWidget(obj.tag, row, pos[0], h, w, alignment=Qt.AlignmentFlag.AlignRight)
            self._layout.addWidget(obj.input_field, row, pos[1], h, w, alignment=Qt.AlignmentFlag.AlignLeft)
        if right:
            self.right_row_count += h
        else:
            self.left_row_count += h
    
    def add_augment_param(self, obj:ParamComponent):
        """Add a parameter component to the augmentation layout."""
        self.param_objects[obj.tag.text()] = obj
        self._augment_layout.addWidget(obj.input_field, alignment=Qt.AlignmentFlag.AlignLeft)
   
    def set_error(self, obj: ParamComponent):
        """Set error state to the parameter component."""
        obj.tag.setStyleSheet(qt.LabelColor.get_style("error"))
        obj.state = ERROR

    def set_ok(self, obj: ParamComponent):
        """Set ok state to the parameter component."""
        obj.tag.setStyleSheet(qt.LabelColor.get_style("default"))
        obj.state = CLEAR

    def update_figure(self):
        """Update the figure for loss."""

    def update_dataset(self, dataset_info: dict):
        # Get dataset informationfrom the dictionary
        dataset_num = dataset_info["dataset_num"]
        num_images = dataset_info["num_images"]
        num_shapes = dataset_info["num_shapes"]
        num_classes = dataset_info["num_classes"]
        num_train = dataset_info["num_train"]
        num_val = dataset_info["num_val"]
        num_test = dataset_info["num_test"]
        class_names = dataset_info["class_names"]
        num_per_class = dataset_info["num_per_class"]
        train_per_class = dataset_info["train_per_class"]
        val_per_class = dataset_info["val_per_class"]
        test_per_class = dataset_info["test_per_class"]
        path_dataset = dataset_info.get("path_dataset", "")

        # Set train and validation steps for progress tracking
        self.train_steps = dataset_info["train_steps"]
        self.val_steps = dataset_info["val_steps"]

        # Build HTML formatted text with better readability
        html = []
        html.append("<html><body style='font-family: sans-serif;'>")
        
        # Dataset overview section
        html.append("<div style='margin-bottom: 15px;'>")
        if path_dataset:
            html.append(f"<p style='margin: 3px 0;'><b>{self.tr('Directory')}:</b> <span style='font-size: 11px;'>{path_dataset}</span></p>")
        # Submode information
        submode_status = self.tr("ON") if self.config.SUBMODE else self.tr("OFF")
        submode_color = "#27ae60" if self.config.SUBMODE else "#95a5a6"
        html.append(f"<p style='margin: 3px 0;'><b>{self.tr('Submode')}:</b> <span style='color: {submode_color}; font-weight: bold;'>{submode_status}</span></p>")
        # html.append(f"<p style='margin: 3px 0;'><b>{self.tr('Dataset Number')}:</b> <span style='color: #3498db; font-weight: bold;'>{dataset_num}</span></p>")
        html.append(f"<p style='margin: 3px 0;'><b>{self.tr('Number of Data')}:</b> <span style='color: #2ecc71; font-weight: bold;'>{num_images}</span></p>")
        # html.append(f"<p style='margin: 3px 0;'><b>{self.tr('Number of Shapes')}:</b> <span style='color: #2ecc71; font-weight: bold;'>{num_shapes}</span></p>")
        html.append("</div>")
        
        # Data split section
        html.append("<div style='margin-bottom: 15px; padding: 8px; background-color: rgba(52, 152, 219, 0.1); border-left: 3px solid #3498db;'>")
        html.append(f"<p style='margin: 3px 0;'><b>{self.tr('Number of Train')}:</b> <span style='color: #e74c3c; font-weight: bold;'>{num_train}</span></p>")
        html.append(f"<p style='margin: 3px 0;'><b>{self.tr('Number of Validation')}:</b> <span style='color: #f39c12; font-weight: bold;'>{num_val}</span></p>")
        html.append(f"<p style='margin: 3px 0;'><b>{self.tr('Number of Test')}:</b> <span style='color: #9b59b6; font-weight: bold;'>{num_test}</span></p>")
        
        if self.config.SUBMODE and self.config.DIR_SPLIT:
            html.append(f"<p style='margin: 3px 0; margin-top: 8px;'><i>{self.tr('Number of Train Directories')}:</i> {dataset_info['num_train_subdir']}</p>")
            html.append(f"<p style='margin: 3px 0;'><i>{self.tr('Number of Validation Directories')}:</i> {dataset_info['num_val_subdir']}</p>")
            html.append(f"<p style='margin: 3px 0;'><i>{self.tr('Number of Test Directories')}:</i> {dataset_info['num_test_subdir']}</p>")
        html.append("</div>")
        
        # Training steps section
        html.append("<div style='margin-bottom: 15px;'>")
        html.append(f"<p style='margin: 3px 0;'><b>{self.tr('Train Steps')}:</b> {self.train_steps}</p>")
        html.append(f"<p style='margin: 3px 0;'><b>{self.tr('Validation Steps')}:</b> {self.val_steps}</p>")
        html.append("</div>")
        
        html.append("</body></html>")
        
        text = "".join(html)
        self.text_dataset.setText(text)

        # Update class information table
        self.table_classes.setRowCount(num_classes)
        for i in range(num_classes):
            # ID column
            id_item = QtWidgets.QTableWidgetItem(f"{i}")
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            id_item.setFont(QtGui.QFont("", -1, QtGui.QFont.Weight.Bold))
            self.table_classes.setItem(i, 0, id_item)
            
            # Label column
            label_item = QtWidgets.QTableWidgetItem(class_names[i])
            label_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_classes.setItem(i, 1, label_item)
            
            # All column
            all_item = QtWidgets.QTableWidgetItem(str(num_per_class[i]))
            all_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            all_item.setFont(QtGui.QFont("", -1, QtGui.QFont.Weight.Bold))
            self.table_classes.setItem(i, 2, all_item)
            
            # Train column
            train_item = QtWidgets.QTableWidgetItem(str(train_per_class[i]))
            train_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            train_item.setForeground(QtGui.QColor("#e74c3c"))
            self.table_classes.setItem(i, 3, train_item)
            
            # Val column
            val_item = QtWidgets.QTableWidgetItem(str(val_per_class[i]))
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            val_item.setForeground(QtGui.QColor("#f39c12"))
            self.table_classes.setItem(i, 4, val_item)
            
            # Test column
            test_item = QtWidgets.QTableWidgetItem(str(test_per_class[i]))
            test_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            test_item.setForeground(QtGui.QColor("#9b59b6"))
            self.table_classes.setItem(i, 5, test_item)

        # Update label distribution
        self.ax_pie.clear()
        self.ax_pie.pie(num_per_class,
                    labels=class_names,
                    #  autopct="%1.1f%%",
                    wedgeprops={'linewidth': 1, 'edgecolor': qt.LabelColor.get_color("default")},
                    textprops={'color': qt.LabelColor.get_color("default"),
                               'fontsize': 20})
        self.image_widget_pie.loadPixmap(fig2img(self.fig_pie, add_alpha=True))
        self.image_widget_pie.setMinimumHeight(300)

    def update_status(self, value):
        """Update status text."""
        self.text_status.setText(str(value))

    def popup_error(self, text):
        """Popup error message."""
        self.parent().error_message(text)
    
    def popup_info(self, text):
        """Popup information message."""
        self.parent().info_message(text)

    def update_batch(self, value):
        """Update batch status."""
        epoch = len(self.epoch) + 1
        batch = value.get("batch")
        loss = value.get("loss")
        if self.config.TASK in [DET]:
            acc = value.get("map50")
        else:
            acc = value.get("acc")

        text = f"epoch: {epoch:>4}/{self.config.EPOCHS} "
        if batch is not None:
            if batch > self.train_steps:
                batch = self.train_steps
            text += f"batch: {batch:>6} / {self.train_steps} "
        if loss is not None:
            text += f"loss: {loss:>8.4f} "
        if acc is not None:
            text += f"acc: {acc:>8.4f} "
        if len(self.val_loss):
            text += f"val_loss: {self.val_loss[-1]:>8.4f} "
        if len(self.val_acc):
            text += f"val_acc: {self.val_acc[-1]:>8.4f}"

        # Calculate estimated remaining time based on batch progress
        if self.start_epoch_time != 0 and self.end_epoch_time != 0:
            batch_time = (self.end_epoch_time - self.start_epoch_time) / self.train_steps
            remaining_time = batch_time * (self.config.EPOCHS - epoch + 1) * self.train_steps + batch_time * (self.train_steps - batch)
            
            # Format remaining time
            remaining_h = int(remaining_time // 3600)
            remaining_m = int(remaining_time // 60 % 60)
            remaining_s = int(remaining_time % 60)
            
            # Add remaining time to status text
            if remaining_s > 0:
                text += f" - ETA: {remaining_h}h {remaining_m}m {remaining_s}s"
            elif remaining_h > 0:
                text += f" - ETA: {remaining_h}h {remaining_m}m"
            elif remaining_m > 0:
                text += f" - ETA: {remaining_m}m"
        
        self.text_status.setText(text)
    
    def update_logs(self, value):
        """Update training logs."""
        epoch = value.get("epoch")
        loss = value.get("loss")
        val_loss = value.get("val_loss")
        if self.config.TASK in [DET]:
            acc = value.get("map50")
            val_acc = value.get("val_map50")
        else:
            acc = value.get("acc")
            val_acc = value.get("val_acc")

        progress_value = int(epoch / self.config.EPOCHS * 100)

        if epoch is not None:
            self.epoch.append(epoch)
            self.progress.setValue(progress_value)

        if loss is not None:
            self.loss.append(loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        
        if val_acc is not None:
            self.val_acc.append(val_acc)

        if self.start_epoch_time == 0:
            self.start_epoch_time = time.time()
        elif self.end_epoch_time == 0:
            self.end_epoch_time = time.time()

        fontsize = 16

        if len(self.epoch):
            # Update loss figure
            if len(self.loss) or len(self.val_loss):
                self.ax_loss.clear()
                self.ax_loss.set_xlabel("Epoch", fontsize=fontsize, color=qt.LabelColor.get_color("default"))
                self.ax_loss.set_ylabel("Loss", fontsize=fontsize, color=qt.LabelColor.get_color("default"))
                self.ax_loss.tick_params(axis='both', labelsize=fontsize//2, colors=qt.LabelColor.get_color("default"))
                self.ax_loss.spines['top'].set_visible(False)
                self.ax_loss.spines['right'].set_visible(False)
                self.ax_loss.spines['left'].set_color(qt.LabelColor.get_color("default"))
                self.ax_loss.spines['bottom'].set_color(qt.LabelColor.get_color("default"))
                self.ax_loss.patch.set_alpha(0.0)
                self.ax_loss.grid(alpha=0.3, color=qt.LabelColor.get_color("default"), linestyle="--", linewidth=1)
                self.ax_loss.plot(self.epoch, self.loss, color="red", linestyle="solid", label="train")
                self.ax_loss.plot(self.epoch, self.val_loss, color="green", linestyle="solid", label="val")
                self.ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
                mx = min((len(self.epoch) // 10 + 1) * 10, self.config.EPOCHS)
                self.ax_loss.set_xlim([1, mx])
                self.ax_loss.legend(fontsize=fontsize, labelcolor=qt.LabelColor.get_color("default"), frameon=False)
                self.image_widget_loss.loadPixmap(fig2img(self.fig_loss, add_alpha=True))

            # Update accuracy figure if available
            if len(self.val_acc):
                self.ax_acc.clear()
                self.ax_acc.set_xlabel("Epoch", fontsize=fontsize, color=qt.LabelColor.get_color("default"))
                if self.config.is_ultralytics():
                    y_label = "mAP50"
                else:
                    y_label = "Accuracy"
                self.ax_acc.set_ylabel(y_label, fontsize=fontsize, color=qt.LabelColor.get_color("default"))
                self.ax_acc.tick_params(axis='both', labelsize=fontsize//2, colors=qt.LabelColor.get_color("default"))
                self.ax_acc.spines['top'].set_visible(False)
                self.ax_acc.spines['right'].set_visible(False)
                self.ax_acc.spines['left'].set_color(qt.LabelColor.get_color("default"))
                self.ax_acc.spines['bottom'].set_color(qt.LabelColor.get_color("default"))
                self.ax_acc.patch.set_alpha(0.0)
                self.ax_acc.grid(alpha=0.3, color=qt.LabelColor.get_color("default"), linestyle="--", linewidth=1)
                self.ax_acc.plot(self.epoch, self.val_acc, color="green", linestyle="solid", label="val")
                self.ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))
                mx = min((len(self.epoch) // 10 + 1) * 10, self.config.EPOCHS)
                self.ax_acc.set_xlim([1, mx])
                self.ax_acc.legend(fontsize=fontsize, labelcolor=qt.LabelColor.get_color("default"), frameon=False)
                self.image_widget_acc.loadPixmap(fig2img(self.fig_acc, add_alpha=True))

    def check_errors(self):
        """Check if there are any errors in the parameters."""
        for tag_text, obj in self.param_objects.items():
            if obj.state == ERROR:
                # self.text_status.setText(self.tr("Please check {}.").format(tag_text))
                self.popup_error(self.tr("Please check {}.").format(tag_text))
                return False
        return True
    
    def may_continue(self, message="Continue?"):
        """Ask user for confirmation to continue."""
        mb = QtWidgets.QMessageBox
        answer = mb.question(self,
                             self.tr("Confirmation"),
                             message,
                             mb.Yes | mb.No,
                             mb.Yes)
        if answer == mb.Yes:
            return True
        elif answer == mb.No:
            return False
        else:  # answer == mb.Cancel
            return False

    def train(self):
        if not self.check_errors():
            return
        
        self.config.build_params()  # update parameters

        if self.config.log_dir is not None and os.path.exists(self.config.log_dir):
            answer = self.may_continue(self.tr("'{}' already exists. Overwrite?").format(os.path.basename(self.config.log_dir)))
            if not answer:
                self.text_status.setText(self.tr("Training was cancelled."))
                self.popup_info(self.tr("Training was cancelled."))
                return
            else:
                shutil.rmtree(self.config.log_dir, ignore_errors=True)
                os.makedirs(self.config.log_dir, exist_ok=True)

        self.disable_params()
        self.reset_state()

        config_path = os.path.join(self.dataset_dir, LOCAL_DATA_DIR_NAME, CONFIG_JSON)
        self.config.save(config_path)
        self.ai.set_config(self.config)
        self.start_time = time.time()
        self.ai.start()
        self.aiRunning.emit(True)

    def reset_state(self):
        self.epoch = []
        self.loss = []
        self.val_loss = []
        self.progress.setValue(0)
        self.text_dataset.clear()
        self.table_classes.clearContents()
        self.table_classes.setRowCount(0)
        self.image_widget_loss.clear()
        self.image_widget_acc.clear()
        self.image_widget_pie.clear()

    def augment_setting_popup(self):
        """Open data augmentation settings dialog."""
        dialog = AIAugmentDialog(self)
        result = dialog.popup(self.config)
        
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            self.update_augment_checkboxes()
            self.update_augment_availability()

    def update_augment_checkboxes(self):
        """Update augmentation checkbox states based on config values."""
        self.param_vflip.input_field.setChecked(self.config.RANDOM_VFLIP)
        self.param_hflip.input_field.setChecked(self.config.RANDOM_HFLIP)
        if self.config.RANDOM_ROTATE > 0:
            self.param_rotate.input_field.setChecked(True)
        else:
            self.param_rotate.input_field.setChecked(False)
        if self.config.RANDOM_SHIFT > 0:
            self.param_shift.input_field.setChecked(True)
        else:
            self.param_shift.input_field.setChecked(False)
        if self.config.RANDOM_SCALE > 0:
            self.param_scale.input_field.setChecked(True)
        else:
            self.param_scale.input_field.setChecked(False)
        if self.config.RANDOM_SHEAR > 0:
            self.param_shear.input_field.setChecked(True)
        else:
            self.param_shear.input_field.setChecked(False)
        if self.config.RANDOM_BLUR > 0:
            self.param_blur.input_field.setChecked(True)
        else:
            self.param_blur.input_field.setChecked(False)
        if self.config.RANDOM_NOISE > 0:
            self.param_noise.input_field.setChecked(True)
        else:
            self.param_noise.input_field.setChecked(False)
        if self.config.RANDOM_BRIGHTNESS > 0:
            self.param_brightness.input_field.setChecked(True)
        else:
            self.param_brightness.input_field.setChecked(False)
        if self.config.RANDOM_CONTRAST > 0:
            self.param_contrast.input_field.setChecked(True)
        else:
            self.param_contrast.input_field.setChecked(False)
        
    def update_augment_availability(self):
        """Update augmentation parameter availability based on selected model."""
        if self.config.TASK == TEST:
            # For MNIST models, disable all augmentations
            self.param_vflip.input_field.setEnabled(False)
            self.param_hflip.input_field.setEnabled(False)
            self.param_rotate.input_field.setEnabled(False)
            self.param_scale.input_field.setEnabled(False)
            self.param_shift.input_field.setEnabled(False)
            self.param_shear.input_field.setEnabled(False)
            self.param_blur.input_field.setEnabled(False)
            self.param_noise.input_field.setEnabled(False)
            self.param_brightness.input_field.setEnabled(False)
            self.param_contrast.input_field.setEnabled(False)
            return

    def update_logdir_list(self):
        """Update the list of log directories."""
        self.param_select_logdir.input_field.clear()
        for glob_dir in glob.glob(os.path.join(self.dataset_dir, 'aidia_data', "*")):
            if os.path.isdir(glob_dir) and os.path.exists(os.path.join(glob_dir, "model.onnx")):
                name = os.path.basename(glob_dir)
                self.param_select_logdir.input_field.addItem(name)
        if self.param_select_logdir.input_field.count() == 0:
            self.disable_utility()
        self.param_select_logdir.input_field.setCurrentText(self.prev_log_name)
        
    def predict_image(self):
        if not os.path.exists(self.target_logdir):
            self.popup_error(self.tr(
                '''The directory was not found. Please select a valid log directory and try again.'''
            ))
            return

        # Load config
        config_path = os.path.join(self.target_logdir, "config.json")
        if not os.path.exists(config_path):
            self.popup_error(self.tr("Config file was not found. Please check the log directory and try again."))
            return
        
        config = AIConfig(self.dataset_dir)
        config.load(config_path)
        config.SHOW_LABELS = self.show_labels
        config.SHOW_CONF = self.show_conf

        if config.TASK not in [SEG, DET]:
            self.popup_error(self.tr("Not implemented function."))
            return
        
        # check onnx model
        onnx_path = os.path.join(self.target_logdir, "model.onnx")
        if not os.path.exists(onnx_path):
            self.popup_error(self.tr("The ONNX model was not found. Please check the log directory and try again."))
            return
        
        # target image file
        from aidia import HOME_DIR
        opendir = HOME_DIR
        if self.prev_dir and os.path.exists(self.prev_dir):
            opendir = self.prev_dir
        
        from aidia import EXTS
        _exts = [f"*{e}" for e in EXTS]
        _exts = " ".join(_exts)
        target_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Test Image"),
            opendir,
            filter = self.tr("Image files ({});;All files (*)").format(_exts)
        )
        target_path = target_path.replace("/", os.sep)
        if not target_path:
            return
        self.prev_dir = os.path.dirname(target_path)
        self._result_dir = os.path.join(self.target_logdir, 'predict_images')
        
        # AI run
        self.text_status.setText(self.tr("Processing..."))

        self.task = config.TASK
        self.disable_params()
        self.progress.setValue(0)
        # self.reset_state()

        self.ai_pred.set_params(config, target_path, onnx_path)
        self.ai_pred.start()
        self.aiRunning.emit(True)

    def predict_images_from_directory(self):
        # load config
        config_path = os.path.join(self.target_logdir, "config.json")
        if not os.path.exists(config_path):
            self.popup_error(self.tr("Config file was not found. Please check the log directory and try again."))
            return
        
        config = AIConfig(self.dataset_dir)
        config.load(config_path)
        config.SHOW_LABELS = self.show_labels
        config.SHOW_CONF = self.show_conf

        if config.TASK not in [SEG, DET]:
            self.popup_error(self.tr("Not implemented function."))
            return
        
        # check onnx model
        onnx_path = os.path.join(self.target_logdir, "model.onnx")
        if not os.path.exists(onnx_path):
            self.popup_error(self.tr("The ONNX model was not found. Please check the log directory and try again."))
            return
        
        # target data directory
        from aidia import HOME_DIR
        opendir = HOME_DIR
        if self.prev_dir and os.path.exists(self.prev_dir):
            opendir = self.prev_dir
        target_path = str(QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Test Images Directory"),
            opendir,
            QtWidgets.QFileDialog.DontResolveSymlinks))
        target_path = target_path.replace("/", os.sep)
        if not target_path:
            return
        # self._predicted_dir = os.path.join(target_path, 'AI_results')
        self._result_dir = os.path.join(self.target_logdir, 'predict_images', utils.get_basename(target_path))
        
        if not len(os.listdir(target_path)):
            self.popup_error(self.tr("The directory is empty. Please select a directory with images and try again."))
            return

        # AI run
        self.text_status.setText(self.tr("Processing..."))

        self.task = config.TASK
        self.prev_dir = target_path
        self.disable_params()
        self.progress.setValue(0)
        # self.reset_state()

        self.ai_pred.set_params(config, target_path, onnx_path)
        self.ai_pred.start()
        self.aiRunning.emit(True)
    
    def update_pred_status(self, value):
        """Update prediction status.""" 
        self.text_status.setText(str(value))
    
    def update_pred_progress(self, value):
        self.progress.setValue(value)

    def export_model(self):
        if not os.path.exists(self.target_logdir):
            self.popup_error(self.tr(
                '''The directory was not found.'''
            ))
            return

        opendir = HOME_DIR
        target_path = str(QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Output Directory"),
            opendir,
            QtWidgets.QFileDialog.ShowDirsOnly |
            QtWidgets.QFileDialog.DontResolveSymlinks))
        if not target_path:
            return None
        target_path = target_path.replace('/', os.sep)

        cd = CopyDataDialog(self, self.target_logdir, target_path, only_model=True)
        cd.popup()

        self.popup_info(self.tr("Model exported!"))

    def export_model_to_pretrained(self):
        if not os.path.exists(self.target_logdir):
            self.popup_error(self.tr('The directory was not found.'))
            return

        from aidia import PRETRAINED_DIR
        if not os.path.exists(PRETRAINED_DIR):
            os.makedirs(PRETRAINED_DIR, exist_ok=True)

        cd = CopyDataDialog(self, self.target_logdir, PRETRAINED_DIR, only_model=True)
        cd.popup()

        self.popup_info(self.tr("Model registered!"))
        self.parent().update_ai_select()


class AITrainThread(QtCore.QThread):
    """Thread for AI training process."""

    epochLogList = QtCore.Signal(dict)
    batchLogList = QtCore.Signal(dict)
    notifyMessage = QtCore.Signal(str)
    errorMessage = QtCore.Signal(str)
    datasetInfo = QtCore.Signal(dict)

    def __init__(self, parent):
        super().__init__(parent)
        self.config = None
        # self.stop_training = False

    def set_config(self, config: AIConfig):
        self.config = config

    def quit(self):
        super().quit()
        clear_session()
        return

    def run(self):
        if self.config is None:
            self.errorMessage.emit(self.tr("Not configured. Terminated."))
            return

        model = None
        if self.config.TASK == TEST:
            model = TestModel(self.config)
        elif self.config.TASK == DET:
            model = DetectionModel(self.config)
        elif self.config.TASK == SEG:
            model = SegmentationModel(self.config)
        else:
            self.errorMessage.emit(self.tr("Model error. Terminated."))
            return
        
        # Build dataset
        self.notifyMessage.emit(self.tr("Data loading..."))
        try:
            model.build_dataset()
            model.dataset.save()
        except errors.DataLoadingError as e:
            self.errorMessage.emit(self.tr("Failed to load data.<br>Please check the settings or data."))
            aidia_logger.error(e)
            return
        except errors.DataFewError as e:
            self.errorMessage.emit(self.tr("Failed to split data because of the few data."))
            aidia_logger.error(e)
            return
        except errors.BatchsizeError as e:
            self.errorMessage.emit(self.tr("Please reduce the batch size."))
            aidia_logger.error(e)
            return
        except Exception as e:
            self.errorMessage.emit(self.tr('Unexpected error.<br>{}'.format(e)))
            aidia_logger.error(e)
            return

        if isinstance(model.dataset, Dataset):
            dataset_json_path = os.path.join(self.config.log_dir, 'dataset.json')
            _info_dict = None
            
            if os.path.exists(dataset_json_path):
                try:
                    with open(dataset_json_path, encoding="utf-8") as f:
                        _info_dict = json.load(f)
                except Exception as e:
                    aidia_logger.warning(f"Failed to load dataset.json: {e}")
                    _info_dict = None
            self.datasetInfo.emit(_info_dict)

        # Build model
        self.notifyMessage.emit(self.tr("Model building..."))
        try:
            model.build_model(mode="train")
        except torch.OutOfMemoryError as e:
            self.errorMessage.emit(self.tr("Out of memory error.<br>Please reduce the batch size or use a smaller model."))
            aidia_logger.error(e)
            return
        except Exception as e:
            self.errorMessage.emit(self.tr("Failed to build model."))
            aidia_logger.error(e)
            return

        self.notifyMessage.emit(self.tr("Preparing..."))

        # set progress callback
        self.batch = 0
        self.epoch = 0
        self.loss = 0.0

        if self.config.is_ultralytics():
            # for ultralytics models, use custom callback
            def on_train_batch_end(trainer):
                """Callback function for training batch end."""
                if trainer.tloss.nelement() > 1:
                    loss = trainer.tloss.sum().item()
                else:
                    loss = trainer.tloss.item()
                logs = {
                    "batch": self.batch + 1,
                    "loss": loss,
                }
                self.loss = loss
                self.batch += 1
                self.batchLogList.emit(logs)
            
            def on_val_end(validator):
                """Callback function for validation batch end."""
                if validator.loss.nelement() > 1:
                    val_loss = validator.loss.sum().item()
                else:
                    val_loss = validator.loss.item()
                if np.isnan(val_loss):
                    raise errors.LossGetNaNError

                val_map50 = validator.metrics.box.map50
                logs = {
                    "epoch": self.epoch + 1,
                    "loss": self.loss,
                    "val_loss": val_loss,
                    "val_map50": val_map50,
                }
                self.epoch += 1
                self.batch = 0
                self.epochLogList.emit(logs)
       
            callbacks = [on_train_batch_end, on_val_end]

        else:
            # set custom callback for other models
            def on_train_batch_end(loss):
                """Callback function for training batch end."""
                logs = {
                    "batch": self.batch + 1,
                    "loss": loss,
                }
                self.loss = loss
                self.batch += 1
                self.batchLogList.emit(logs)
            
            def on_val_end(val_loss, val_acc):
                """Callback function for validation batch end."""
                if np.isnan(val_loss) or np.isnan(val_acc):
                    raise errors.LossGetNaNError
                logs = {
                    "epoch": self.epoch + 1,
                    "loss": self.loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                self.epoch += 1
                self.batch = 0
                self.epochLogList.emit(logs)
       
            callbacks = [on_train_batch_end, on_val_end]
            
        try:
            # self.fitStarted.emit(True)
            model.train(callbacks)
        except Exception as e:
            self.errorMessage.emit(self.tr("Failed to train."))
            aidia_logger.error(e)
            return
        
        # save all training setting and used data
        if isinstance(model.dataset, Dataset):
            config_path = os.path.join(self.config.dataset_dir, LOCAL_DATA_DIR_NAME, "config.json")
            shutil.copy(config_path, self.config.log_dir)

        ### Evaluation ###
        if self.config.is_ultralytics():
            self.notifyMessage.emit(self.tr("Done"))
            return
        elif self.config.TASK == TEST:
            self.notifyMessage.emit(self.tr("Done"))
            return
        else:
            # Set inference model
            self.notifyMessage.emit(self.tr("Setting inference model..."))
            model.set_inference_model()
            save_dir = utils.get_dirpath_with_mkdir(self.config.log_dir, 'evaluation', 'test_images')

            self.notifyMessage.emit(self.tr("Generate test result images..."))
            n = model.dataset.num_test
            for i in range(n):
                image_id = model.dataset.test_ids[i]
                img_path = model.dataset.image_info[image_id]["path"]
                name = os.path.splitext(os.path.basename(img_path))[0]
                if self.config.SUBMODE:
                    dirname = utils.get_basedir(img_path)
                    subdir_path = os.path.join(save_dir, dirname)
                    if not os.path.exists(subdir_path):
                        os.mkdir(subdir_path)
                    save_path = os.path.join(save_dir, dirname, f"{name}.png")
                else:
                    save_path = os.path.join(save_dir, f"{name}.png")
                
                try:
                    result_img = model.predict_by_id(image_id)
                except FileNotFoundError as e:
                    self.notifyMessage.emit(self.tr("Error: {} was not found.").format(img_path))
                    aidia_logger.error(e)
                    return
                image.imwrite(result_img, save_path)
        
            self.notifyMessage.emit(self.tr("Evaluating..."))
            try:
                model.evaluate()
            except Exception as e:
                self.notifyMessage.emit(self.tr("Failed to evaluate."))
                aidia_logger.error(e)
                return

            self.notifyMessage.emit(self.tr("Convert model to ONNX..."))
            if not model.convert():
                self.notifyMessage.emit(self.tr("Failed to convert model to ONNX."))
                return

            self.notifyMessage.emit(self.tr("Done"))
            return



class AIPredThread(QtCore.QThread):
    """Thread for AI prediction process."""
    notifyMessage = QtCore.Signal(str)
    progressValue = QtCore.Signal(int)

    def __init__(self, parent):
        super().__init__(parent)
        self.config = None
        self.target_path = None
        self.onnx_path = None

    def set_params(self, config:AIConfig, target_path, onnx_path):
        self.config = config
        self.target_path = target_path
        self.onnx_path = onnx_path

    def quit(self):
        super().quit()
        clear_session()
        return 
    
    def run(self):
        model = None
        # single image
        if os.path.isfile(self.target_path):
            savedir = utils.get_dirpath_with_mkdir(self.config.log_dir, 'predict_images')
            name = utils.get_basename(self.target_path)
            save_path = os.path.join(savedir, f"{name}.png")
            if self.config.is_ultralytics():
                model = InferenceModel_Ultralytics(self.onnx_path, config=self.config)
            else:
                model = InferenceModel(self.onnx_path, config=self.config)
            model.run(self.target_path, save_path)
            self.notifyMessage.emit(self.tr("Prediction results saved."))
            return
        
        # directory
        else:
            savedir = utils.get_dirpath_with_mkdir(self.config.log_dir, 'predict_images', utils.get_basename(self.target_path))
            n = len(os.listdir(self.target_path))

            if self.config.is_ultralytics():
                model = InferenceModel_Ultralytics(self.onnx_path, config=self.config)
            else:
                model = InferenceModel(self.onnx_path, config=self.config)

            # iterate all files
            for i, file_path in enumerate(glob.glob(os.path.join(self.target_path, "*"))):
                if utils.extract_ext(file_path) == ".json":
                    continue

                self.notifyMessage.emit(f"{i} / {n}: {os.path.basename(file_path)}")
                name = utils.get_basename(file_path)
                save_path = os.path.join(savedir, f"{name}.png")

                model.run(file_path, save_path)
                
                self.progressValue.emit(int(i / n * 100))

            self.progressValue.emit(0)
            self.notifyMessage.emit(self.tr("Prediction results saved."))
            return
    
