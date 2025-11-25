import os
import shutil
import time
import random
import glob
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
from aidia.ai.det import DetectionModel
from aidia.ai.seg import SegmentationModel
from aidia.ai.ai_utils import InferenceModel, InferenceModel_Ultralytics
from aidia.widgets import ImageWidget
from aidia.widgets.ai_augment_dialog import AIAugmentDialog
from aidia.widgets.ai_label_replace_dialog import AILabelReplaceDialog
from aidia.widgets.copy_data_dialog import CopyDataDialog

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


class LabelStyle:
    """Styles for QLabel."""
    DEFAULT = "QLabel{ color: white; }" if qt.is_dark_mode() else "QLabel{ color: black; }"
    ERROR = "QLabel{ color: red; }"
    DISABLED = "QLabel{ color: gray; }"


class ParamComponent(object):
    """Base class for AI parameter components."""

    def __init__(self, type, tag, tips, validate_func=None, items=None):
        super().__init__()

        minimum_width = 250

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
    # aiTerminated = QtCore.Signal()

    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowFlags(Qt.Window
                            | Qt.CustomizeWindowHint
                            | Qt.WindowTitleHint
                            | Qt.WindowCloseButtonHint
                            | Qt.WindowMaximizeButtonHint
                            )
        self.setWindowTitle(self.tr("AI Training"))

        self.setMinimumSize(QtCore.QSize(1200, 900))

        self.dataset_dir = None
        self.target_logdir = None
        self.prev_dir = None
        self.start_time = 0
        self.start_epoch_time = 0
        self.end_epoch_time = 0
        self.epoch = []
        self.loss = []
        self.val_loss = []
        self.train_steps = 0
        self.val_steps = 0

        # inference display settings
        self.show_labels = True
        self.show_conf = False

        self.fig_loss, self.ax_loss = plt.subplots(figsize=(12, 6))
        self.fig_loss.patch.set_alpha(0.0)
        self.ax_loss.axis("off")
        
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

        self._dataset_layout = QtWidgets.QVBoxLayout()
        self._dataset_widget = QtWidgets.QWidget()

        title_dataset = qt.head_text(self.tr("Dataset Information"))
        title_dataset.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        title_dataset.setMaximumHeight(30)
        self._dataset_layout.addWidget(title_dataset)

        self._augment_layout = QtWidgets.QVBoxLayout()
        self._augment_widget = QtWidgets.QWidget()

        title_augment = qt.head_text(self.tr("Data Augmentation"))
        title_augment.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        title_augment.setMaximumHeight(30)
        self._augment_layout.addWidget(title_augment)

        # utility layout
        self._utility_layout = QtWidgets.QVBoxLayout()
        self._utility_widget = QtWidgets.QWidget()

        title_utility = qt.head_text(self.tr("Utilities"))
        title_utility.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        title_utility.setMaximumHeight(30)
        self._utility_layout.addWidget(title_utility)

        self.tag_logdir = QtWidgets.QLabel(self.tr("Select Experiment Directory"))
        self.tag_logdir.setMaximumHeight(16)
        self.tag_logdir.setToolTip(self.tr("Select the experiment directory."))
        self._utility_layout.addWidget(self.tag_logdir, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)

        self.input_logdir = QtWidgets.QComboBox()
        def _validate(idx):
            idx = int(idx)
            if idx < 0:
                return
            name = self.input_logdir.itemText(idx)
            self.target_logdir = os.path.join(self.dataset_dir, LOCAL_DATA_DIR_NAME, name)
        self.input_logdir.currentIndexChanged.connect(_validate)
        self._utility_layout.addWidget(self.input_logdir)

        self.button_open_logdir = QtWidgets.QPushButton(self.tr("Open Log Directory"))
        self.button_open_logdir.setToolTip(self.tr("Open the selected log directory."))
        self.button_open_logdir.setAutoDefault(False)
        self.button_open_logdir.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.target_logdir)))
        self._utility_layout.addWidget(self.button_open_logdir)

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

        # export model button
        self.button_export_model = QtWidgets.QPushButton(self.tr("Export Model"))
        self.button_export_model.setToolTip(self.tr("Export the model data."))
        self.button_export_model.clicked.connect(self.export_model)
        self._utility_layout.addWidget(self.button_export_model)

        # export model to pretrained button
        self.button_export_model_to_pretrained = QtWidgets.QPushButton(self.tr("Export Model to Pretrained"))
        self.button_export_model_to_pretrained.setToolTip(self.tr("Export the model data to pretrained directory."))
        self.button_export_model_to_pretrained.clicked.connect(self.export_model_to_pretrained)
        self._utility_layout.addWidget(self.button_export_model_to_pretrained)

        # connect AI prediction thread
        self.ai_pred = AIPredThread(self)
        self.ai_pred.notifyMessage.connect(self.update_pred_status)
        self.ai_pred.progressValue.connect(self.update_pred_progress)
        self.ai_pred.finished.connect(self.ai_pred_finished)

        # directory information
        # self.tag_directory = QtWidgets.QLabel()
        # self.tag_directory.setMaximumHeight(100)
        # self._layout.addWidget(self.tag_directory, 0, 1, 1, 4)
        # self.left_row += 1
        # self.right_row_count += 1

        # task selection
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

        # model selection
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

        # name
        def _validate(text):
            text = text.strip().replace(" ", "_")
            self.config.NAME = text
        self.param_name = ParamComponent(
            type="text",
            tag=self.tr("Experiment Directory Name"),
            tips=self.tr("Set the name of the experiment directory."),
            validate_func=_validate,
        )
        self.add_param_component(self.param_name)

        # dataset idx
        def _validate(idx):
            self.config.DATASET_NUM = int(idx+1)
        self.param_dataset = ParamComponent(
            type="combo",
            tag=self.tr("Dataset"),
            tips=self.tr("""Select the dataset pattern.
Aidia splits the data into a 4:1 ratio (train:test) depend on the selected pattern.
You can use this function for 5-fold cross-validation."""),
            validate_func=_validate,
            items=[
                self.tr("Pattern 1"),
                self.tr("Pattern 2"),
                self.tr("Pattern 3"),
                self.tr("Pattern 4"),
                self.tr("Pattern 5"),
            ]
        )
        self.add_param_component(self.param_dataset)

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
                "768", "832", "896", "960", "1024", "1088",
                "1152", "1216", "1280"
            ]
        )
        self.add_param_component(self.param_input_size)

        # epochs
        def _validate(text):
            if text.isdigit() and 0 < int(text):
                self.set_ok(self.param_epochs)
                self.config.EPOCHS = int(text)
            else:
                self.set_error(self.param_epochs)
        self.param_epochs = ParamComponent(
            type="text",
            tag=self.tr("Epochs"),
            tips=self.tr("""Set the epochs.
If you set 100, all data are trained 100 times."""),
            validate_func=_validate,
        )
        self.add_param_component(self.param_epochs)

        # batch size
        def _validate(text):
            if text.isdigit() and 0 < int(text) <= 256:
                self.set_ok(self.param_batchsize)
                self.config.BATCH_SIZE = int(text)
            else:
                self.set_error(self.param_batchsize)
        self.param_batchsize = ParamComponent(
            type="text",
            tag=self.tr("Batch Size"),
            tips=self.tr("""Set the batch size.
If you set 8, 8 samples are trained per step."""),
            validate_func=_validate,
        )
        self.add_param_component(self.param_batchsize)

        # learning rate
        def _validate(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 1.0:
                self.set_ok(self.param_lr)
                self.config.LEARNING_RATE = float(text)
            else:
                self.set_error(self.param_lr)
        self.param_lr = ParamComponent(
            type="text",
            tag=self.tr("Learning Rate"),
            tips=self.tr("Set the initial learning rate."),
            validate_func=_validate,
        )
        self.add_param_component(self.param_lr)

        # label definition
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
            tag=self.tr("Label Definition"),
            tips=self.tr("""Set target labels.
The labels are separated with line breaks."""),
            validate_func=_validate,
        )
        self.add_param_component(self.param_labels, right=True, custom_size=(4, 1))

        # save best only
        # self.tag_is_savebest = QtWidgets.QLabel(self.tr("Save Only the Best Weights"))
        # self.tag_is_savebest.setToolTip(self.tr("""Enable saving only the weights achived the minimum validation loss."""))
        # self.input_is_savebest = QtWidgets.QCheckBox()
        # def _validate(state): # check:2, empty:0
        #     if state == 2:
        #         self.config.SAVE_BEST = True
        #     else:
        #         self.config.SAVE_BEST = False
        # self.input_is_savebest.stateChanged.connect(_validate)
        # self._add_basic_params(self.tag_is_savebest, self.input_is_savebest, right=True, reverse=True)

        # early stopping
        # def _validate(state): # check:2, empty:0
        #     if state == 2:
        #         self.config.EARLY_STOPPING = True
        #     else:
        #         self.config.EARLY_STOPPING = False
        # self.param_is_earlystop = AIParamComponent(
        #     type="checkbox",
        #     tag=self.tr("Early Stopping"),
        #     tips=self.tr("""(BETA) Enable Early Stopping."""),
        #     validate_func=_validate
        # )
        # self.add_param_component(self.param_is_earlystop, right=True, reverse=True)

        # use multiple gpu
        # self.tag_is_multi = QtWidgets.QLabel(self.tr("Use Multiple GPUs"))
        # self.tag_is_multi.setToolTip(self.tr("""Enable parallel calculation with multiple GPUs."""))
        # self.input_is_multi = QtWidgets.QCheckBox()
        # def _validate(state): # check:2, empty:0
        #     if state == 2:
        #         self.config.USE_MULTI_GPUS = True
        #     else:
        #         self.config.USE_MULTI_GPUS = False
        # self.input_is_multi.stateChanged.connect(_validate)
        # self._add_basic_params(self.tag_is_multi, self.input_is_multi, right=True, reverse=True)

        # label replacement button
        button_label_replace = QtWidgets.QPushButton(self.tr("Label Replacement"))
        button_label_replace.setToolTip(self.tr("Open label replacement dialog."))
        button_label_replace.setAutoDefault(False)
        self._layout.addWidget(button_label_replace, self.right_row_count, 3, 1, 2)
        button_label_replace.clicked.connect(self.label_replace_popup)
        self.right_row_count += 1
        self.button_label_replace = button_label_replace

        # train target select
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

        ### add augment params ###
        # vertical flip
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

        # horizontal flip
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

        # rotation
        def _validate_rotate(state): # check:2, empty:0
            if state == 2:
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

        # scale
        def _validate_scale(state): # check:2, empty:0
            if state == 2:
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

        # shift
        def _validate_shift(state): # check:2, empty:0
            if state == 2:
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

        # shear
        def _validate_shear(state): # check:2, empty:0
            if state == 2:
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

        # brightness
        def _validate_brightness(state): # check:2, empty:0
            if state == 2:
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

        # contrast
        def _validate_contrast(state): # check:2, empty:0
            if state == 2:
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

        # blur
        def _validate_blur(state): # check:2, empty:0
            if state == 2:
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

        # noise
        def _validate_noise(state): # check:2, empty:0
            if state == 2:
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

        # advanced settings
        button_advanced = QtWidgets.QPushButton(self.tr("Advanced Settings"))
        button_advanced.setToolTip(self.tr("Open advanced data augmentation settings."))
        button_advanced.setAutoDefault(False)
        button_advanced.clicked.connect(self.augment_setting_popup)
        self._augment_layout.addWidget(button_advanced)
        self.button_advanced = button_advanced

        # update lowest row
        row_count = max(self.left_row_count, self.right_row_count)

        # train button
        self.button_train = QtWidgets.QPushButton(self.tr("Train"))
        self.button_train.setMinimumHeight(64)
        self.button_train.setStyleSheet("font-size: 20px;")
        self.button_train.clicked.connect(self.train)
        self._layout.addWidget(self.button_train, row_count, 1, 1, 4)
        row_count += 1

        # figure area
        self.image_widget_loss = ImageWidget(self)
        self.image_widget_loss.setMinimumHeight(800)
        self._layout.addWidget(self.image_widget_loss, row_count, 1, 1, 4)
        row_count += 1

        # progress bar
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setStyleSheet("""QProgressBar {
    border: 2px solid grey;
    border-radius: 5px;
    text-align: center; }
QProgressBar::chunk {
    background-color: #05B8CC;
    width: 20px; }""")
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self._layout.addWidget(self.progress, row_count, 1, 1, 4)
        row_count += 1

        # status
        self.text_status = QtWidgets.QLabel()
        self.text_status.setMaximumHeight(32)
        self._layout.addWidget(self.text_status, row_count, 1, 1, 4)

        # stop button
        # self.button_stop = QtWidgets.QPushButton(self.tr("Terminate"))
        # self.button_stop.clicked.connect(self.stop_training)
        # self._layout.addWidget(self.button_stop, row, 4, 1, 1, Qt.AlignRight)
        # row += 1

        # dataset information
        self.text_dataset = QtWidgets.QLabel()
        self.text_dataset.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._dataset_layout.addWidget(self.text_dataset)

        self.image_widget_pie = ImageWidget(self)
        self._dataset_layout.addWidget(self.image_widget_pie)

        ### set layouts ###
        self._dataset_widget.setLayout(self._dataset_layout)
        self._layout.addWidget(self._dataset_widget, 0, 0, row_count + 1, 1)

        self._augment_widget.setLayout(self._augment_layout)
        self._layout.addWidget(self._augment_widget, 0, 5, row_count - 2, 1)

        self._utility_widget.setLayout(self._utility_layout)
        self._layout.addWidget(self._utility_widget, row_count - 2, 5, 3, 1)
        
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

        # self.aiTerminated.connect(self.ai.quit)

        self.text_status.setText(self.tr("Ready"))

    # def stop_training(self):
    #     self.aiTerminated.emit()
    #     self.button_stop.setEnabled(False)

    def popup(self, dataset_dir, is_submode=False, data_labels=None):
        """Popup train window and set config parameters to input fields."""
        self.dataset_dir = dataset_dir
        self.setWindowTitle(self.tr("AI Training - {}").format(dataset_dir))

        # create data directory
        data_dirpath = utils.get_dirpath_with_mkdir(dataset_dir, LOCAL_DATA_DIR_NAME)

        # load config parameters
        self.config = AIConfig(dataset_dir)
        config_path = os.path.join(data_dirpath, CONFIG_JSON)
        if os.path.exists(config_path):
            try:
                self.config.load(config_path)
            except Exception as e:
                aidia_logger.error(e, exc_info=True)

        self.config.SUBMODE = is_submode
        if is_submode:
            self.label_current_mode.setText(self.tr('Search data from <span style="color: red;"><b>PARENT</b></span> directory'))
        else:
            self.label_current_mode.setText(self.tr('Search data from <span style="color: green;"><b>CURRENT</b></span> directory'))

        # basic params
        self.param_task.input_field.setCurrentIndex(TASK_LIST.index(self.config.TASK))
        self.enable_params_by_task(self.config.TASK)
        self.param_model.input_field.setCurrentText(self.config.MODEL)
        self.param_name.input_field.setText(self.config.NAME)
        self.param_dataset.input_field.setCurrentIndex(int(self.config.DATASET_NUM) - 1)
        if self.config.INPUT_SIZE in [int(self.param_input_size.input_field.itemText(i)) for i in range(self.param_input_size.input_field.count())]:
            self.param_input_size.input_field.setCurrentText(str(self.config.INPUT_SIZE))
        else:
            self.param_input_size.input_field.setCurrentIndex(0)  # Reset to first item if not found
        self.param_epochs.input_field.setText(str(self.config.EPOCHS))
        self.param_batchsize.input_field.setText(str(self.config.BATCH_SIZE))
        self.param_lr.input_field.setText(str(self.config.LEARNING_RATE))

        if len(self.config.LABELS) > 0:
            self.param_labels.input_field.setText("\n".join(self.config.LABELS))
        else:
            self.param_labels.input_field.setText("\n".join(data_labels))
        # if self.config.gpu_num < 2:
        #     self.input_is_multi.setEnabled(False)
        # self.input_is_multi.setChecked(self.config.USE_MULTI_GPUS)
        # self.input_is_savebest.setChecked(self.config.SAVE_BEST)
        # self.param_is_earlystop.input_field.setChecked(self.config.EARLY_STOPPING)
        if not self.config.SUBMODE:
            self.param_is_dir_split.input_field.setEnabled(False)
        self.param_is_dir_split.input_field.setChecked(self.config.DIR_SPLIT)

        # prediction display settings
        self.checkbox_show_labels.setChecked(self.config.SHOW_LABELS)
        if not self.config.SHOW_LABELS:
            self.checkbox_show_conf.setChecked(False)
            self.checkbox_show_conf.setEnabled(False)
            self.show_labels = False
            self.show_conf = False
        else:
            self.checkbox_show_conf.setChecked(self.config.SHOW_CONF)

        # augment params
        self.update_augment_checkboxes()
        # Update augmentation parameter availability after loading config
        self.update_augment_availability()

        self.update_logdir_list()

        self.exec_()
        if os.path.exists(os.path.join(dataset_dir, LOCAL_DATA_DIR_NAME)):
            self.config.save(config_path)
    
    ### Callbacks ###
    def ai_finished(self):
        """Call back function when AI thread finished."""
        self.enable_params_by_task(self.config.TASK)

        # clear cuda cache
        clear_session()
        
        # raise error handle
        config_path = os.path.join(self.config.log_dir, CONFIG_JSON)
        dataset_path = os.path.join(self.config.log_dir, DATASET_JSON)
        if not os.path.exists(config_path) or not os.path.exists(dataset_path):
            # self.text_status.setText(self.tr("Training was failed."))
            self.reset_state()
            self.aiRunning.emit(False)
            self.text_status.setText(self.tr("Terminated training."))
            return
        
        # display elapsed time
        now = time.time()
        etime = now - self.start_time
        h = int(etime // 3600)
        m = int(etime // 60 % 60)
        s = int(etime % 60)
        self.text_status.setText(self.tr("Done -- Elapsed time: {}h {}m {}s").format(h, m, s))

        # save metrics
        df_dic = {
            "epoch": self.epoch,
            "loss": self.loss,
            "val_loss": self.val_loss
        }
        utils.save_dict_to_excel(df_dic, os.path.join(self.config.log_dir, "loss.xlsx"))

        # save figure
        self.fig_loss.savefig(os.path.join(self.config.log_dir, "loss.png"))

        # convet YOLO model to ONNX
        if self.config.is_ultralytics():
            from aidia.ai.ai_utils import write_onnx_u
            try:
                model_path = os.path.join(self.config.log_dir, self.config.MODEL, "weights", "best.pt")
                onnx_path = write_onnx_u(model_path)
            except Exception as e:
                aidia_logger.error(e, exc_info=True)
                self.text_status.setText(self.tr("Failed to convert to ONNX model."))
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

        self.button_label_replace.setEnabled(True)
        self.button_advanced.setEnabled(True)
        # self.button_stop.setEnabled(False)

        self.switch_utility()

    def switch_utility(self):
        """Switch enabled state of utility components."""
        if self.input_logdir.count() == 0:
            self.disable_utility()
        else:
            self.enable_utility()

    def switch_enabled(self, targets: list[ParamComponent], enabled:bool):
        for obj in targets:
            if enabled:
                obj.tag.setStyleSheet(LabelStyle.DEFAULT)
            else:
                obj.tag.setStyleSheet(LabelStyle.DISABLED)
            obj.input_field.setEnabled(enabled)
        # if enabled and self.config.gpu_num < 2:
        #     self.tag_is_multi.setStyleSheet(LabelStyle.DISABLED)
        #     self.input_is_multi.setEnabled(False)
        if enabled and not self.config.SUBMODE:
            self.param_is_dir_split.tag.setStyleSheet(LabelStyle.DISABLED)
            self.param_is_dir_split.input_field.setEnabled(False)

    def switch_global_params(self):
        # if self.config.gpu_num < 2:
        #     self.tag_is_multi.setStyleSheet(LabelStyle.DISABLED)
        #     self.input_is_multi.setEnabled(False)
        # else:
        #     self.tag_is_multi.setStyleSheet(LabelStyle.DEFAULT)
        #     self.input_is_multi.setEnabled(True)
        if not self.config.SUBMODE or self.config.TASK in [TEST]:
            self.param_is_dir_split.tag.setStyleSheet(LabelStyle.DISABLED)
            self.param_is_dir_split.input_field.setEnabled(False)
        else:
            self.param_is_dir_split.tag.setStyleSheet(LabelStyle.DEFAULT)
            self.param_is_dir_split.input_field.setEnabled(True)
    
    def _enable_params(self):
        """Enable all components."""
        for obj in self.param_objects.values():
            obj.input_field.setEnabled(True)
            obj.tag.setStyleSheet(LabelStyle.DEFAULT)
            if obj.state == ERROR:
                obj.tag.setStyleSheet(LabelStyle.ERROR)

    def enable_utility(self):
        """Enable utility components."""
        self.input_logdir.setEnabled(True)
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
        self.button_export_model.setEnabled(True)
        self.button_export_model_to_pretrained.setEnabled(True)
    
    def disable_utility(self):
        """Disable utility components."""
        self.input_logdir.setEnabled(False)
        self.button_open_logdir.setEnabled(False)
        self.checkbox_show_labels.setEnabled(False)
        self.checkbox_show_conf.setEnabled(False)
        self.button_pred.setEnabled(False)
        self.button_pred_dir.setEnabled(False)
        self.button_export_model.setEnabled(False)
        self.button_export_model_to_pretrained.setEnabled(False)
    
    def disable_params(self):
        """Disable all components."""
        for obj in self.param_objects.values():
            obj.input_field.setEnabled(False)
            obj.tag.setStyleSheet(LabelStyle.DISABLED)
        self.button_label_replace.setEnabled(False)
        self.button_advanced.setEnabled(False)
        self.button_train.setEnabled(False)
        
        # Disable utility components
        self.input_logdir.setEnabled(False)
        self.button_open_logdir.setEnabled(False)
        self.checkbox_show_labels.setEnabled(False)
        self.checkbox_show_conf.setEnabled(False)
        self.button_pred.setEnabled(False)
        self.button_pred_dir.setEnabled(False)
        self.button_export_model.setEnabled(False)
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
    
    def label_replace_popup(self):
        """Open label replacement dialog."""
        dialog = AILabelReplaceDialog(self)
        result = dialog.popup(self.config.REPLACE_DICT)
        
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            self.config.REPLACE_DICT = dialog.replace_dict
            if self.config.REPLACE_DICT:
                self.text_status.setText(
                    self.tr("Label replacement dictionary updated with {} rules.").format(
                        len(self.config.REPLACE_DICT)
                    )
                )
            else:
                self.text_status.setText(self.tr("Label replacement dictionary cleared."))
        else:
            self.text_status.setText(self.tr("Label replacement cancelled."))
  
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
        obj.tag.setStyleSheet(LabelStyle.ERROR)
        obj.state = ERROR

    def set_ok(self, obj: ParamComponent):
        """Set ok state to the parameter component."""
        obj.tag.setStyleSheet(LabelStyle.DEFAULT)
        obj.state = CLEAR

    def update_figure(self):
        """Update the figure for loss."""
        

    def update_dataset(self, value):
        """Update dataset information."""
        dataset_num = value["dataset_num"]
        num_images = value["num_images"]
        num_shapes = value["num_shapes"]
        num_classes = value["num_classes"]
        num_train = value["num_train"]
        num_val = value["num_val"]
        num_test = value["num_test"]
        class_names = value["class_names"]
        num_per_class = value["num_per_class"]
        train_per_class = value["train_per_class"]
        val_per_class = value["val_per_class"]
        test_per_class = value["test_per_class"]
        self.train_steps = value["train_steps"]
        self.val_steps = value["val_steps"]

        labels_info = [self.tr("[*] labels (all|train|val|test)")]
        for i in range(num_classes):
            name = class_names[i]
            n = num_per_class[i]
            n_train = train_per_class[i]
            n_val = val_per_class[i]
            n_test = test_per_class[i]
            labels_info.append(f"[{i}] {name} ({n} | {n_train} | {n_val} | {n_test})")
        labels_info = "\n".join(labels_info)

        text = []
        text.append(self.tr("Dataset Number: {}").format(dataset_num))
        text.append(self.tr("Number of Data: {}").format(num_images))
        text.append(self.tr("Number of Train: {}").format(num_train))
        text.append(self.tr("Number of Validation: {}").format(num_val))
        text.append(self.tr("Number of Test: {}").format(num_test))
        if self.config.SUBMODE and self.config.DIR_SPLIT:
            text.append(self.tr("Number of Train Directories: {}").format(value["num_train_subdir"]))
            text.append(self.tr("Number of Validation Directories: {}").format(value["num_val_subdir"]))
            text.append(self.tr("Number of Test Directories: {}").format(value["num_test_subdir"]))
        text.append(self.tr("Train Steps: {}").format(self.train_steps))
        text.append(self.tr("Validation Steps: {}").format(self.val_steps))
        text.append(self.tr("Number of Shapes: {}").format(num_shapes))
        text.append(self.tr("Class Information:\n{}").format(labels_info))
        text = "\n".join(text)
        self.text_dataset.setText(text)

        # update label distribution
        self.ax_pie.clear()
        self.ax_pie.set_title('Label Distribusion', fontsize=20, color=qt.get_default_color())
        self.ax_pie.pie(num_per_class,
                    labels=class_names,
                    #  autopct="%1.1f%%",
                    wedgeprops={'linewidth': 1, 'edgecolor': qt.get_default_color()},
                    textprops={'color': qt.get_default_color(),
                               'fontsize': 16})
        self.image_widget_pie.loadPixmap(fig2img(self.fig_pie, add_alpha=True))

    def update_status(self, value):
        """Update status text."""
        self.text_status.setText(str(value))

    def popup_error(self, text):
        """Popup error message."""
        self.parent().error_message(text)

    def update_batch(self, value):
        """Update batch status."""
        epoch = len(self.epoch) + 1
        batch = value.get("batch")
        loss = value.get("loss")

        text = f"epoch: {epoch:>4}/{self.config.EPOCHS} "
        if batch is not None:
            if batch > self.train_steps:
                batch = self.train_steps
            text += f"batch: {batch:>6} / {self.train_steps} "
        if loss is not None:
            text += f"loss: {loss:>8.4f} "
        if len(self.val_loss):
            text += f"val_loss: {self.val_loss[-1]:>8.4f}"

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
        progress_value = int(epoch / self.config.EPOCHS * 100)

        if epoch is not None:
            self.epoch.append(epoch)
            self.progress.setValue(progress_value)

        if loss is not None:
            self.loss.append(loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)

        if self.start_epoch_time == 0:
            self.start_epoch_time = time.time()
        elif self.end_epoch_time == 0:
            self.end_epoch_time = time.time()

        # update figure
        self.ax_loss.clear()
        self.ax_loss.set_xlabel("Epoch", fontsize=16, color=qt.get_default_color())
        self.ax_loss.set_ylabel("Loss", fontsize=16, color=qt.get_default_color())
        self.ax_loss.tick_params(axis='both', labelsize=14, colors=qt.get_default_color())
        self.ax_loss.spines['top'].set_visible(False)
        self.ax_loss.spines['right'].set_visible(False)
        self.ax_loss.spines['left'].set_color(qt.get_default_color())
        self.ax_loss.spines['bottom'].set_color(qt.get_default_color())
        self.ax_loss.patch.set_alpha(0.0)
        self.ax_loss.grid(alpha=0.3, color=qt.get_default_color(), linestyle="--", linewidth=1)
        if len(self.epoch):
            if len(self.loss):
                self.ax_loss.plot(self.epoch, self.loss, color="red", linestyle="solid", label="train")
            if len(self.val_loss):
                self.ax_loss.plot(self.epoch, self.val_loss, color="green", linestyle="solid", label="val")
            self.ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
            mx = min((len(self.epoch) // 10 + 1) * 10, self.config.EPOCHS)
            self.ax_loss.set_xlim([1, mx])
            if len(self.loss) > 1 and len(self.val_loss) > 1:
                max_loss = max(self.loss[1:] + self.val_loss[1:])
                self.ax_loss.set_ylim([0, max_loss * 1.1])
            self.ax_loss.legend(fontsize=16, labelcolor=qt.get_default_color(), frameon=False)
            self.image_widget_loss.loadPixmap(fig2img(self.fig_loss, add_alpha=True))

    def check_errors(self):
        """Check if there are any errors in the parameters."""
        for tag_text, obj in self.param_objects.items():
            if obj.state == ERROR:
                self.text_status.setText(self.tr("Please check {}.").format(tag_text))
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
        self.image_widget_loss.clear()
        self.image_widget_pie.clear()

    def augment_setting_popup(self):
        """Open data augmentation settings dialog."""
        # Check if Ultralytics model and warn user about limitations
        is_ultralytics = self.config.is_ultralytics()
        
        dialog = AIAugmentDialog(self)
        
        # If Ultralytics model, disable specific parameters in the dialog
        if is_ultralytics:
            # Set the values to 0 for Ultralytics models before opening dialog
            original_contrast = self.config.RANDOM_CONTRAST
            original_blur = self.config.RANDOM_BLUR
            original_noise = self.config.RANDOM_NOISE
            
            self.config.RANDOM_CONTRAST = 0.0
            self.config.RANDOM_BLUR = 0.0
            self.config.RANDOM_NOISE = 0.0
        
        result = dialog.popup(self.config)
        
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            # For Ultralytics models, ensure disabled parameters remain 0
            if is_ultralytics:
                self.config.RANDOM_CONTRAST = 0.0
                self.config.RANDOM_BLUR = 0.0
                self.config.RANDOM_NOISE = 0.0
            
            # Update checkbox states based on new config values
            self.update_augment_checkboxes()
            # Update parameter availability
            self.update_augment_availability()
            self.text_status.setText(self.tr("Augmentation settings updated."))
        else:
            # Restore original values if user cancelled and it was Ultralytics
            if is_ultralytics:
                self.config.RANDOM_CONTRAST = original_contrast
                self.config.RANDOM_BLUR = original_blur
                self.config.RANDOM_NOISE = original_noise
            self.text_status.setText(self.tr("Augmentation settings unchanged."))

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

        is_ultralytics = self.config.is_ultralytics()
            
        # For Ultralytics models, disable contrast, blur, and noise
        if is_ultralytics:
            self.param_contrast.input_field.setEnabled(False)
            self.param_contrast.input_field.setChecked(False)
            self.param_contrast.tag.setStyleSheet(LabelStyle.DISABLED)
            self.config.RANDOM_CONTRAST = 0.0
        else:
            self.param_contrast.input_field.setEnabled(True)
            self.param_contrast.tag.setStyleSheet(LabelStyle.DEFAULT)
    
        if is_ultralytics:
            self.param_blur.input_field.setEnabled(False)
            self.param_blur.input_field.setChecked(False)
            self.param_blur.tag.setStyleSheet(LabelStyle.DISABLED)
            self.config.RANDOM_BLUR = 0.0
        else:
            self.param_blur.input_field.setEnabled(True)
            self.param_blur.tag.setStyleSheet(LabelStyle.DEFAULT)
    
        if is_ultralytics:
            self.param_noise.input_field.setEnabled(False)
            self.param_noise.input_field.setChecked(False)
            self.param_noise.tag.setStyleSheet(LabelStyle.DISABLED)
            self.config.RANDOM_NOISE = 0.0
        else:
            self.param_noise.input_field.setEnabled(True)
            self.param_noise.tag.setStyleSheet(LabelStyle.DEFAULT)
        
    def update_logdir_list(self):
        """Update the list of log directories."""
        self.input_logdir.clear()
        for glob_dir in glob.glob(os.path.join(self.dataset_dir, LOCAL_DATA_DIR_NAME, "*")):
            if os.path.isdir(glob_dir) and os.path.exists(os.path.join(glob_dir, "model.onnx")):
                name = os.path.basename(glob_dir)
                self.input_logdir.addItem(name)
        if self.input_logdir.count() == 0:
            self.disable_utility()
        
    def predict_image(self):
        if not os.path.exists(self.target_logdir):
            self.parent().error_message(self.tr(
                '''The directory was not found.'''
            ))
            return

        # load config
        config_path = os.path.join(self.target_logdir, "config.json")
        if not os.path.exists(config_path):
            self.text_status.setText(self.tr("Config file was not found."))
            return
        
        config = AIConfig(self.dataset_dir)
        config.load(config_path)
        config.SHOW_LABELS = self.show_labels
        config.SHOW_CONF = self.show_conf

        if config.TASK not in [SEG, DET]:
            self.text_status.setText(self.tr("Not implemented function."))
            return
        
        # check onnx model
        onnx_path = os.path.join(self.target_logdir, "model.onnx")
        if not os.path.exists(onnx_path):
            self.text_status.setText(self.tr("The ONNX model was not found."))
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
            self.text_status.setText(self.tr("Config file was not found."))
            return
        
        config = AIConfig(self.dataset_dir)
        config.load(config_path)
        config.SHOW_LABELS = self.show_labels
        config.SHOW_CONF = self.show_conf

        if config.TASK not in [SEG, DET]:
            self.text_status.setText(self.tr("Not implemented function."))
            return
        
        # check onnx model
        onnx_path = os.path.join(self.target_logdir, "model.onnx")
        if not os.path.exists(onnx_path):
            self.text_status.setText(self.tr("The ONNX model was not found."))
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
            self.text_status.setText(self.tr("The Directory is empty."))
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
            self.parent().error_message(self.tr(
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

        self.text_status.setText(self.tr("Export data to {}").format(target_path))

    def export_model_to_pretrained(self):
        if not os.path.exists(self.target_logdir):
            self.parent().error_message(self.tr(
                '''The directory was not found.'''
            ))
            return

        from aidia import PRETRAINED_DIR
        if not os.path.exists(PRETRAINED_DIR):
            os.makedirs(PRETRAINED_DIR, exist_ok=True)

        cd = CopyDataDialog(self, self.target_logdir, PRETRAINED_DIR, only_model=True)
        cd.popup()

        self.text_status.setText(self.tr("Export data to {}").format(PRETRAINED_DIR))
        self.parent().update_ai_select()


class AITrainThread(QtCore.QThread):
    """Thread for AI training process."""

    # fitStarted = QtCore.Signal(bool)
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
        
        self.notifyMessage.emit(self.tr("Data loading..."))
        try:
            model.build_dataset()
        except errors.DataLoadingError as e:
            self.errorMessage.emit(self.tr("Failed to load data.<br>Please check the settings or data."))
            aidia_logger.error(e, exc_info=True)
            return
        except errors.DataFewError as e:
            self.errorMessage.emit(self.tr("Failed to split data because of the few data."))
            aidia_logger.error(e, exc_info=True)
            return
        except errors.BatchsizeError as e:
            self.errorMessage.emit(self.tr("Please reduce the batch size."))
            aidia_logger.error(e, exc_info=True)
            return
        except Exception as e:
            self.errorMessage.emit(self.tr('Unexpected error.<br>{}'.format(e)))
            aidia_logger.error(e, exc_info=True)
            return

        if isinstance(model.dataset, Dataset):
            _info_dict = {
                "dataset_num": model.dataset.dataset_num,
                "num_images": model.dataset.num_images,
                "num_shapes": model.dataset.num_shapes,
                "num_classes": model.dataset.num_classes,
                "num_per_class": model.dataset.num_per_class,
                "num_train": model.dataset.num_train,
                "num_val": model.dataset.num_val,
                "num_test": model.dataset.num_test,
                "class_ids": model.dataset.class_ids,
                "class_names": model.dataset.class_names,
                "train_per_class": model.dataset.train_per_class,
                "val_per_class": model.dataset.val_per_class,
                "test_per_class": model.dataset.test_per_class,
                "train_steps": model.dataset.train_steps,
                "val_steps": model.dataset.val_steps
            }
            if self.config.SUBMODE and self.config.DIR_SPLIT:
                _info_dict["num_subdir"] = model.dataset.num_subdir
                _info_dict["num_train_subdir"] = model.dataset.num_train_subdir
                _info_dict["num_val_subdir"] = model.dataset.num_val_subdir
                _info_dict["num_test_subdir"] = model.dataset.num_test_subdir
            self.datasetInfo.emit(_info_dict)

        self.notifyMessage.emit(self.tr("Model building..."))
        # if self.config.gpu_num > 1 and self.config.USE_MULTI_GPUS: # apply multiple GPU support
        #     strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
        #     with strategy.scope():
        #         model.build_model(mode="train")
        # else:
        #     model.build_model(mode="train")
        try:
            model.build_model(mode="train")
        except torch.OutOfMemoryError as e:
            self.errorMessage.emit(self.tr("Out of memory error.<br>Please reduce the batch size or use a smaller model."))
            aidia_logger.error(e, exc_info=True)
            return
        except Exception as e:
            self.errorMessage.emit(self.tr("Failed to build model."))
            aidia_logger.error(e, exc_info=True)
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
                logs = {
                    "epoch": self.epoch + 1,
                    "loss": self.loss,
                    "val_loss": val_loss,
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
            
            def on_val_end(val_loss):
                """Callback function for validation batch end."""
                if np.isnan(val_loss):
                    raise errors.LossGetNaNError
                logs = {
                    "epoch": self.epoch + 1,
                    "loss": self.loss,
                    "val_loss": val_loss,
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
            aidia_logger.error(e, exc_info=True)
            return
        
        # save all training setting and used data
        if isinstance(model.dataset, Dataset):
            config_path = os.path.join(self.config.dataset_dir, LOCAL_DATA_DIR_NAME, "config.json")
            shutil.copy(config_path, self.config.log_dir)
            p = os.path.join(self.config.log_dir, "dataset.json")
            model.dataset.save(p)

        ### Evaluation ###
        if self.config.is_ultralytics():
            self.notifyMessage.emit(self.tr("Done"))
            return
        elif self.config.TASK == TEST:
            self.notifyMessage.emit(self.tr("Done"))
            return
        else:
            # set inference model
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
                    aidia_logger.error(e, exc_info=True)
                    return
                image.imwrite(result_img, save_path)
        
            self.notifyMessage.emit(self.tr("Evaluating..."))
            try:
                model.evaluate()
            except Exception as e:
                self.notifyMessage.emit(self.tr("Failed to evaluate."))
                aidia_logger.error(e, exc_info=True)
                return

            self.notifyMessage.emit(self.tr("Convert model to ONNX..."))
            if not model.convert2onnx():
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
    
