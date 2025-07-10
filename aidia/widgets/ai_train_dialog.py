
import os
import shutil
import time
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Qt

from aidia import CLS, DET, SEG, MNIST, CLEAR, ERROR, AI_DIR_NAME
from aidia import ModelTypes
from aidia import LabelStyle
from aidia import aidia_logger
from aidia import qt
from aidia import utils
from aidia import errors
from aidia.image import fig2img
from aidia.ai.config import AIConfig
from aidia.ai.dataset import Dataset
from aidia.ai.test import TestModel
from aidia.ai.det import DetectionModel
from aidia.ai.seg import SegmentationModel
from aidia.widgets import ImageWidget

import torch

# Set random seeds for reproducibility
seed = AIConfig().SEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set PyTorch to use the CPU or GPU
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import keras


class AIParamComponent(object):
    """Base class for AI parameter components."""

    def __init__(self, type, tag, tips, validate_func=None, items=None, unit=None):
        super().__init__()

        if type == "text":
            self.input_field = QtWidgets.QLineEdit()
            self.input_field.setPlaceholderText(tips)
            self.input_field.setToolTip(tips)
            self.input_field.setMinimumWidth(200)
            self.input_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if validate_func is not None:
                self.input_field.textChanged.connect(validate_func)
        elif type == "combo":
            self.input_field = QtWidgets.QComboBox()
            self.input_field.setToolTip(tips)
            self.input_field.setMinimumWidth(200)
            if items is not None:
                self.input_field.addItems(items)
            if validate_func is not None:
                self.input_field.currentTextChanged.connect(validate_func)
        elif type == "checkbox":
            self.input_field = QtWidgets.QCheckBox()
            self.input_field.setToolTip(tips)
            if validate_func is not None:
                self.input_field.stateChanged.connect(validate_func)
        else:
            return None

        self.tag = QtWidgets.QLabel(tag)

        if unit is not None:
            self.unit = QtWidgets.QLabel(unit)
            self.unit.setToolTip(tips)
        
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
        self.setWindowTitle(self.tr("AI Training"))

        self.setMinimumSize(QtCore.QSize(1200, 800))

        self._layout = QtWidgets.QGridLayout()
        self._dataset_layout = QtWidgets.QVBoxLayout()
        self._dataset_widget = QtWidgets.QWidget()
        # self._dataset_widget.setMinimumWidth(200)
        self._augment_layout = QtWidgets.QGridLayout()
        self._augment_widget = QtWidgets.QWidget()
        # self._augment_widget.setMinimumWidth(200)

        self.dataset_dir = None
        self.start_time = 0
        self.epoch = []
        self.loss = []
        self.val_loss = []
        self.train_steps = 0
        self.val_steps = 0

        self.fig, self.ax_loss = plt.subplots(figsize=(12, 6))
        self.ax_loss.axis("off")
        
        self.fig2, self.ax_pie = plt.subplots(figsize=(6, 6))
        self.ax_pie.axis("off")

        self.error_flags = {}
        self.input_fields = []
        self.tags = []
        self.units = []
        self.param_idx = 0
        self.left_row = 0
        self.right_row = 0
        self.augment_row = 0

        # directory information
        self.tag_directory = QtWidgets.QLabel()
        self.tag_directory.setMaximumHeight(100)
        self._layout.addWidget(self.tag_directory, 0, 1, 1, 4)
        self.left_row += 1
        self.right_row += 1

        # task selection
        def _validate(text):
            self.config.TASK = text
            self.switch_enabled_by_task(text)
        self.param_task = AIParamComponent(
            type="combo",
            tag=self.tr("Task"),
            tips=self.tr("""Select the task.
Detection uses YOLO and Segmentation uses U-Net.
If MNIST Test are selected, the training test using MNIST dataset are performed and you can check the calculation performance."""),
            validate_func=_validate,
            items=[DET, SEG, MNIST]
        )
        self.add_param_component(self.param_task)

        # model selection
        def _validate(text):
            self.config.MODEL = text
        self.param_model = AIParamComponent(
            type="combo",
            tag=self.tr("Model"),
            tips=self.tr("""Select the model architecture."""),
            validate_func=_validate
        )
        self.param_model.input_field.setMinimumWidth(200)
        self.add_param_component(self.param_model)

        # name
        def _validate(text):
            p1 = os.path.join(self.dataset_dir, AI_DIR_NAME, text, "weights")
            p2 = os.path.join(self.dataset_dir, AI_DIR_NAME, text, "dataset.json")
            p3 = os.path.join(self.dataset_dir, AI_DIR_NAME, text, "config.json")
            if (len(text)
                and not os.path.exists(p1)
                and not os.path.exists(p2)
                and not os.path.exists(p3)):
                self._set_ok(self.param_name.tag)
                self.config.NAME = text
            else:
                self._set_error(self.param_name.tag)
        self.param_name = AIParamComponent(
            type="text",
            tag=self.tr("Name"),
            tips=self.tr("""Set the experiment name.\nYou cannot set existed experiment names."""),
            validate_func=_validate,
        )
        self.add_param_component(self.param_name)

        # dataset idx
        def _validate(text):
            self.config.DATASET_NUM = int(text.split(" ")[1])
        self.param_dataset = AIParamComponent(
            type="combo",
            tag=self.tr("Dataset"),
            tips=self.tr("""Select the dataset pattern.
Aidia splits the data into a 4:1 ratio (train:test) depend on the selected pattern.
You can use this function for 5-fold cross-validation."""),
            validate_func=_validate,
            items=["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4", "Pattern 5"]
        )
        self.add_param_component(self.param_dataset)

        # input size
        def _validate(text):
            if text.isdigit() and 32 <= int(text) <= 2048 and int(text) % 32 == 0:
                self._set_ok(self.param_size.tag)
                self.config.INPUT_SIZE = int(text)
            else:
                self._set_error(self.param_size.tag)
        self.param_size = AIParamComponent(
            type="text",
            tag=self.tr("Input Size"),
            tips=self.tr("""Set the size of input images on a side.
If you set 256, input images are resized to (256, 256)."""),
            validate_func=_validate,
        )
        self.add_param_component(self.param_size)

        # epochs
        def _validate(text):
            if text.isdigit() and 0 < int(text):
                self._set_ok(self.param_epochs.tag)
                self.config.EPOCHS = int(text)
            else:
                self._set_error(self.param_epochs.tag)
        self.param_epochs = AIParamComponent(
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
                self._set_ok(self.param_batchsize.tag)
                self.config.BATCH_SIZE = int(text)
            else:
                self._set_error(self.param_batchsize.tag)
        self.param_batchsize = AIParamComponent(
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
                self._set_ok(self.param_lr.tag)
                self.config.LEARNING_RATE = float(text)
            else:
                self._set_error(self.param_lr.tag)
        self.param_lr = AIParamComponent(
            type="text",
            tag=self.tr("Learning Rate"),
            tips=self.tr("""Set the initial learning rate of Adam.
The value is 0.001 by default.
Other parameters of Adam uses the default values of Keras 3."""),
            validate_func=_validate,
        )
        self.add_param_component(self.param_lr)

        # label definition
        def _validate():
            text = self.param_labels.input_field.toPlainText()
            text = text.strip().replace(" ", "")
            if len(text) == 0:
                self._set_error(self.param_labels.tag)
                return
            parsed = text.split("\n")
            res = [p for p in parsed if p != ""]
            res = list(dict.fromkeys(res))   # delete duplicates
            if utils.is_full_width(text):  # error if the text includes 2-bytes codes.
                self._set_error(self.param_labels.tag)
            else:
                self._set_ok(self.param_labels.tag)
                self.config.LABELS = res
        self.param_labels = AIParamComponent(
            type="text",
            tag=self.tr("Label Definition"),
            tips=self.tr("""Set target labels.
The labels are separated with line breaks."""),
            validate_func=_validate,
        )
        self.param_labels.input_field = QtWidgets.QTextEdit()
        self.param_labels.input_field.setToolTip(self.param_labels.input_field.placeholderText())
        self.param_labels.input_field.textChanged.connect(_validate)
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

        # train target select
        def _validate(state): # check:2, empty:0
            if state == 2:
                self.config.DIR_SPLIT = True
            else:
                self.config.DIR_SPLIT = False
        self.param_is_dir_split = AIParamComponent(
            type="checkbox",
            tag=self.tr("Separate Data by Directory"),
            tips=self.tr("""Separate data by directories when training."""),
            validate_func=_validate
        )
        self.add_param_component(self.param_is_dir_split, right=True, reverse=True)


        ### add augment params ###
        # title
        title_augment = qt.head_text(self.tr("Data Augmentation"))
        title_augment.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self._augment_layout.addWidget(title_augment, 0, 0, 1, 3)
        self.augment_row += 1

        # vertical flip
        self.tag_is_vflip = QtWidgets.QLabel(self.tr("Vertical Flip"))
        self.input_is_vflip = QtWidgets.QCheckBox()
        self.unit_vflip = QtWidgets.QLabel()
        self.units.append(self.unit_vflip)
        def _validate(state): # check:2, empty:0
            if state == 2:
                self.config.RANDOM_VFLIP = True
                self.unit_vflip.setText(self.tr("Enabled"))
            else:
                self.config.RANDOM_VFLIP = False
                self.unit_vflip.setText(self.tr("Disabled"))
        self.input_is_vflip.stateChanged.connect(_validate)
        self._add_augment_params(self.tag_is_vflip, self.input_is_vflip, self.unit_vflip)

        # horizontal flip
        self.tag_is_hflip = QtWidgets.QLabel(self.tr("Horizontal Flip"))
        self.input_is_hflip = QtWidgets.QCheckBox()
        self.unit_hflip = QtWidgets.QLabel()
        self.units.append(self.unit_hflip)
        def _validate(state): # check:2, empty:0
            if state == 2:
                self.config.RANDOM_HFLIP = True
                self.unit_hflip.setText(self.tr("Enabled"))
            else:
                self.config.RANDOM_HFLIP = False
                self.unit_hflip.setText(self.tr("Disabled"))
        self.input_is_hflip.stateChanged.connect(_validate)
        self._add_augment_params(self.tag_is_hflip, self.input_is_hflip, self.unit_hflip)

        # rotation
        self.tag_rotate = QtWidgets.QLabel(self.tr("Rotation"))
        self.input_rotate = self.create_input_field(50)
        self.unit_rotate = QtWidgets.QLabel()
        self.units.append(self.unit_rotate)
        def _validate(text):
            if text.isdigit() and 0 < int(text) < 90:
                self.config.RANDOM_ROTATE = int(text)
                self.unit_rotate.setText(self.tr("(-{} to +{} degree)").format(
                    int(text), int(text)
                ))
            else:
                self.config.RANDOM_ROTATE = 0
                self.unit_rotate.setText(self.tr("Disabled"))
        self.input_rotate.textChanged.connect(_validate)
        self._add_augment_params(self.tag_rotate, self.input_rotate, self.unit_rotate)

        # zoom
        self.tag_scale = QtWidgets.QLabel(self.tr("Scale"))
        self.input_scale = self.create_input_field(50)
        self.unit_scale = QtWidgets.QLabel()
        self.units.append(self.unit_scale)
        def _validate(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 1.0:
                self.config.RANDOM_SCALE = float(text)
                self.unit_scale.setText(self.tr("({:.1f} to {:.1f} times)").format(
                    1.0 - float(text), 1.0 + float(text)
                ))
            else:
                self.config.RANDOM_SCALE = 0.0
                self.unit_scale.setText(self.tr("Disabled"))
        self.input_scale.textChanged.connect(_validate)
        self._add_augment_params(self.tag_scale, self.input_scale, self.unit_scale)

        # shift
        self.tag_shift = QtWidgets.QLabel(self.tr("Shift"))
        self.input_shift = self.create_input_field(50)
        self.unit_shift = QtWidgets.QLabel()
        self.units.append(self.unit_shift)
        def _validate(text):
            if text.isdigit() and 0 < int(text) < self.config.INPUT_SIZE:
                self.config.RANDOM_SHIFT = int(text)
                self.unit_shift.setText(self.tr("({} to +{} px)").format(
                    - int(text), int(text)
                ))
            else:
                self.config.RANDOM_SHIFT = 0
                self.unit_shift.setText(self.tr("Disabled"))
        self.input_shift.textChanged.connect(_validate)
        self._add_augment_params(self.tag_shift, self.input_shift, self.unit_shift)

        # shear
        self.tag_shear = QtWidgets.QLabel(self.tr("Shear"))
        self.input_shear = self.create_input_field(50)
        self.unit_shear = QtWidgets.QLabel()
        self.units.append(self.unit_shear)
        def _validate(text):
            if text.isdigit() and 0 < int(text) < 30:
                self.config.RANDOM_SHEAR = int(text)
                self.unit_shear.setText(self.tr("(-{} to +{} degree)").format(
                    int(text), int(text)
                ))
            else:
                self.config.RANDOM_SHEAR = 0
                self.unit_shear.setText(self.tr("Disabled"))
        self.input_shear.textChanged.connect(_validate)
        self._add_augment_params(self.tag_shear, self.input_shear, self.unit_shear)

        # blur
        self.tag_blur = QtWidgets.QLabel(self.tr("Blur"))
        self.input_blur = self.create_input_field(50)
        self.unit_blur = QtWidgets.QLabel()
        self.units.append(self.unit_blur)
        def _validate(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 20.0:
                self.config.RANDOM_BLUR = float(text)
                self.unit_blur.setText(self.tr("(std = 0.0 to {})").format(
                    float(text)
                ))
            else:
                self.config.RANDOM_BLUR = 0.0
                self.unit_blur.setText(self.tr("Disabled"))
        self.input_blur.textChanged.connect(_validate)
        self._add_augment_params(self.tag_blur, self.input_blur, self.unit_blur)

        # noise
        self.tag_noise = QtWidgets.QLabel(self.tr("Noise"))
        self.input_noise = self.create_input_field(50)
        self.unit_noise = QtWidgets.QLabel()
        self.units.append(self.unit_noise)
        def _validate(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 1.0:
                self.config.RANDOM_NOISE = float(text)
                self.unit_noise.setText(self.tr("(std = {})").format(
                    float(text)
                ))
            else:
                self.config.RANDOM_NOISE = 0.0
                self.unit_noise.setText(self.tr("Disabled"))
        self.input_noise.textChanged.connect(_validate)
        self._add_augment_params(self.tag_noise, self.input_noise, self.unit_noise)

        # brightness
        self.tag_brightness = QtWidgets.QLabel(self.tr("Brightness"))
        self.input_brightness = self.create_input_field(50)
        self.unit_brightness = QtWidgets.QLabel()
        self.units.append(self.unit_brightness)
        def _validate(text):
            if text.isdigit() and 0 < int(text) < 255:
                self.config.RANDOM_BRIGHTNESS = int(text)
                self.unit_brightness.setText(self.tr("({} to {})").format(
                    - int(text), int(text)
                ))
            else:
                self.config.RANDOM_BRIGHTNESS = 0
                self.unit_brightness.setText(self.tr("Disabled"))
        self.input_brightness.textChanged.connect(_validate)
        self._add_augment_params(self.tag_brightness, self.input_brightness, self.unit_brightness)

        # contrast
        self.tag_contrast = QtWidgets.QLabel(self.tr("Contrast"))
        self.input_contrast = self.create_input_field(50)
        self.unit_contrast = QtWidgets.QLabel()
        self.units.append(self.unit_contrast)
        def _validate(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 1.0:
                self.config.RANDOM_CONTRAST = float(text)
                self.unit_contrast.setText(self.tr("({:.1f} to {:.1f} times)").format(
                    1.0 - float(text), 1.0 + float(text)
                ))
            else:
                self.config.RANDOM_CONTRAST = 0.0
                self.unit_contrast.setText(self.tr("Disabled"))
        self.input_contrast.textChanged.connect(_validate)
        self._add_augment_params(self.tag_contrast, self.input_contrast, self.unit_contrast)

        ### add buttons ###
        # train button
        self.button_train = QtWidgets.QPushButton(self.tr("Train"))
        self.button_train.clicked.connect(self.train)
        row = max(self.left_row, self.right_row)
        self._layout.addWidget(self.button_train, row, 1, 1, 4)
        row += 1

        # figure area
        self.image_widget = ImageWidget(self, self._plt2img())
        self._layout.addWidget(self.image_widget, row, 1, 1, 4)
        row += 1

        # progress bar
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self._layout.addWidget(self.progress, row, 1, 1, 4)
        row += 1

        # status
        self.text_status = QtWidgets.QLabel()
        self._layout.addWidget(self.text_status, row, 1, 1, 3)
        # row += 1

        # stop button
        self.button_stop = QtWidgets.QPushButton(self.tr("Terminate"))
        def _stop_training():
            self.ai.quit()
            self.button_stop.setEnabled(False)
        self.button_stop.clicked.connect(_stop_training)
        self._layout.addWidget(self.button_stop, row, 4, 1, 1, Qt.AlignRight)
        # row += 1

        ### add dataset information ###
        # title
        title_dataset = qt.head_text(self.tr("Dataset Information"))
        title_dataset.setMaximumHeight(100)
        title_dataset.setAlignment(Qt.AlignTop)
        self._dataset_layout.addWidget(title_dataset)

        # dataset information
        self.text_dataset = QtWidgets.QLabel()
        self.text_dataset.setAlignment(Qt.AlignLeading)
        self._dataset_layout.addWidget(self.text_dataset)

        self.image_widget2 = ImageWidget(self, self._plt2img2())
        self._dataset_layout.addWidget(self.image_widget2)

        ### set layouts ###
        self._augment_widget.setLayout(self._augment_layout)
        self._layout.addWidget(self._augment_widget, 0, 5, row - 1, 1)
        self._dataset_widget.setLayout(self._dataset_layout)
        self._layout.addWidget(self._dataset_widget, 0, 0, row + 1, 1)

        self.setLayout(self._layout)

        # connect AI thread
        self.ai = AITrainThread(self)
        self.ai.fitStarted.connect(self.callback_fit_started)
        self.ai.notifyMessage.connect(self.update_status)
        self.ai.errorMessage.connect(self.popup_error)
        self.ai.datasetInfo.connect(self.update_dataset)
        self.ai.epochLogList.connect(self.update_logs)
        self.ai.batchLogList.connect(self.update_batch)
        self.ai.finished.connect(self.ai_finished)

        self.text_status.setText(self.tr("Ready"))


    def popup(self, dataset_dir, is_submode=False, data_labels=None):
        """Popup train window and set config parameters to input fields."""
        self.dataset_dir = dataset_dir
        self.setWindowTitle(self.tr("AI Training - {}").format(dataset_dir))
        if is_submode and len(os.listdir(dataset_dir)) > 1:
            dir_list = glob.glob(os.path.join(dataset_dir, "*/"))
            self.tag_directory.setText(self.tr("Target Directory:\n{},\n{},\n...").format(dir_list[0], dir_list[1]))
        else:
            self.tag_directory.setText(self.tr("Target Directory:\n{}").format(dataset_dir))

        # create data directory
        data_dirpath = utils.get_dirpath_with_mkdir(dataset_dir, AI_DIR_NAME)
        # if not os.path.exists(os.path.join(dataset_dir, AI_DIR_NAME)):
            # os.mkdir(os.path.join(dataset_dir, AI_DIR_NAME))

        # load config parameters
        self.config = AIConfig(dataset_dir)
        config_path = os.path.join(data_dirpath, "config.json")
        if os.path.exists(config_path):
            try:
                self.config.load(config_path)
            except Exception as e:
                aidia_logger.error(e, exc_info=True)
        self.config.SUBMODE = is_submode

        # basic params
        self.param_task.input_field.setCurrentText(self.config.TASK)
        self.switch_enabled_by_task(self.config.TASK)
        self.param_model.input_field.setCurrentText(self.config.MODEL)
        self.param_name.input_field.setText(self.config.NAME)
        self.param_dataset.input_field.setCurrentText("Pattern " + str(self.config.DATASET_NUM))
        self.param_size.input_field.setText(str(self.config.INPUT_SIZE))
        self.param_epochs.input_field.setText(str(self.config.EPOCHS))
        self.param_batchsize.input_field.setText(str(self.config.BATCH_SIZE))
        self.param_lr.input_field.setText(str(self.config.LEARNING_RATE))

        if data_labels:
            self.param_labels.input_field.setText("\n".join(data_labels))
        else:
            self.param_labels.input_field.setText("\n".join(self.config.LABELS))
        # if self.config.gpu_num < 2:
        #     self.input_is_multi.setEnabled(False)
        # self.input_is_multi.setChecked(self.config.USE_MULTI_GPUS)
        # self.input_is_savebest.setChecked(self.config.SAVE_BEST)
        # self.param_is_earlystop.input_field.setChecked(self.config.EARLY_STOPPING)
        if not self.config.SUBMODE:
            self.param_is_dir_split.input_field.setEnabled(False)
        self.param_is_dir_split.input_field.setChecked(self.config.DIR_SPLIT)

        # augment params
        self.input_is_vflip.setChecked(self.config.RANDOM_VFLIP)
        self.input_is_hflip.setChecked(self.config.RANDOM_HFLIP)
        self.input_rotate.setText(str(self.config.RANDOM_ROTATE))
        self.input_shift.setText(str(self.config.RANDOM_SHIFT))
        self.input_scale.setText(str(self.config.RANDOM_SCALE))
        self.input_shear.setText(str(self.config.RANDOM_SHEAR))
        self.input_blur.setText(str(self.config.RANDOM_BLUR))
        self.input_noise.setText(str(self.config.RANDOM_NOISE))
        self.input_brightness.setText(str(self.config.RANDOM_BRIGHTNESS))
        self.input_contrast.setText(str(self.config.RANDOM_CONTRAST))

        self.exec_()
        if os.path.exists(os.path.join(dataset_dir, AI_DIR_NAME)):
            self.config.save(config_path)
    
    ### Callbacks ###
    def ai_finished(self):
        """Call back function when AI thread finished."""
        self.switch_enabled_by_task(self.config.TASK)

        # raise error handle
        config_path = os.path.join(self.config.log_dir, "config.json")
        dataset_path = os.path.join(self.config.log_dir, "dataset.json")
        if not os.path.exists(config_path) or not os.path.exists(dataset_path):
            # self.text_status.setText(self.tr("Training was failed."))
            self.reset_state()
            self.aiRunning.emit(False)
            self.text_status.setText(self.tr("Terminated training."))
            return
        
        self._set_error(self.param_name.tag) # to avoid NAME duplication.

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
        self.fig.savefig(os.path.join(self.config.log_dir, "loss.png"))

        self.aiRunning.emit(False)

    def callback_fit_started(self, value):
        self.button_stop.setEnabled(True)

    def switch_enabled_by_task(self, task):
        if task == CLS:
            raise NotImplementedError
        
        elif task in [DET, SEG]:
            self.param_model.input_field.clear()
            if task == DET:
                self.param_model.input_field.addItems(ModelTypes.DET_MODEL)
            elif task == SEG:
                self.param_model.input_field.addItems(ModelTypes.SEG_MODEL)
            self.enable_all()

        elif task == MNIST:
            self.param_model.input_field.clear()
            self.disable_all()
            self.switch_enabled([
                self.param_name.tag,
                self.param_batchsize.tag,
                self.param_epochs.tag,
                self.param_lr.tag,
                self.param_task.tag], True)
            self.button_train.setEnabled(True)

        else:
            raise ValueError

        # global setting
        self.switch_global_params()
        self.button_train.setEnabled(True)
        self.button_stop.setEnabled(False)

    def switch_enabled(self, targets:list[QtWidgets.QLabel], enabled:bool):
        for t in targets:
            if enabled:
                t.setStyleSheet(LabelStyle.DEFAULT)
            else:
                t.setStyleSheet(LabelStyle.DISABLED)
            i = self.tags.index(t)
            self.input_fields[i].setEnabled(enabled)
        # if enabled and self.config.gpu_num < 2:
        #     self.tag_is_multi.setStyleSheet(LabelStyle.DISABLED)
        #     self.input_is_multi.setEnabled(False)            if enabled and not self.config.SUBMODE:
            self.param_is_dir_split.tag.setStyleSheet(LabelStyle.DISABLED)
            self.param_is_dir_split.input_field.setEnabled(False)

    def switch_global_params(self):
        # if self.config.gpu_num < 2:
        #     self.tag_is_multi.setStyleSheet(LabelStyle.DISABLED)
        #     self.input_is_multi.setEnabled(False)
        # else:
        #     self.tag_is_multi.setStyleSheet(LabelStyle.DEFAULT)
        #     self.input_is_multi.setEnabled(True)
        if not self.config.SUBMODE or self.config.TASK in [MNIST]:
            self.param_is_dir_split.tag.setStyleSheet(LabelStyle.DISABLED)
            self.param_is_dir_split.input_field.setEnabled(False)
        else:
            self.param_is_dir_split.tag.setStyleSheet(LabelStyle.DEFAULT)
            self.param_is_dir_split.input_field.setEnabled(True)
    
    def enable_all(self):
        for x in self.input_fields:
            x.setEnabled(True)
        for x in self.tags:
            x.setStyleSheet(LabelStyle.DEFAULT)
        for x in self.units:
            x.setStyleSheet(LabelStyle.DEFAULT)
        for i, v in enumerate(self.error_flags.values()):
            if v == 1:
                self.tags[i].setStyleSheet(LabelStyle.ERROR)

    
    def disable_all(self):
        for x in self.input_fields:
            x.setEnabled(False)
        for x in self.tags:
            x.setStyleSheet(LabelStyle.DISABLED)
        for x in self.units:
            x.setStyleSheet(LabelStyle.DISABLED)
        self.button_train.setEnabled(False)

    def closeEvent(self, event):
        pass
        
    def showEvent(self, event):
        if self.ai.isRunning():
            self.disable_all()
            self.button_stop.setEnabled(True)
        else:
            # self.reset_state()
            self.switch_enabled_by_task(self.config.TASK)

    def add_param_component(self, object:AIParamComponent, right=False, reverse=False, custom_size=None):
        self.error_flags[object.tag.text()] = 0
        self.tags.append(object.tag)
        self.input_fields.append(object.input_field)
        row = self.left_row
        pos = [1, 2]
        align = [Qt.AlignmentFlag.AlignRight, Qt.AlignmentFlag.AlignLeft]
        h, w = (1, 1)
        if right:
            row = self.right_row
            pos = [3, 4]
        if reverse:
            pos = pos[::-1]
            align = align[::-1]
        if custom_size:
            h = custom_size[0]
            w = custom_size[1]
        self._layout.addWidget(object.tag, row, pos[0], h, w, alignment=align[0])
        self._layout.addWidget(object.input_field, row, pos[1], h, w, alignment=align[1])
        if right:
            self.right_row += h
        else:
            self.left_row += h
    
    def _add_augment_params(self, tag, widget, unit=None):
        self.error_flags[tag.text()] = 0
        self.tags.append(tag)
        self.input_fields.append(widget)
        row = self.augment_row
        pos = [0, 1, 2]
        align = [Qt.AlignRight, Qt.AlignCenter, Qt.AlignLeft]
        self._augment_layout.addWidget(tag, row, pos[0], alignment=align[0])
        self._augment_layout.addWidget(widget, row, pos[1], alignment=align[1])
        if unit is not None:
            self._augment_layout.addWidget(unit, row, pos[2], alignment=align[2])
        self.augment_row += 1

    def _set_error(self, tag:QtWidgets.QLabel):
        tag.setStyleSheet(LabelStyle.ERROR)
        self.error_flags[tag.text()] = ERROR

    def _set_ok(self, tag:QtWidgets.QLabel):
        tag.setStyleSheet(LabelStyle.DEFAULT)
        self.error_flags[tag.text()] = CLEAR

    def update_figure(self):
        self.ax_loss.clear()
        self.ax_loss.set_title('Loss Curve', fontsize=20)
        if len(self.epoch):
            if len(self.loss):
                self.ax_loss.plot(self.epoch, self.loss, color="red", linestyle="solid", label="Training")
            if len(self.val_loss):
                self.ax_loss.plot(self.epoch, self.val_loss, color="green", linestyle="solid", label="Validation")
            self.ax_loss.set_xlabel("Epoch", fontsize=16)
            self.ax_loss.set_ylabel("Loss", fontsize=16)
            self.ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
            mx = min((len(self.epoch) // 10 + 1) * 10, self.config.EPOCHS)
            self.ax_loss.set_xlim([1, mx])
            if len(self.val_loss) > 1:
                top_limit = np.max(self.val_loss[1:])
                # top_limit = np.mean(self.val_loss[1:]) * 1.5
                self.ax_loss.set_ylim([0, top_limit])
            self.ax_loss.legend(fontsize=16)
            self.ax_loss.grid()
            self.image_widget.loadPixmap(self._plt2img())

    def _plt2img(self):
        return fig2img(self.fig)

    def _plt2img2(self):
        return fig2img(self.fig2)

    def update_dataset(self, value):
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
        self.ax_pie.set_title('Label Distribusion', fontsize=20)
        self.ax_pie.pie(num_per_class,
                    labels=class_names,
                    #  autopct="%1.1f%%",
                    wedgeprops={'linewidth': 1, 'edgecolor':"white"},
                    textprops={'color': "black", 'fontsize': 16})
        self.image_widget2.loadPixmap(self._plt2img2())

    def update_status(self, value):
        self.text_status.setText(str(value))

    def popup_error(self, text):
        self.parent().error_message(text)

    def update_batch(self, value):
        epoch = len(self.epoch) + 1
        batch = value.get("batch")
        loss = value.get("loss")

        text = f"Epoch: {epoch}/{self.config.EPOCHS} "
        if batch is not None:
            text += f"Batch: {batch} / {self.train_steps} "
        if loss is not None:
            text += f"Loss: {loss:.6f} "
        if len(self.val_loss):
            text += f"Val Loss: {self.val_loss[-1]:.6f}"
        
        self.text_status.setText(text)
    
    def update_logs(self, value):
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

        self.update_figure()
                

    def create_input_field(self, size):
        l = QtWidgets.QLineEdit()
        l.setAlignment(Qt.AlignCenter)
        # l.setMaximumWidth(size)
        # l.setMinimumWidth(size)
        return l


    def _print_errors(self):
        for tag_text, flag in self.error_flags.items():
            if flag == ERROR:
                if tag_text == self.tr("Name"):
                    self.text_status.setText(self.tr("Change the name."))
                    return
                if tag_text == self.tr("Input Size"):
                    self.text_status.setText(self.tr("Set an appropriate input size."))
                    return
                if tag_text == self.tr("Epochs"):
                    self.text_status.setText(self.tr("Set an appropriate epochs."))
                    return
                if tag_text == self.tr("Batch Size"):
                    self.text_status.setText(self.tr("Set an appropriate batch size."))
                    return
                if tag_text == self.tr("Learning Rate"):
                    self.text_status.setText(self.tr("Set an appropriate learning rate."))
                    return
                if tag_text == self.tr("Label Definition"):
                    self.text_status.setText(self.tr("Set an appropriate label definition."))
                    return


    def train(self):
        error = sum(self.error_flags.values())
        if error > 0:
            self._print_errors()
            # self.text_status.setText(self.tr("Please check parameters."))
            return
        
        self.disable_all()
        self.reset_state()

        self.config.build_params()  # update parameters

        config_path = os.path.join(self.dataset_dir, AI_DIR_NAME, "config.json")
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
        self.image_widget.clear()
        self.image_widget2.clear()


class AITrainThread(QtCore.QThread):

    fitStarted = QtCore.Signal(bool)
    epochLogList = QtCore.Signal(dict)
    batchLogList = QtCore.Signal(dict)
    notifyMessage = QtCore.Signal(str)
    errorMessage = QtCore.Signal(str)
    datasetInfo = QtCore.Signal(dict)

    def __init__(self, parent):
        super().__init__(parent)
        self.config = None
        self.model = None
        self.cb_getprocess = None

    def set_config(self, config: AIConfig):
        self.config = config

    def quit(self):
        if self.cb_getprocess.is_fitting:
            super().quit()
            self.model.stop_training()
            self.notifyMessage.emit(self.tr("Interrupt training."))
            return
        else:
            self.notifyMessage.emit(self.tr("Fitting process has not started yet."))
            return

    def run(self):
        if self.config is None:
            self.errorMessage.emit(self.tr("Not configured. Terminated."))
            return

        model = None
        if self.config.TASK == MNIST:
            model = TestModel(self.config)
        elif self.config.TASK == DET:
            model = DetectionModel(self.config)
        elif self.config.TASK == SEG:
            model = SegmentationModel(self.config)
        else:
            self.errorMessage.emit(self.tr("Model error. Terminated."))
            return
        self.model = model
        
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
        model.build_model(mode="train")

        self.notifyMessage.emit(self.tr("Preparing..."))

        # set callbacks
        self.batch = 0
        self.epoch = 0
        self.loss = 0.0
        
        # set progress callback
        if ModelTypes.is_ultralytics(self.config.MODEL):
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
            class GetProgress(keras.callbacks.Callback):
                """Custom keras callback to get progress values while AI training."""
                def __init__(self, widget: AITrainThread):
                    super().__init__()

                    self.widget = widget
                    self.is_fitting = False

                def on_train_batch_end(self, batch, logs=None):
                    if logs is not None:
                        logs["batch"] = batch + 1
                        self.widget.batchLogList.emit(logs)
                        if not self.is_fitting:
                            self.is_fitting = True
                            self.widget.fitStarted.emit(True)

                def on_epoch_end(self, epoch, logs=None):
                    if logs is not None:
                        if np.isnan(logs.get("loss")) or np.isnan(logs.get("val_loss")):
                            self.widget.model.stop_training()
                            raise errors.LossGetNanError
                        logs["epoch"] = epoch + 1
                        self.widget.epochLogList.emit(logs)

            progress_callback = GetProgress(self)
            callbacks = [progress_callback]
            self.cb_getprocess = progress_callback
            
        # if self.config.EARLY_STOPPING:
        #     cb = [cb_getprocess, keras.callbacks.EarlyStopping(patience=10)]
        # else:
        #     cb = [cb_getprocess]

        try:
            model.train(callbacks)
        # except tf.errors.ResourceExhaustedError as e:
        #     self.errorMessage.emit(self.tr("Memory error. Please reduce the input size or batch size."))
        #     aidia_logger.error(e, exc_info=True)
        #     return
        # except tf.errors.NotFoundError as e:
        #     self.errorMessage.emit(self.tr("Memory error. Please reduce the input size or batch size."))
        #     aidia_logger.error(e, exc_info=True)
        #     return
        except errors.LossGetNanError as e:
            self.errorMessage.emit(self.tr("Loss got NaN. Please adjust the learning rate."))
            aidia_logger.error(e, exc_info=True)
            return
        except Exception as e:
            self.errorMessage.emit(self.tr("Failed to train."))
            aidia_logger.error(e, exc_info=True)
            return
        
        # save all training setting and used data
        if isinstance(model.dataset, Dataset):
            config_path = os.path.join(self.config.dataset_dir, AI_DIR_NAME, "config.json")
            shutil.copy(config_path, self.config.log_dir)
            p = os.path.join(self.config.log_dir, "dataset.json")
            model.dataset.save(p)


