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

from aidia import CLS, DET, SEG, MNIST, CLEAR, ERROR, TASK_LIST
from aidia import LOCAL_DATA_DIR_NAME, CONFIG_JSON, DATASET_JSON
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


class ParamComponent(object):
    """Base class for AI parameter components."""

    def __init__(self, type, tag, tips, validate_func=None, items=None):
        super().__init__()

        if type == "text":
            self.input_field = QtWidgets.QLineEdit()
            self.input_field.setPlaceholderText(tips)
            self.input_field.setToolTip(tips)
            self.input_field.setMinimumWidth(200)
            self.input_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if validate_func is not None:
                self.input_field.textChanged.connect(validate_func)
        elif type == "textbox":
            self.input_field = QtWidgets.QTextEdit()
            self.input_field.setPlaceholderText(tips)
            self.input_field.setToolTip(tips)
            self.input_field.setMinimumWidth(200)
            if validate_func is not None:
                self.input_field.textChanged.connect(validate_func)
        elif type == "combo":
            self.input_field = QtWidgets.QComboBox()
            self.input_field.setToolTip(tips)
            self.input_field.setMinimumWidth(200)
            if items is not None:
                self.input_field.addItems(items)
            if validate_func is not None:
                self.input_field.currentIndexChanged.connect(validate_func)
        elif type == "checkbox":
            self.input_field = QtWidgets.QCheckBox()
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

        self.fig_loss, self.ax_loss = plt.subplots(figsize=(12, 6))
        self.fig_loss.patch.set_alpha(0.0)
        self.ax_loss.axis("off")
        
        self.fig_pie, self.ax_pie = plt.subplots(figsize=(6, 6))
        self.fig_pie.patch.set_alpha(0.0)
        self.ax_pie.axis("off")

        self.param_objects = {}

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
        def _validate(idx):
            idx = int(idx)
            self.config.TASK = TASK_LIST[idx]
            self.switch_enabled_by_task(self.config.TASK)
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
            tag=self.tr("Name"),
            tips=self.tr("""Set the experiment name."""),
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
        def _validate(text):
            if text.isdigit() and 32 <= int(text) <= 2048 and int(text) % 32 == 0:
                self.set_ok(self.param_size)
                self.config.INPUT_SIZE = int(text)
            else:
                self.set_error(self.param_size)
        self.param_size = ParamComponent(
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
        self.add_param_component(self.param_is_dir_split, right=True, reverse=True)


        ### add augment params ###
        # title
        title_augment = qt.head_text(self.tr("Data Augmentation"))
        title_augment.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self._augment_layout.addWidget(title_augment, 0, 0, 1, 3)
        self.augment_row += 1

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
        def _validate_rotate(text):
            if text.isdigit() and 0 < int(text) < 90:
                self.config.RANDOM_ROTATE = int(text)
            else:
                self.config.RANDOM_ROTATE = 0
        self.param_rotate = ParamComponent(
            type="text",
            tag=self.tr("Rotation"),
            tips=self.tr("Set rotation angle range in degrees (0-90)."),
            validate_func=_validate_rotate
        )
        self.add_augment_param(self.param_rotate)

        # scale
        def _validate_scale(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 1.0:
                self.config.RANDOM_SCALE = float(text)
            else:
                self.config.RANDOM_SCALE = 0.0
        self.param_scale = ParamComponent(
            type="text",
            tag=self.tr("Scale"),
            tips=self.tr("Set scale variation range (0.0-1.0)."),
            validate_func=_validate_scale
        )
        self.add_augment_param(self.param_scale)

        # shift
        def _validate_shift(text):
            if text.isdigit() and 0 < int(text) < self.config.INPUT_SIZE:
                self.config.RANDOM_SHIFT = int(text)
            else:
                self.config.RANDOM_SHIFT = 0
        self.param_shift = ParamComponent(
            type="text",
            tag=self.tr("Shift"),
            tips=self.tr("Set shift range in pixels."),
            validate_func=_validate_shift
        )
        self.add_augment_param(self.param_shift)

        # shear
        def _validate_shear(text):
            if text.isdigit() and 0 < int(text) < 30:
                self.config.RANDOM_SHEAR = int(text)
            else:
                self.config.RANDOM_SHEAR = 0
        self.param_shear = ParamComponent(
            type="text",
            tag=self.tr("Shear"),
            tips=self.tr("Set shear angle range in degrees (0-30)."),
            validate_func=_validate_shear
        )
        self.add_augment_param(self.param_shear)

        # blur
        def _validate_blur(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 20.0:
                self.config.RANDOM_BLUR = float(text)
            else:
                self.config.RANDOM_BLUR = 0.0
        self.param_blur = ParamComponent(
            type="text",
            tag=self.tr("Blur"),
            tips=self.tr("Set blur standard deviation (0.0-20.0)."),
            validate_func=_validate_blur
        )
        self.add_augment_param(self.param_blur)

        # noise
        def _validate_noise(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 1.0:
                self.config.RANDOM_NOISE = float(text)
            else:
                self.config.RANDOM_NOISE = 0.0
        self.param_noise = ParamComponent(
            type="text",
            tag=self.tr("Noise"),
            tips=self.tr("Set noise standard deviation (0.0-1.0)."),
            validate_func=_validate_noise
        )
        self.add_augment_param(self.param_noise)

        # brightness
        def _validate_brightness(text):
            if text.isdigit() and 0 < int(text) < 255:
                self.config.RANDOM_BRIGHTNESS = int(text)
            else:
                self.config.RANDOM_BRIGHTNESS = 0
        self.param_brightness = ParamComponent(
            type="text",
            tag=self.tr("Brightness"),
            tips=self.tr("Set brightness adjustment range (0-255)."),
            validate_func=_validate_brightness
        )
        self.add_augment_param(self.param_brightness)

        # contrast
        def _validate_contrast(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 1.0:
                self.config.RANDOM_CONTRAST = float(text)
            else:
                self.config.RANDOM_CONTRAST = 0.0
        self.param_contrast = ParamComponent(
            type="text",
            tag=self.tr("Contrast"),
            tips=self.tr("Set contrast variation range (0.0-1.0)."),
            validate_func=_validate_contrast
        )
        self.add_augment_param(self.param_contrast)

        ### add buttons ###

        # update lowest row
        row = max(self.left_row, self.right_row)

        # train button
        self.button_train = QtWidgets.QPushButton(self.tr("Train"))
        self.button_train.setMinimumHeight(100)
        self.button_train.setStyleSheet("font-size: 20px;")
        self.button_train.clicked.connect(self.train)
        self._layout.addWidget(self.button_train, row, 1, 1, 4)
        row += 1

        # figure area
        self.image_widget_loss = ImageWidget(self)
        self._layout.addWidget(self.image_widget_loss, row, 1, 1, 4)
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

        self.image_widget_pie = ImageWidget(self)
        self._dataset_layout.addWidget(self.image_widget_pie)

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
        data_dirpath = utils.get_dirpath_with_mkdir(dataset_dir, LOCAL_DATA_DIR_NAME)
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
        self.param_task.input_field.setCurrentIndex(TASK_LIST.index(self.config.TASK))
        self.switch_enabled_by_task(self.config.TASK)
        self.param_model.input_field.setCurrentText(self.config.MODEL)
        self.param_name.input_field.setText(self.config.NAME)
        self.param_dataset.input_field.setCurrentIndex(int(self.config.DATASET_NUM) - 1)
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
        self.param_vflip.input_field.setChecked(self.config.RANDOM_VFLIP)
        self.param_hflip.input_field.setChecked(self.config.RANDOM_HFLIP)
        self.param_rotate.input_field.setText(str(self.config.RANDOM_ROTATE))
        self.param_shift.input_field.setText(str(self.config.RANDOM_SHIFT))
        self.param_scale.input_field.setText(str(self.config.RANDOM_SCALE))
        self.param_shear.input_field.setText(str(self.config.RANDOM_SHEAR))
        self.param_blur.input_field.setText(str(self.config.RANDOM_BLUR))
        self.param_noise.input_field.setText(str(self.config.RANDOM_NOISE))
        self.param_brightness.input_field.setText(str(self.config.RANDOM_BRIGHTNESS))
        self.param_contrast.input_field.setText(str(self.config.RANDOM_CONTRAST))

        self.exec_()
        if os.path.exists(os.path.join(dataset_dir, LOCAL_DATA_DIR_NAME)):
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

        self.aiRunning.emit(False)

        # open log directory
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.config.log_dir))

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
                self.param_name,
                self.param_batchsize,
                self.param_epochs,
                self.param_lr,
                self.param_task], True)
            self.button_train.setEnabled(True)

        else:
            raise ValueError

        # global setting
        self.switch_global_params()
        self.button_train.setEnabled(True)
        self.button_stop.setEnabled(False)

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
        if not self.config.SUBMODE or self.config.TASK in [MNIST]:
            self.param_is_dir_split.tag.setStyleSheet(LabelStyle.DISABLED)
            self.param_is_dir_split.input_field.setEnabled(False)
        else:
            self.param_is_dir_split.tag.setStyleSheet(LabelStyle.DEFAULT)
            self.param_is_dir_split.input_field.setEnabled(True)
    
    def enable_all(self):
        for obj in self.param_objects.values():
            obj.input_field.setEnabled(True)
            obj.tag.setStyleSheet(LabelStyle.DEFAULT)
            if obj.state == ERROR:
                obj.tag.setStyleSheet(LabelStyle.ERROR)

    
    def disable_all(self):
        for obj in self.param_objects.values():
            obj.input_field.setEnabled(False)
            obj.tag.setStyleSheet(LabelStyle.DISABLED)
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

    def add_param_component(self, obj:ParamComponent, right=False, reverse=False, custom_size=None):
        self.param_objects[obj.tag.text()] = obj
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
        self._layout.addWidget(obj.tag, row, pos[0], h, w, alignment=align[0])
        self._layout.addWidget(obj.input_field, row, pos[1], h, w, alignment=align[1])
        if right:
            self.right_row += h
        else:
            self.left_row += h
    
    def add_augment_param(self, obj:ParamComponent):
        self.param_objects[obj.tag.text()] = obj
        row = self.augment_row
        self._augment_layout.addWidget(obj.tag, row, 0, alignment=Qt.AlignmentFlag.AlignRight)
        self._augment_layout.addWidget(obj.input_field, row, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        self.augment_row += 1

   
    def set_error(self, obj: ParamComponent):
        obj.tag.setStyleSheet(LabelStyle.ERROR)
        obj.state = ERROR

    def set_ok(self, obj: ParamComponent):
        obj.tag.setStyleSheet(LabelStyle.DEFAULT)
        obj.state = CLEAR

    def update_figure(self):
        self.ax_loss.clear()
        self.ax_loss.set_xlabel("Epoch", fontsize=16, color=qt.get_default_color())
        self.ax_loss.set_ylabel("Loss", fontsize=16, color=qt.get_default_color())
        self.ax_loss.tick_params(axis='both', labelsize=14, colors=qt.get_default_color())
        self.ax_loss.patch.set_alpha(0.0)
        self.ax_loss.grid()
        if len(self.epoch):
            if len(self.loss):
                self.ax_loss.plot(self.epoch, self.loss, color="red", linestyle="solid", label="train")
            if len(self.val_loss):
                self.ax_loss.plot(self.epoch, self.val_loss, color="green", linestyle="solid", label="val")
            self.ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
            mx = min((len(self.epoch) // 10 + 1) * 10, self.config.EPOCHS)
            self.ax_loss.set_xlim([1, mx])
            self.ax_loss.legend(fontsize=16, labelcolor=qt.get_default_color(), framealpha=0.3)
            self.add_fig_loss()

    def add_fig_loss(self):
        self.image_widget_loss.loadPixmap(fig2img(self.fig_loss, add_alpha=True))
        return

    def add_fig_pie(self):
        self.image_widget_pie.loadPixmap(fig2img(self.fig_pie, add_alpha=True))
        return

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
        self.ax_pie.set_title('Label Distribusion', fontsize=20, color="white" if qt.is_dark_mode() else "black")
        self.ax_pie.pie(num_per_class,
                    labels=class_names,
                    #  autopct="%1.1f%%",
                    wedgeprops={'linewidth': 1, 'edgecolor':"white"},
                    textprops={'color': "white" if qt.is_dark_mode() else "black",
                               'fontsize': 16})
        self.add_fig_pie()

    def update_status(self, value):
        self.text_status.setText(str(value))

    def popup_error(self, text):
        self.parent().error_message(text)

    def update_batch(self, value):
        epoch = len(self.epoch) + 1
        batch = value.get("batch")
        loss = value.get("loss")

        text = f"epoch: {epoch}/{self.config.EPOCHS} "
        if batch is not None:
            text += f"batch: {batch} / {self.train_steps} "
        if loss is not None:
            text += f"loss: {loss:.6f} "
        if len(self.val_loss):
            text += f"val_loss: {self.val_loss[-1]:.6f}"
        
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
                

    def check_errors(self):
        for tag_text, obj in self.param_objects.items():
            if obj.state == ERROR:
                self.text_status.setText(self.tr("Please check {}.").format(tag_text))
                return False
        return True
    
    def may_continue(self, message="Continue?"):
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

        if self.config.log_dir is not None and os.path.exists(os.path.join(self.config.log_dir, "config.json")):
            answer = self.may_continue(self.tr("'{}' already exists. Overwrite?").format(os.path.basename(self.config.log_dir)))
            if not answer:
                self.text_status.setText(self.tr("Training was cancelled."))
                return
            else:
                shutil.rmtree(self.config.log_dir, ignore_errors=True)
        
        self.disable_all()
        self.reset_state()

        self.config.build_params()  # update parameters

        config_path = os.path.join(self.dataset_dir, LOCAL_DATA_DIR_NAME, "config.json")
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

    def set_config(self, config: AIConfig):
        self.config = config

    def quit(self):
        super().quit()
        self.model.stop_training()
        self.notifyMessage.emit(self.tr("Interrupt training."))
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

        # set progress callback
        self.batch = 0
        self.epoch = 0
        self.loss = 0.0

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

                def on_train_batch_end(self, batch, logs=None):
                    if logs is not None:
                        logs["batch"] = batch + 1
                        self.widget.batchLogList.emit(logs)

                def on_epoch_end(self, epoch, logs=None):
                    if logs is not None:
                        if np.isnan(logs.get("loss")) or np.isnan(logs.get("val_loss")):
                            self.widget.model.stop_training()
                            raise errors.LossGetNanError
                        logs["epoch"] = epoch + 1
                        self.widget.epochLogList.emit(logs)

            progress_callback = GetProgress(self)
            callbacks = [progress_callback]
            
        try:
            self.fitStarted.emit(True)
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
            config_path = os.path.join(self.config.dataset_dir, LOCAL_DATA_DIR_NAME, "config.json")
            shutil.copy(config_path, self.config.log_dir)
            p = os.path.join(self.config.log_dir, "dataset.json")
            model.dataset.save(p)


