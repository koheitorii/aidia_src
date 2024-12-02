
import os
import logging
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from qtpy import QtCore, QtWidgets, QtGui

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from aidia import qt
from aidia import utils
from aidia import aidia_logger
from aidia import HOME_DIR, CLS, DET, SEG
from aidia.ai.config import AIConfig
from aidia.ai.dataset import Dataset
from aidia.ai.det import DetectionModel
from aidia.ai.models.yolov4.yolov4_utils import postprocess_boxes, nms
from aidia.ai.models.yolov4.yolov4_config import YOLO_Config
from aidia.ai.seg import SegmentationModel
from aidia.widgets import ImageWidget
from aidia.widgets import CopyDataDialog
from aidia import image

tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class AIEvalDialog(QtWidgets.QDialog):

    aiRunning = QtCore.Signal(bool)

    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint
                            | QtCore.Qt.WindowCloseButtonHint
                            | QtCore.Qt.WindowMaximizeButtonHint
                            )
        self.setWindowTitle(self.tr("AI Evaluation"))

        self.setMinimumSize(QtCore.QSize(1200, 800))

        self._layout = QtWidgets.QGridLayout()

        self._dataset_layout = QtWidgets.QVBoxLayout()
        self._dataset_widget = QtWidgets.QWidget()
        self._results_layout = QtWidgets.QVBoxLayout()
        self._results_widget = QtWidgets.QWidget()
        self._images_layout = QtWidgets.QVBoxLayout()
        self._images_widget = QtWidgets.QWidget()

        self.dataset_dir = ""
        self.target_name = ""
        self.log_dir = ""
        self.prev_dir = ""
        self.weights_path = ""
        # self.class_names = []
        self.task = None

        self.fig, self.ax_pie = plt.subplots(figsize=(6, 6))
        self.ax_pie.axis("off")

        self.error_flags = {}
        self.input_fields = []
        self.left_row = 0
        self.right_row = 0

        # name
        self.tag_name = QtWidgets.QLabel(self.tr("Name"))
        self.input_name = QtWidgets.QComboBox()
        self.input_name.setMinimumWidth(200)
        self.input_name.currentTextChanged.connect(self.input_name_changed)
        self._add_basic_params(self.tag_name, self.input_name)

        # select weights box
        self.tag_weights = QtWidgets.QLabel(self.tr("Select Weights"))
        self.input_weights = QtWidgets.QComboBox()
        self.input_weights.setToolTip(self.tr(
            '''Select weights.
If you selected "Save Only the Best Weights", you can only set "best_weights.h5" got the best validation loss.
If you did not select it, you can set "last_model.h5" or "****.h5" saved at the **** epochs.'''
        ))
        self.input_weights.setMinimumWidth(200)
        def _validate(text):
            self.weights_path = os.path.join(self.log_dir, "weights", text)
        self.input_weights.currentTextChanged.connect(_validate)
        self._add_basic_params(self.tag_weights, self.input_weights)
        self.input_weights.setEnabled(False)

        ### add result fields ###
        title_result = qt.head_text(self.tr("Results"))
        title_result.setMaximumHeight(100)
        title_result.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter)
        self._results_layout.addWidget(title_result)

        self.text_results = QtWidgets.QLabel()
        self._results_layout.addWidget(self.text_results)

        ### add buttons ###
        # evaluate button
        self.button_eval = QtWidgets.QPushButton(self.tr("Evaluate"))
        self.button_eval.setToolTip(self.tr(
            """Evaluate and Export the trained model based on weights you selected."""
        ))
        self.button_eval.setMinimumWidth(200)
        self.button_eval.clicked.connect(self.evaluate)
        row = max(self.left_row, self.right_row)
        self._layout.addWidget(self.button_eval, row, 1, 1, 1)

        # predict button
        self.button_pred = QtWidgets.QPushButton(self.tr("Predict Test"))
        self.button_pred.setToolTip(self.tr(
            """Predict images in the directory you selected."""
        ))
        self.button_pred.clicked.connect(self.predict_unknown)
        self._layout.addWidget(self.button_pred, row, 2, 1, 1)

        # export data button
        self.button_open_logdir = QtWidgets.QPushButton(self.tr("Open Result Directory"))
        self.button_open_logdir.setToolTip(self.tr(
            '''Open the experiment result directory.'''
        ))
        self.button_open_logdir.clicked.connect(self.open_log_dir)
        self._layout.addWidget(self.button_open_logdir, row, 3, 1, 1)

        # export model button
        self.button_export_model = QtWidgets.QPushButton(self.tr("Save Model"))
        self.button_export_model.setToolTip(self.tr(
            """Save the model data."""
        ))
        self.button_export_model.clicked.connect(self.export_model)
        self._layout.addWidget(self.button_export_model, row, 4, 1, 1)
        row += 1

        # figure area
        _bg = np.zeros((200, 300, 3), np.uint8)
        _bg.fill(255)
        self.image_widget = ImageWidget(self, _bg)
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
        self.text_status.setMaximumHeight(50)
        self._layout.addWidget(self.text_status, row, 1, 1, 4)

        ### add dataset information ###
        # title
        title_dataset = qt.head_text(self.tr("Dataset Information"))
        title_dataset.setMaximumHeight(100)
        title_dataset.setAlignment(QtCore.Qt.AlignTop)
        self._dataset_layout.addWidget(title_dataset)

        # dataset information
        self.text_dataset = QtWidgets.QLabel()
        self.text_dataset.setAlignment(QtCore.Qt.AlignLeading)
        self._dataset_layout.addWidget(self.text_dataset)

        self.image_widget2 = ImageWidget(self, self._plt2img2())
        self._dataset_layout.addWidget(self.image_widget2)

        ### add preditcs images ###
        self.iw1 = ImageWidget(self)
        self.iw2 = ImageWidget(self)
        self.iw3 = ImageWidget(self)
        self.iw4 = ImageWidget(self)
        self.iw5 = ImageWidget(self)

        self._images_layout.addWidget(self.iw1)
        self._images_layout.addWidget(self.iw2)
        self._images_layout.addWidget(self.iw3)
        self._images_layout.addWidget(self.iw4)
        self._images_layout.addWidget(self.iw5)

        ### set layouts ###
        self._results_widget.setLayout(self._results_layout)
        self._dataset_widget.setLayout(self._dataset_layout)
        self._images_widget.setLayout(self._images_layout)

        self._layout.addWidget(self._results_widget, 0, 3, self.left_row, 2)
        self._layout.addWidget(self._dataset_widget, 0, 0, row + 1, 1)
        self._layout.addWidget(self._images_widget, 0, 5, row + 1, 1)

        self.setLayout(self._layout)

        # connect AI evaluation thread
        self.ai = AIEvalThread(self)
        self.ai.notifyMessage.connect(self.update_status)
        self.ai.datasetInfo.connect(self.update_dataset)
        self.ai.resultsList.connect(self.update_results)
        self.ai.predictList.connect(self.update_images)
        self.ai.progressValue.connect(self.update_progress)
        self.ai.finished.connect(self.ai_finished)

        # connect AI prediction thread
        self.ai_pred = AIPredThread(self)
        self.ai_pred.notifyMessage.connect(self.update_status)
        self.ai_pred.progressValue.connect(self.update_progress)
        self.ai_pred.finished.connect(self.ai_pred_finished)

        self.text_status.setText(self.tr("Ready"))
    
    def popup(self, dirpath):
        self.dataset_dir = dirpath
        self.setWindowTitle(self.tr("AI Evaluation - {}").format(dirpath))

        # if not self.ai.isRunning():
        #     self.reset_state()

        # pickup log directories
        data_dir = os.path.join(dirpath, "data")
        if os.path.exists(data_dir):
            targets = []
            for name in os.listdir(data_dir):
                logdir = os.path.join(self.dataset_dir, "data", name)
                if self._check_logdir(logdir):
                    targets.append(name)
            if len(targets):
                self.input_name.addItems(targets)
            else:
                self.disable_all()
        else:
            self.disable_all()

        self.exec_()

    def ai_finished(self):
        """Call back function when AI thread finished."""
        self.enable_all()
        self.aiRunning.emit(False)

    def ai_pred_finished(self):
        self.enable_all()
        self.aiRunning.emit(False)

        # open prediction result directory
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._predicted_dir))

    def disable_all(self):
        for x in self.input_fields:
            x.setEnabled(False)
        self.button_eval.setEnabled(False)
        self.button_pred.setEnabled(False)
        self.button_open_logdir.setEnabled(False)
        self.button_export_model.setEnabled(False)
    
    def enable_all(self):
        for x in self.input_fields:
            x.setEnabled(True)
        self.button_eval.setEnabled(True)
        self.button_open_logdir.setEnabled(True)
        self._switch_enables()

    def closeEvent(self, event):
        self.input_name.clear()
        
    def showEvent(self, event):
        if self.ai.isRunning():
            self.disable_all()

    def _plt2img2(self):
        self.fig.canvas.draw()
        data = self.fig.canvas.tostring_rgb()
        w, h = self.fig.canvas.get_width_height()
        c = len(data) // (w * h)
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
    
    def get_log_dirpath(self, name):
        return os.path.join(self.dataset_dir, "data", name)
    
    def get_weights_dirpath(self):
        return os.path.join(self.log_dir, "weights")
    
    ### Callback Functions ###
    def input_name_changed(self, name):
        logdir = self.get_log_dirpath(name)
        if self._check_logdir(logdir):
            self._set_ok(self.tag_name)
            self.enable_all()

            # activate select weights
            _wlist = os.path.join(self.get_weights_dirpath(), "*.h5")
            _wpath = sorted(glob.glob(_wlist), reverse=True)
            _wname = [os.path.basename(w) for w in _wpath]
            self.input_weights.clear()
            if len(_wname):
                self.input_weights.addItems(_wname)
            else:
                self.input_weights.setEnabled(False)
        else:
            self._set_error(self.tag_name)
            # self.disable_all()
    
    def update_images(self, images):
        self.iw1.loadPixmap(images[0])
        self.iw2.loadPixmap(images[1])
        self.iw3.loadPixmap(images[2])
        self.iw4.loadPixmap(images[3])
        self.iw5.loadPixmap(images[4])

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
        if value.get("num_train_subdir") is not None:
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
        self.ax_pie.set_title('Label Distribution', fontsize=20)
        self.ax_pie.pie(
            num_per_class,
            labels=class_names,
            # autopct="%1.1f%%",
            wedgeprops={'linewidth': 1, 'edgecolor':"white"},
            textprops={'color': "black", 'fontsize': 16})
        self.image_widget2.loadPixmap(self._plt2img2())

        # write dataset information
        with open(os.path.join(self.log_dir, "dataset_info.txt"), mode="w", encoding="utf-8") as f:
            f.write(text)
    

    def update_progress(self, value):
        self.progress.setValue(value)


    def update_status(self, value):
        self.text_status.setText(str(value))


    def update_results(self, value:dict):
        text = ""
        for k, v in value.items():
            if k == "img":
                continue
            text += f"{k}: {v:.6f}\n"
        self.text_results.setText(text)
        
        # if self.task == DET:
        #     save_dict = {
        #         "Metrics": list(value.keys()),
        #         "Values": list(value.values()),
        #     }
        #     utils.save_dict_to_excel(save_dict, os.path.join(self.log_dir, "eval.xlsx"))

        # get confusion matrix
        if self.task == SEG:
            img = value.pop("img")
            self.image_widget.loadPixmap(img)
        

    def _add_basic_params(self, tag:QtWidgets.QLabel, widget, right=False, reverse=False, custom_size=None):
        # add dict [tag:flag]
        self.error_flags[tag.text()] = 0

        self.input_fields.append(widget)
        row = self.left_row
        pos = [1, 2]
        align = [QtCore.Qt.AlignRight, QtCore.Qt.AlignLeft]
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
        self._layout.addWidget(tag, row, pos[0], h, w, alignment=align[0])
        self._layout.addWidget(widget, row, pos[1], h, w, alignment=align[1])
        if right:
            self.right_row += h
        else:
            self.left_row += h

    def _set_error(self, tag:QtWidgets.QLabel):
        tag.setStyleSheet("QLabel{ color: red; }")
        self.error_flags[tag.text()] = 1
    
    def _set_ok(self, tag:QtWidgets.QLabel):
        tag.setStyleSheet("QLabel{ color: black; }")
        self.error_flags[tag.text()] = 0
                
    def reset_state(self):
        # self.input_class.clear()
        self.general_results = []
        self.results = []
        self.metrics_dict = {}
        self.progress.setValue(0)
        self.text_dataset.clear()
        self.text_results.clear()
        self.image_widget.clear()
        self.image_widget2.clear()
        self.iw1.clear()
        self.iw2.clear()
        self.iw3.clear()
        self.iw4.clear()
        self.iw5.clear()

    def evaluate(self):
        if not os.path.exists(self.log_dir):
            self.parent().error_message(self.tr(
                '''The directory was not found.'''
            ))
            return

        error = sum(self.error_flags.values())
        if error > 0:
            self.text_status.setText(self.tr("Please check parameters."))
            return
        
        config_path = os.path.join(self.log_dir, "config.json")
        if not os.path.exists(config_path):
            self.text_status.setText(self.tr("Config file was not found."))
            return
        
        config = AIConfig(dataset_dir=self.dataset_dir)
        config.load(config_path)

        dataset_path = os.path.join(self.log_dir, "dataset.json")
        if not os.path.exists(dataset_path):
            self.text_status.setText(self.tr("Dataset file was not found."))
            return
        
        self.task = config.TASK

        # for update new log dir
        config.log_dir = self.log_dir

        self.disable_all()
        self.reset_state()

        self.ai.set_config(config, self.weights_path)
        self.ai.start()
        self.aiRunning.emit(True)
    
    def predict_unknown(self):
        if not os.path.exists(self.log_dir):
            self.parent().error_message(self.tr(
                '''The directory was not found.'''
            ))
            return

        error = sum(self.error_flags.values())
        if error > 0:
            self.text_status.setText(self.tr("Please check parameters."))
            return
        
        # load config
        config_path = os.path.join(self.log_dir, "config.json")
        if not os.path.exists(config_path):
            self.text_status.setText(self.tr("Config file was not found."))
            return
        
        config = AIConfig(self.dataset_dir)
        config.load(config_path)

        if config.TASK not in [SEG, DET]:
            self.text_status.setText(self.tr("Not implemented function."))
            return
        
        # check onnx model
        onnx_path = os.path.join(self.log_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            self.text_status.setText(self.tr("The ONNX model was not found."))
            return
        
        # target data directory
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
        self._predicted_dir = os.path.join(self.log_dir, 'predict_images', utils.get_basename(target_path))
        
        if not len(os.listdir(target_path)):
            self.text_status.setText(self.tr("The Directory is empty."))
            return

        # AI run
        self.text_status.setText(self.tr("Processing..."))

        self.task = config.TASK
        self.prev_dir = target_path
        self.disable_all()
        self.progress.setValue(0)
        # self.reset_state()

        # for update new log dir
        config.log_dir = self.log_dir

        self.ai_pred.set_params(config, target_path, onnx_path)
        self.ai_pred.start()
        self.aiRunning.emit(True)
    
    def open_log_dir(self):
        if os.path.exists(self.log_dir):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.log_dir))
        else:
            self.parent().error_message(self.tr(
                '''The directory was not found.'''
            ))
        # opendir = HOME_DIR
        # target_path = str(QtWidgets.QFileDialog.getExistingDirectory(
        #     self,
        #     self.tr("Select Output Directory"),
        #     opendir,
        #     QtWidgets.QFileDialog.ShowDirsOnly |
        #     QtWidgets.QFileDialog.DontResolveSymlinks))
        # if not target_path:
        #     return None
        # target_path = target_path.replace('/', os.sep)

        # cd = CopyDataDialog(self, self.log_dir, target_path)
        # cd.popup()

        # self.text_status.setText(self.tr("Export data to {}").format(target_path))

    def export_model(self):
        if not os.path.exists(self.log_dir):
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

        cd = CopyDataDialog(self, self.log_dir, target_path, only_model=True)
        cd.popup()

        self.text_status.setText(self.tr("Export data to {}").format(target_path))

    
    def _switch_enables(self):
        onnx_path = os.path.join(self.log_dir, "model.onnx")
        if os.path.exists(onnx_path):
            self.button_export_model.setEnabled(True)
            self.button_pred.setEnabled(True)
        else:
            self.button_export_model.setEnabled(False)
            self.button_pred.setEnabled(False)
        
        # evaldir_path = os.path.join(self.log_dir, 'evaluation')
        # if os.path.exists(evaldir_path):
        #     self.button_open_logdir.setEnabled(True)
        # else:
        #     self.button_open_logdir.setEnabled(False)

    def _check_logdir(self, logdir):
        config_path = os.path.join(logdir, "config.json")
        dataset_path = os.path.join(logdir, "dataset.json")
        weights_path = os.path.join(logdir, "weights")
        onnx_path = os.path.join(logdir, 'model.onnx')
        if os.path.exists(config_path) and os.path.exists(dataset_path):
            if os.path.exists(weights_path) and len(os.listdir(weights_path)):
                self.log_dir = logdir
                return True
            elif os.path.exists(onnx_path):
                self.log_dir = logdir
                return True
            else:
                self.log_dir = ''
                return False
        else:
            self.log_dir = ''
            return False

    
class AIEvalThread(QtCore.QThread):

    resultsList = QtCore.Signal(dict)
    # resultsList = QtCore.Signal(list)
    notifyMessage = QtCore.Signal(str)
    datasetInfo = QtCore.Signal(dict)
    predictList = QtCore.Signal(list)
    progressValue = QtCore.Signal(int)

    def __init__(self, parent):
        super().__init__(parent)

    def set_config(self, config:AIConfig, weights_path):
        self.config = config
        self.weights_path = weights_path

    def run(self):
        model = None
        if self.config.TASK == CLS:
            raise NotImplementedError
        elif self.config.TASK == DET:
            model = DetectionModel(self.config)
        if self.config.TASK == SEG:
            model = SegmentationModel(self.config)
        
        if model is None:
            self.notifyMessage.emit(self.tr("Model error. Terminated."))
            return
        
        self.notifyMessage.emit(self.tr("Data loading..."))
        try:
            model.load_dataset()
        except Exception as e:
            self.notifyMessage.emit(self.tr("Failed to load dataset."))
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
        try:
            model.build_model(mode="test", weights_path=self.weights_path)
        except Exception as e:
            self.notifyMessage.emit(self.tr("Failed to build the model."))
            aidia_logger.error(e, exc_info=True)
            return

        self.notifyMessage.emit(self.tr("Generate test result images..."))

        # save_dir = os.path.join(self.config.log_dir, "test_preds")
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        save_dir = utils.get_dirpath_with_mkdir(self.config.log_dir, 'evaluation', 'test_images')

        n = model.dataset.num_test
        predicts = []
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
            
            # continue if output already exists
            if os.path.exists(save_path):
                self.progressValue.emit(int(i / n * 100))
                # update 5 images to the widget
                if len(predicts) <= 5:
                    w = self.config.image_size[1]
                    result_img = image.imread(save_path)
                    if self.config.TASK in [SEG]:
                        result_img = result_img[:, w:w*2]
                    predicts.append(result_img)
                else:
                    self.predictList.emit(predicts)
                    # predicts = []
                    break
                continue

            try:
                result_img = model.predict_by_id(image_id)
            except FileNotFoundError as e:
                self.notifyMessage.emit(self.tr("Error: {} was not found.").format(img_path))
                return
            image.imwrite(result_img, save_path)
            # update latest 5 images to the widget
            if len(predicts) <= 5:
                w = self.config.image_size[1]
                if self.config.TASK in [SEG]:
                    result_img = result_img[:, w:w*2]
                predicts.append(result_img)
            else:
                self.predictList.emit(predicts)
                predicts = []
            # update progress bar
            self.progressValue.emit(int(i / n * 100))
        self.progressValue.emit(0)

        self.notifyMessage.emit(self.tr("Convert model to ONNX..."))
        model.convert2onnx()
    
        self.notifyMessage.emit(self.tr("Evaluating..."))
        try:
            results = model.evaluate(cb_widget=self)
        except Exception as e:
            self.notifyMessage.emit(self.tr("Failed to evaluate."))
            aidia_logger.error(e, exc_info=True)
            return
        self.resultsList.emit(results)

        self.notifyMessage.emit(self.tr("Done"))


class AIPredThread(QtCore.QThread):

    notifyMessage = QtCore.Signal(str)
    progressValue = QtCore.Signal(int)

    def __init__(self, parent):
        super().__init__(parent)

    def set_params(self, config:AIConfig, target_path, onnx_path):
        self.config = config
        self.target_path = target_path
        self.onnx_path = onnx_path
    
    def run(self):
        # savedir = os.path.join(self.target_path, "AI_results")
        # if not os.path.exists(savedir):
        #     os.mkdir(savedir)
        savedir = utils.get_dirpath_with_mkdir(self.config.log_dir, 'predict_images', utils.get_basename(self.target_path))

        n = len(os.listdir(self.target_path))
        model = InferenceSession(self.onnx_path)

        for i, file_path in enumerate(glob.glob(os.path.join(self.target_path, "*"))):
            if utils.extract_ext(file_path) == ".json":
                continue
            try:
                img = image.read_image(file_path)
            except Exception as e:
                continue

            if img is None:
                continue
            
            self.notifyMessage.emit(f"{i} / {n} - {file_path}")
            name = utils.get_basename(file_path)
            
            if self.config.TASK == SEG:
                img = cv2.resize(img, self.config.image_size)
                inputs = image.preprocessing(img, is_tensor=True)
                input_name = model.get_inputs()[0].name
                result = model.run([], {input_name: inputs})[0][0]
                result_img = image.mask2merge(img, result, self.config.LABELS)
                save_path = os.path.join(savedir, f"{name}.png")
                image.imwrite(result_img, save_path)

            elif self.config.TASK == DET:
                inputs = cv2.resize(img, self.config.image_size)
                inputs = image.preprocessing(inputs, is_tensor=True)
                input_name = model.get_inputs()[0].name
                result = model.run([], {input_name: inputs})

                # post processing
                if self.config.MODEL.find("YOLO") > -1:
                    bboxes = self.yolo_postprocessing(img, result)
                    bbox_dict_pred = []
                    for bbox_pred in bboxes:
                        bbox = list(map(float, bbox_pred[:4]))
                        score = bbox_pred[4]
                        class_id = int(bbox_pred[5])
                        class_name = self.config.LABELS[class_id]
                        score = '%.4f' % score
                        bbox_dict_pred.append({"class_id": class_id,
                                                "class_name": class_name,
                                                "confidence": score,
                                                "bbox": bbox})
                    bbox_dict_pred.sort(key=lambda x:float(x['confidence']), reverse=True)

                    result_img = image.det2merge(img, bbox_dict_pred)
                    save_path = os.path.join(savedir, f"{name}.png")
                    image.imwrite(result_img, save_path)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            
            self.progressValue.emit(int(i / n * 100))

        self.progressValue.emit(0)
        self.notifyMessage.emit(self.tr("Saved result images to {}").format(savedir))
    
    def yolo_postprocessing(self, img, result):
        is_tiny = True if self.config.MODEL.split("-")[-1] == "tiny" else False
        if is_tiny:
            _, pred_mbbox, _, pred_lbbox = result
        else:
            _, pred_sbbox, _, pred_mbbox, _, pred_lbbox = result
    
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + 3)),
                                    np.reshape(pred_mbbox, (-1, 5 + 3)),
                                    np.reshape(pred_lbbox, (-1, 5 + 3))], axis=0)
        bboxes = postprocess_boxes(pred_bbox, img.shape[:2], self.config.INPUT_SIZE, YOLO_Config().SCORE_THRESHOLD)
        bboxes = nms(bboxes, YOLO_Config().IOU_THRESHOLD)
        return bboxes