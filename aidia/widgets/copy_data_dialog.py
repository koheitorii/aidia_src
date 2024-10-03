import os
import shutil
from glob import glob

from qtpy import QtCore, QtWidgets, QtGui

from aidia import utils

class CopyDataDialog(QtWidgets.QDialog):
    def __init__(self, parent, src, dst, only_model=False) -> None:
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint
                            )
        self.setWindowTitle(self.tr("Copying Files..."))
        self.setMinimumSize(QtCore.QSize(500, 100))

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel(self)

        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setMaximum(100)
        self.progress.setValue(0)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)

        self.setLayout(layout)

        self.th = _thread(self, src, dst, only_model)
        self.th.progressValue.connect(self.update_progress)
        self.th.pathValue.connect(self.update_label)
        self.th.finished.connect(self.accept)

    def update_progress(self, value):
        self.progress.setValue(value)
    
    def update_label(self, value):
        self.label.setText(
            self.tr("{}\n-> {}").format(value[0], value[1])
            )
    
    def popup(self):
        self.th.start()
        self.exec_()


class _thread(QtCore.QThread):
    progressValue = QtCore.Signal(int)
    pathValue = QtCore.Signal(list)

    def __init__(self, parent, src, dst, only_model) -> None:
        super().__init__(parent)

        self.src_dir = src
        self.dst_dir = dst
        self.only_model = only_model

    def run(self):
        if self.only_model:
            config_path = os.path.join(self.src_dir, "config.json")
            onnx_path = os.path.join(self.src_dir, "model.onnx")
            if not os.path.exists(config_path) and not os.path.exists(onnx_path):
                return
            for p in [config_path, onnx_path]:
                filename = os.path.basename(p)
                dirname = os.path.basename(self.src_dir)
                new_path = os.path.join(self.dst_dir, dirname, filename)
                self.pathValue.emit([p, new_path])
                if not os.path.exists(os.path.dirname(new_path)):
                    os.makedirs(os.path.dirname(new_path))
                shutil.copy(p, new_path)
            return

        file_list = glob(os.path.join(self.src_dir, "*"), recursive=True)
        if not len(file_list):
            return
        for i, p in enumerate(file_list):
            if os.path.isfile(p):
                filename = os.path.basename(p)
                dirname = os.path.basename(self.src_dir)
                new_path = os.path.join(self.dst_dir, dirname, filename)
                self.pathValue.emit([p, new_path])
                if not os.path.exists(os.path.dirname(new_path)):
                    os.makedirs(os.path.dirname(new_path))
                shutil.copy(p, new_path)
            elif os.path.isdir(p):
                filename = os.path.basename(p)
                dirname = os.path.basename(self.src_dir)
                new_path = os.path.join(self.dst_dir, dirname, filename)
                self.pathValue.emit([p, new_path])
                if not os.path.exists(os.path.dirname(new_path)):
                    os.makedirs(os.path.dirname(new_path))
                shutil.copytree(p, new_path, dirs_exist_ok=True)
            else:
                continue

            n = int((i + 1) / len(file_list) * 100)
            self.progressValue.emit(n)

    