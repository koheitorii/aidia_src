import os
import shutil
from glob import glob

from qtpy import QtCore, QtWidgets, QtGui

from aidia import utils

class CopyAnnotationsDialog(QtWidgets.QDialog):
    def __init__(self, parent, work_dir, target_dir, is_submode) -> None:
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint
                            )
        self.setWindowTitle(self.tr("Copying Annotation Files..."))
        self.setMinimumSize(QtCore.QSize(500, 100))

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel(self)

        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setMaximum(100)
        self.progress.setValue(0)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)

        self.setLayout(layout)

        self.th = _thread(self, work_dir, target_dir, is_submode)
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

    def __init__(self, parent, work_dir, target_dir, is_submode) -> None:
        super().__init__(parent)

        self.work_dir = work_dir
        self.target_dir = target_dir
        self.is_submode = is_submode

    def run(self):
        anno_list = []
        if self.is_submode:
            src_dir = utils.get_parent_path(self.work_dir)
            anno_list = glob(os.path.join(src_dir, "*", "*.json"))
            if not len(anno_list):
                return
            for i, p in enumerate(anno_list):
                filename = os.path.basename(p)
                dirname = os.path.basename(src_dir)
                sub_dirname = utils.get_basedir(p)
                new_path = os.path.join(self.target_dir, dirname, sub_dirname, filename)
                self.pathValue.emit([p, new_path])
                if not os.path.exists(os.path.dirname(new_path)):
                    os.makedirs(os.path.dirname(new_path))
                shutil.copy(p, new_path)
                n = int((i + 1) / len(anno_list) * 100)
                self.progressValue.emit(n)
        else:
            src_dir = self.work_dir
            anno_list = glob(os.path.join(src_dir, "*.json"))
            if not len(anno_list):
                return
            for i, p in enumerate(anno_list):
                filename = os.path.basename(p)
                dirname = os.path.basename(src_dir)
                new_path = os.path.join(self.target_dir, dirname, filename)
                self.pathValue.emit([p, new_path])
                if not os.path.exists(os.path.dirname(new_path)):
                    os.makedirs(os.path.dirname(new_path))
                shutil.copy(p, new_path)
                n = int((i + 1) / len(anno_list) * 100)
                self.progressValue.emit(n)
            