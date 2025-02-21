from qtpy import QtCore
from qtpy import QtWidgets

from aidia import qt
from aidia import S_EPSILON, S_AREA_LIMIT, CLEAR, ERROR


class LabelEditDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint)
        
        self.setWindowTitle(self.tr('Label Edit'))
        
        self.label = ''
        
        layout = QtWidgets.QVBoxLayout()
        _widget_layout = QtWidgets.QVBoxLayout()

        # label edit
        self.label_edit = QtWidgets.QLabel(self.tr('''Please split labels by "_" if multi label
e.g. apple_banana_orange'''))
        self.label_edit_box = QtWidgets.QLineEdit()
        self.label_edit_box.setAlignment(QtCore.Qt.AlignCenter)

        _widget_layout.addWidget(self.label_edit)
        _widget_layout.addWidget(self.label_edit_box)
        _widget = QtWidgets.QWidget()
        _widget.setLayout(_widget_layout)
        layout.addWidget(_widget)

        # accept and reject button
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        bb.button(bb.Ok).setIcon(qt.newIcon('done'))
        bb.button(bb.Cancel).setIcon(qt.newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)

    def validate(self):
        label = self.label_edit_box.text().strip('_')
        self.label = label
        self.accept()

    def popup(self, current_label=''):
        self.label_edit_box.setText(current_label)

        if self.exec_():
            return self.label
        else:
            return None
