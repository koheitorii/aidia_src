from qtpy import QtCore
from qtpy import QtWidgets

from aidia import qt
from aidia.qt import hline, head_text
from aidia import S_EPSILON, S_AREA_LIMIT, CLEAR, ERROR


class SettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint)

        layout = QtWidgets.QVBoxLayout()
        general_layout = QtWidgets.QGridLayout()

        self.approx_epsilon = None
        self.area_limit = None

        self.error = {}

        # approx epsilon definition
        self.approx_epsilon_text = QtWidgets.QLabel(self.tr("Approximation Accuracy"))
        self.approx_epsilon_input = QtWidgets.QLineEdit()
        self.approx_epsilon_input.setAlignment(QtCore.Qt.AlignCenter)
        self.approx_epsilon_input.setToolTip(self.tr(
            """Set polygonal approximation accuracy."""
        ))
        def _validate(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 1.0:
                self.approx_epsilon = float(text)
                self.approx_epsilon_text.setStyleSheet("QLabel{ color: black; }")
                self.error[S_EPSILON] = CLEAR
            else:
                self.approx_epsilon = None
                self.approx_epsilon_text.setStyleSheet("QLabel{ color: red; }")
                self.error[S_EPSILON] = ERROR
        self.approx_epsilon_input.textChanged.connect(_validate)

        # area limit
        self.area_limit_text = QtWidgets.QLabel(self.tr("Area Limit"))
        self.area_limit_input = QtWidgets.QLineEdit()
        self.area_limit_input.setAlignment(QtCore.Qt.AlignCenter)
        self.area_limit_input.setToolTip(self.tr(
            """Delete shapes have the area under the set value.
If you set 0, all shapes will be generated."""
        ))
        def _validate(text):
            if text.isdigit() and 0 <= int(text):
                self.area_limit = int(text)
                self.area_limit_text.setStyleSheet("QLabel{ color: black; }")
                self.error[S_AREA_LIMIT] = CLEAR
            else:
                self.area_limit = None
                self.area_limit_text.setStyleSheet("QLabel{ color: red; }")
                self.error[S_AREA_LIMIT] = ERROR
        self.area_limit_input.textChanged.connect(_validate)

        # set general settings layout
        layout.addWidget(head_text(self.tr("AI Generation Settings")))
        general_layout.addWidget(self.approx_epsilon_text, 0, 0, QtCore.Qt.AlignRight)
        general_layout.addWidget(self.approx_epsilon_input, 0, 1, QtCore.Qt.AlignLeft)
        general_layout.addWidget(self.area_limit_text, 1, 0, QtCore.Qt.AlignRight)
        general_layout.addWidget(self.area_limit_input, 1, 1, QtCore.Qt.AlignLeft)
        general_widget = QtWidgets.QWidget()
        general_widget.setLayout(general_layout)
        layout.addWidget(general_widget)

        # accept and reject button
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        bb.button(bb.Ok).setIcon(qt.new_icon('done'))
        bb.button(bb.Cancel).setIcon(qt.new_icon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)


    def validate(self):
        if sum(self.error.values()) > 0:
            text = self.tr("Please check errors.")
            QtWidgets.QMessageBox.critical(
                self, self.tr("Error"),
                "<p>{}</p>".format(text))
            return
        else:
            self.accept()


    def popUp(self, params_dict=None):
        approx_epsilon = params_dict[S_EPSILON]
        area_limit = params_dict[S_AREA_LIMIT]

        self.approx_epsilon = approx_epsilon
        self.approx_epsilon_input.setText(str(approx_epsilon))

        self.area_limit = area_limit
        self.area_limit_input.setText(str(area_limit))

        if self.exec_():
            return {
                    S_EPSILON: self.approx_epsilon,
                    S_AREA_LIMIT: self.area_limit,
                }
        else:
            return False
