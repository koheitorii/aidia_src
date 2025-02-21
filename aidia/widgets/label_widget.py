import functools
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from aidia.widgets.label_setting_dialog import LabelSettingDialog


class LabelWidget(QtWidgets.QWidget):

    valueChanged = QtCore.Signal()

    def __init__(self, parent=None, label_def=None, is_multi_label=True):

        super().__init__(parent)

        self.labelSettingDialog = LabelSettingDialog(self)

        self.label_def = []
        if label_def:
            self.label_def = label_def

        self.is_multi_label = is_multi_label

        self.button_size = 80

        self.is_active = False
        self.label_list = []
        self.default_label_list = []
        self.label = ""
        self.default_label = ""
        self.prev_button = None
        self.prev_button2 = None
        self.b_dict = {}
        self.b2_dict = {}

        self._layout = QtWidgets.QGridLayout()

        # label text
        t = QtWidgets.QLabel(self.tr("Selected Shape Labels"))
        self._layout.addWidget(t, 0, 0, QtCore.Qt.AlignRight)
        t = QtWidgets.QLabel(self.tr("Default Labels"))
        self._layout.addWidget(t, 1, 0, QtCore.Qt.AlignRight)

        self.bb_layout = QtWidgets.QHBoxLayout()
        self.bb2_layout = QtWidgets.QHBoxLayout()
        self.create_label_buttons()

        # is multi label
        self.is_multi_label_text = QtWidgets.QLabel(self.tr("Multi Label Mode"))
        self.is_multi_label_checkbox = QtWidgets.QCheckBox()
        def _validate(state):  # check:2, empty:0
            if state == 2:
                self.is_multi_label = True
                self._reset_at_is_multi_label_changed()
            else:
                self.is_multi_label = False
                self._reset_at_is_multi_label_changed()
        self.is_multi_label_checkbox.stateChanged.connect(_validate)
        if self.is_multi_label:
            self.is_multi_label_checkbox.setChecked(True)
        else:
            self.is_multi_label_checkbox.setChecked(False)
        l = QtWidgets.QHBoxLayout()
        l.addWidget(self.is_multi_label_checkbox, alignment=QtCore.Qt.AlignRight)
        l.addWidget(self.is_multi_label_text, alignment=QtCore.Qt.AlignLeft)
        w = QtWidgets.QWidget()
        w.setLayout(l)
        self._layout.addWidget(w, 0, 2, QtCore.Qt.AlignCenter)
        
        # popup label setting dialog
        self.popup_setting_button = QtWidgets.QPushButton(self.tr("Label Buttons Setting"), self)
        self.popup_setting_button.clicked.connect(self._setting_popup)
        self._layout.addWidget(self.popup_setting_button, 1, 2, QtCore.Qt.AlignCenter)

        # concatenate layout
        self.setLayout(self._layout)

        # self.setEnabled(False)
        self.disable_buttons()
    

    def _setting_popup(self):
        ret = self.labelSettingDialog.popUp(self.label_def)
        if ret:
            self.label_def = ret
            self.create_label_buttons()
            if self.is_active:
                self.enable_buttons()
            else:
                self.disable_buttons()


    def _reset_at_is_multi_label_changed(self):
        self.prev_button = None
        self.prev_button2 = None
        self.label_list.clear()
        self.label = ""
        self.reset_button()
        self.default_label_list.clear()
        self.default_label = ""
        self.reset_button2()
        if self.is_active:
            self.enable_buttons()
        else:
            self.disable_buttons()


    def enable_buttons(self):
        for b in self.b_dict.values():
            b.setEnabled(True)
        self.is_active = True


    def disable_buttons(self):
        for b in self.b_dict.values():
            b.setEnabled(False)
        self.is_active = False


    def create_label_buttons(self):
        # clear buttons
        if self.bb_layout.count() > 0:
            for i in range(self.bb_layout.count()):
                item = self.bb_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
        if self.bb2_layout.count() > 0:
            for i in range(self.bb2_layout.count()):
                item = self.bb2_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
        self.prev_button = None
        self.prev_button2 = None

        # add buttons
        self.b_dict = {}
        self.b2_dict = {}
        bb = QtWidgets.QDialogButtonBox(QtCore.Qt.Horizontal)
        bb.clicked.connect(self.toggle_button)
        bb2 = QtWidgets.QDialogButtonBox(QtCore.Qt.Horizontal)
        bb2.clicked.connect(self.toggle_button2)

        for i, l in enumerate(self.label_def):
            shortcut_key = "{}".format((i + 1) % 10)
            b = self.create_button(l, self.button_size, shortcut_key=shortcut_key)
            b.clicked.connect(functools.partial(self.set_label, l))
            self.b_dict[l] = b
            bb.addButton(b, QtWidgets.QDialogButtonBox.ActionRole)
        
        for l in self.label_def:
            b = self.create_button(l, self.button_size)
            b.clicked.connect(functools.partial(self.set_default_label, l))
            self.b2_dict[l] = b
            bb2.addButton(b, QtWidgets.QDialogButtonBox.ActionRole)

        self.bb_layout.addWidget(bb, alignment=QtCore.Qt.AlignLeft)
        self.bb2_layout.addWidget(bb2, alignment=QtCore.Qt.AlignLeft)

        # concat layout
        bb_widget = QtWidgets.QWidget()
        bb2_widget = QtWidgets.QWidget()
        bb_widget.setLayout(self.bb_layout)
        bb2_widget.setLayout(self.bb2_layout)
        self._layout.addWidget(bb_widget, 0, 1, alignment=QtCore.Qt.AlignLeft)
        self._layout.addWidget(bb2_widget, 1, 1, alignment=QtCore.Qt.AlignLeft)


    def update(self, label=None):
        if label is None:
            # self.setEnabled(False)
            self.disable_buttons()
        else:
            # self.setEnabled(True)
            self.enable_buttons()
            if len(label) == 0:
                self.label_list = []
                self.validate()
                return
            else:
                self.label_list = label.split("_")

            # update button states by label
            if self.is_multi_label:
                for l in self.label_list:
                    if l in self.label_def:
                        self.b_dict[l].setChecked(True)
            else:
                l = self.label_list[0]
                if l in self.label_def:
                    self.b_dict[l].setChecked(True)
                    self.prev_button = self.b_dict[l]
                
            self.validate()
                        
    
    def validate(self):
        if len(self.label_list):
            self.label = self.label_list[0]
            if len(self.label_list) > 1:
                for l in self.label_list[1:]:
                    self.label += f"_{l}"
        else:
            self.label = ""
    

    def reset(self):
        self.label_list.clear()
        self.label = ""
        self.reset_button()
        # self.setEnabled(False)
        self.disable_buttons()


    def reset_button(self):
        for v in self.b_dict.values():
            v.setChecked(False)


    def reset_button2(self):
        for v in self.b2_dict.values():
            v.setChecked(False)
    

    def toggle_button(self, button):
        if not self.is_multi_label:
            if self.prev_button is None:
                self.prev_button = button
            elif self.prev_button == button:
                pass
            else:
                self.prev_button.setChecked(False)
                self.prev_button = button


    def toggle_button2(self, button):
        if not self.is_multi_label:
            if self.prev_button2 is None:
                self.prev_button2 = button
            elif self.prev_button2 == button:
                pass
            else:
                self.prev_button2.setChecked(False)
                self.prev_button2 = button


    def create_button(self, button_label, size, color=None, shortcut_key=None):
        button = QtWidgets.QPushButton('&{}'.format(button_label))
        if shortcut_key:
            button.setShortcut(shortcut_key)
        button.setCheckable(True)
        button.setAutoDefault(False)
        # button.setMaximumWidth(size)
        # button.setMinimumWidth(size)
        if color:
            pal = QtGui.QPalette()
            pal.setColor(QtGui.QPalette.Button, QtGui.QColor(color))
            # pal.setColor(QtGui.QPalette.Window, color)
            button.setAutoFillBackground(True)
            button.setPalette(pal)
        return button


    def set_label(self, label):
        if self.is_multi_label:
            if label in self.label_list:
                self.label_list.remove(label)
            else:
                self.label_list.append(label)
        else:
            self.label_list = [label]
        self.validate()
        self.valueChanged.emit()
        


    def set_default_label(self, label):
        if self.is_multi_label:
            if label in self.default_label_list:
                self.default_label_list.remove(label)
            else:
                self.default_label_list.append(label)
            self.default_label = "_".join(self.default_label_list)
        else:
            if len(self.default_label) and label == self.default_label:
                self.default_label = ""
            else:
                self.default_label = label