from math import sqrt
import os.path as osp

import numpy as np

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


def is_dark_mode():
    """Check if the current application is in dark mode."""
    palette = QtWidgets.QApplication.palette()
    window_color = palette.color(QtGui.QPalette.Window)
    return window_color.lightness() < 128


def get_default_color(is_qcolor=False):
    """Get the default color based on the current theme."""
    if is_dark_mode():
        if is_qcolor:
            return QtGui.QColor(255, 255, 255)
        else:
            return "white"  # For dark mode, return white as default color
    else:
        if is_qcolor:
            return QtGui.QColor(0, 0, 0)
        else:
            # For light mode, return black as default color
            # This is useful for text or other elements that should be black in light mode
            return "black"


def new_icon(icon):
    icons_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'icons')
    icon_path = osp.join(icons_dir, '%s.png' % icon)
    
    if is_dark_mode():
        # ダークモード用にアイコンの色を調整
        pixmap = QtGui.QPixmap(icon_path)
        if not pixmap.isNull():
            # アイコンを白色に変更
            white_pixmap = QtGui.QPixmap(pixmap.size())
            white_pixmap.fill(QtCore.Qt.GlobalColor.white)
            white_pixmap.setMask(pixmap.createMaskFromColor(QtCore.Qt.GlobalColor.transparent))
            return QtGui.QIcon(white_pixmap)
    
    return QtGui.QIcon(icon_path)


def new_button(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def new_action(parent, text, slot=None, shortcut=None, icon=None,
              tip=None, checkable=False, enabled=True, checked=False):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QtWidgets.QAction(text, parent)
    locale = QtCore.QLocale.system().name()
    if icon is not None:
        if locale == "ja_JP":
            a.setIconText(text)
            a.setText(text.replace("\n", ""))
        else:
            a.setIconText(text.replace(' ', '\n'))
            a.setText(text)
        if isinstance(icon, QtGui.QIcon.ThemeIcon):
            a.setIcon(QtGui.QIcon.fromTheme(icon))
        else:
            a.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    a.setChecked(checked)
    return a


def add_actions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def labelValidator():
    return QtGui.QRegExpValidator(QtCore.QRegExp(r'^[^ \t].+'), None)


class DictObject(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def distancetoline(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / (np.linalg.norm(p2 - p1) + 1e-12)


def fmtShortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)


def head_text(text):
    bold_text = QtWidgets.QLabel(text)
    bold_text.setStyleSheet("font-size: 15pt; font-weight: bold")
    return bold_text


def hline():
    hr_label = QtWidgets.QLabel()
    hr_label.setFrameStyle(QtWidgets.QFrame.HLine | QtWidgets.QFrame.Raised)
    hr_label.setLineWidth(2)
    return hr_label
