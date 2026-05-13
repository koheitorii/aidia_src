from math import sqrt
import os
import os.path as osp

import numpy as np
import cv2

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class DictObject(object):
    """A simple object that can be initialized with a dictionary."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def is_dark_mode():
    """Check if the current application is in dark mode."""
    palette = QtWidgets.QApplication.palette()
    window_color = palette.color(QtGui.QPalette.Window)
    return window_color.lightness() < 128

def new_icon(icon):
    """Create a new icon from the specified icon name or path."""
    icons_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'icons')
    icon_path = osp.join(icons_dir, '%s.png' % icon)
    
    if is_dark_mode():
        pixmap = QtGui.QPixmap(icon_path)
        if not pixmap.isNull():
            white_pixmap = QtGui.QPixmap(pixmap.size())
            white_pixmap.fill(QtCore.Qt.GlobalColor.white)
            white_pixmap.setMask(pixmap.createMaskFromColor(QtCore.Qt.GlobalColor.transparent))
            return QtGui.QIcon(white_pixmap)
    
    return QtGui.QIcon(icon_path)

def new_button(text, icon=None, slot=None):
    """Create a new button with an icon and optional slot."""
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
    """Add a list of actions to a widget."""
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)

def labelValidator():
    """Return a validator for label names."""
    return QtGui.QRegExpValidator(QtCore.QRegExp(r'^[^ \t].+'), None)

def distance(p):
    """Calculate the Euclidean distance from the origin to the point p."""
    return sqrt(p.x() * p.x() + p.y() * p.y())

def distancetoline(point, line):
    """Calculate the distance from a point to a line segment defined by two points."""
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
    """Format a keyboard shortcut for display."""
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)

def head_text(text):
    """Create a bold label for headings."""
    bold_text = QtWidgets.QLabel(text)
    bold_text.setStyleSheet("font-size: 15pt; font-weight: bold")
    return bold_text

def hline():
    """Create a horizontal line widget."""
    hr_label = QtWidgets.QLabel()
    hr_label.setFrameStyle(QtWidgets.QFrame.HLine | QtWidgets.QFrame.Raised)
    hr_label.setLineWidth(2)
    return hr_label

class LabelColor(object):
    """Styles for QLabel."""
    def __init__(self):
        pass
        
    @staticmethod
    def get_style(state):
        """Get the style for a given state."""
        if state == "default":
            return "QLabel{ color: white; }" if is_dark_mode() else "QLabel{ color: black; }"
        elif state == "error":
            return "QLabel{ color: red; }"
        elif state == "disabled":
            return "QLabel{ color: gray; }"
        else:
            raise ValueError("Invalid state: %s" % state)
    
    @staticmethod
    def get_qcolor(state):
        """Get the QColor for a given state."""
        if state == "default":
            return QtGui.QColor(255, 255, 255) if is_dark_mode() else QtGui.QColor(0, 0, 0)
        elif state == "error":
            return QtGui.QColor(255, 0, 0)
        elif state == "disabled":
            return QtGui.QColor(128, 128, 128)
        else:
            raise ValueError("Invalid state: %s" % state)
    
    @staticmethod
    def get_color(state):
        """Get the color for a given state."""
        if state == "default":
            return "white" if is_dark_mode() else "black"
        elif state == "error":
            return "red"
        elif state == "disabled":
            return "gray"
        else:
            raise ValueError("Invalid state: %s" % state)
        
class ImageWidget(QtWidgets.QWidget):
    """A widget for displaying images.
    
    Args:
        parent: The parent widget.
        image: The image to display (file path or numpy array).
        resize: The size to resize the image to (width, height).
        alpha: Whether to include the alpha channel when loading the image.
    """
    def __init__(self, parent, image=None, resize=None, alpha=False):
        super().__init__(parent=parent)

        from aidia.image import imread

        self._painter = QtGui.QPainter()

        self.pixmap = QtGui.QPixmap()
        if image is not None:
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                image = imread(image, alpha=alpha)
                if resize is not None:
                    image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
                self.loadPixmap(image)
            elif isinstance(image, np.ndarray):
                self.loadPixmap(image)
        self.show()
    
    def loadPixmap(self, image: np.ndarray):
        byte_per_line = image[0].nbytes
        h, w = image.shape[0:2]
        if image.shape[2] == 4:
            image = QtGui.QImage(image.flatten(), w, h, byte_per_line,
                            QtGui.QImage.Format.Format_RGBA8888)
        else:
            image = QtGui.QImage(image.flatten(), w, h, byte_per_line,
                            QtGui.QImage.Format.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(image)
        self.setMinimumHeight(10)
        self.setMinimumWidth(10)
        self.update()
    
    def clear(self):
        self.pixmap = QtGui.QPixmap()
        self.update()
    
    def paintEvent(self, event):
        p = self._painter
        p.begin(self)

        x = 0
        y = 0
        scale = 1.0
        if self.pixmap.isNull():
            # return super().paintEvent(event)
            self.pixmap.fill()
        else:
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

            img_w = self.pixmap.width()
            img_h = self.pixmap.height()
            win_w = self.width()
            win_h = self.height()
            # if w > 0 or h > 0:
            #     w_scale = self.width() / w
            #     h_scale = self.height() / h
            #     p.scale(w_scale, h_scale)
            scale = win_h / img_h
            if  win_w < img_w * scale:
                scale = win_w / img_w
                y = int(win_h / 2) - int(img_h * scale / 2)
            else:
                x = (int(win_w / 2) - int(img_w * scale / 2))

        p.scale(scale, scale)
        p.drawPixmap(int(x/scale), int(y/scale), self.pixmap)
        p.end()

    def resizeEvent(self, event):
        self.update()
