import os
import numpy as np
import cv2
from qtpy import QtWidgets, QtGui, QtCore

from aidia.image import imread

class ImageWidget(QtWidgets.QWidget):
    
    def __init__(self, parent, image=None, resize=None, alpha=False):
        super().__init__(parent=parent)

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
