import os
from qtpy import QtCore
from qtpy import QtWidgets

from aidia.qt import head_text, hline
from aidia import __version__, APP_DIR
from aidia.widgets.image_widget import ImageWidget

class CopyrightDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint
                            | QtCore.Qt.WindowCloseButtonHint)
        
        self.setWindowTitle(self.tr("About Aidia"))

        app_title = QtWidgets.QLabel(f"Aidia {__version__}")
        app_title.setStyleSheet("font-size: 24px; font-weight: bold;")

        homepage_link = QtWidgets.QLabel('Official Website: <a href="https://trklibrary.com/">https://trklibrary.com/</a>')
        homepage_link.setOpenExternalLinks(True)
        labelme_link = QtWidgets.QLabel('labelme: <a href=https://github.com/wkentaro/labelme>https://github.com/wkentaro/labelme</a>')
        labelme_link.setOpenExternalLinks(True)
        ultralytics_link = QtWidgets.QLabel('ultralytics: <a href="https://www.ultralytics.com/">https://www.ultralytics.com/</a>')
        ultralytics_link.setOpenExternalLinks(True)

        app_icon = ImageWidget(self, image=os.path.join(APP_DIR, 'icons', 'icon.png'), resize=(256, 256), alpha=True)
        app_icon.setFixedSize(256, 256)
        text = QtWidgets.QLabel("Copyright (C) 2021-2025 Kohei Torii.")
        text2 = QtWidgets.QLabel("""Copyright (C) 2021-2025 Kohei Torii.
Copyright (C) 2016 Kentaro Wada.
Copyright (C) 2011 Michael Pitidis, Hussein Abdulwahid.

Aidia is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Aidia is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Aidia. If not, see <http://www.gnu.org/licenses/>.""")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(app_icon, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(app_title, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QtWidgets.QLabel("Developed by Kohei Torii, Tokushima University, Japan"), alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(homepage_link, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(hline())
        layout.addWidget(head_text("Copyright"))
        layout.addWidget(text)
        layout.addWidget(head_text("License (GPLv3)"))
        layout.addWidget(text2)
        layout.addWidget(head_text("Thanks to"))
        layout.addWidget(labelme_link)
        layout.addWidget(ultralytics_link)

        self.setLayout(layout)

    def popUp(self):
        self.exec_()
