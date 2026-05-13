import os
from qtpy import QtCore
from qtpy import QtWidgets

from aidia import __version__, APP_DIR
from aidia import qt

class CopyrightDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint
                            | QtCore.Qt.WindowCloseButtonHint)
        
        self.setWindowTitle("About Aidia")

        app_title = QtWidgets.QLabel(f"Aidia {__version__}")
        app_title.setStyleSheet("font-size: 24px; font-weight: bold;")

        app_icon = qt.ImageWidget(self, image=os.path.join(APP_DIR, 'icons', 'icon.png'), resize=(128, 128), alpha=True)
        app_icon.setFixedSize(128, 128)

        copyright_text = QtWidgets.QLabel("Copyright (C) 2021-2026 Kohei Torii.")
        homepage_link = QtWidgets.QLabel('Official Website: <a href="https://trklibrary.com/aidia2/">https://trklibrary.com/aidia2/</a>')
        source_link = QtWidgets.QLabel('Source Code: <a href="https://github.com/koheitorii/aidia_src/">https://github.com/koheitorii/aidia_src/</a>')

        homepage_link.setOpenExternalLinks(True)
        source_link.setOpenExternalLinks(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(app_icon, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(app_title, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_text, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(homepage_link, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(source_link, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

    def popUp(self):
        self.exec_()
