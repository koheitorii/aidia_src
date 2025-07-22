import argparse
import sys
import os

from qtpy import QtWidgets
from qtpy import QtGui
from qtpy import QtCore

from aidia import __appname__, __version__
from aidia import APP_DIR, HOME_DIR, CFONT, CFONT_SIZE
from aidia.config import get_config


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Show splash screen
    splash_path = os.path.join(APP_DIR, 'icons', 'splash.png')
    pixmap = QtGui.QPixmap(splash_path)
    if pixmap.isNull():
        print("Error: Splash image not found at {}".format(splash_path))
        sys.exit(1)
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()

    parser = argparse.ArgumentParser()
    parser.add_argument('--reset-config', action='store_true', help='reset qt config')
    # default_config_file = utils.join(APP_DIR, '../config.yaml')
    default_config_file = os.path.join(HOME_DIR, '.aidiarc')
    parser.add_argument(
        '--config',
        dest='config',
        help='config file or yaml-format string (default: {})'.format(default_config_file),
        default=default_config_file
    )
    args = parser.parse_args()

    config_from_args = args.__dict__
    reset_config = config_from_args.pop('reset_config')
    config_file = config_from_args.pop('config')
    config = get_config(config_file, config_from_args)

    # set language translation
    translator_base = None
    translator = None
    if QtCore.QLocale.system().name() == "ja_JP":
        translator_base = QtCore.QTranslator()
        translator = QtCore.QTranslator()
        translator_base.load("qtbase_ja_JP", QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.TranslationsPath))
        #translator.load(QtCore.QLocale.system().name(), '{}/translate'.format(APP_DIR))
        translator.load('ja_JP', os.path.join(APP_DIR, 'translate'))

    app.setApplicationName(__appname__)
    # app.setWindowIcon(new_icon('icon'))
    icons_dir = os.path.join(APP_DIR, 'icons')
    icon_path = os.path.join(icons_dir, 'icon.ico')
    app.setWindowIcon(QtGui.QIcon(icon_path))
    # app.setWindowIcon(QtGui.QIcon.fromTheme(QtGui.QIcon.ThemeIcon.Computer))
    if translator_base is not None and translator is not None:
        app.installTranslator(translator_base)
        app.installTranslator(translator)
    font = QtGui.QFont(CFONT)
    font.setPointSize(CFONT_SIZE)
    app.setFont(font)

    from aidia.app import MainWindow
    win = MainWindow(config=config)

    if reset_config:
        print('Resetting Qt config: {}'.format(win.settings.fileName()))
        win.settings.clear()
        sys.exit(0)

    # win.show()
    win.showMaximized()
    splash.finish(win)
    win.raise_()
    app.setActiveWindow(win)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
