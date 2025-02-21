'''Copyright (C) 2023 Kohei Torii'''

import functools
import re
import os
import shutil
import webbrowser
import collections
from glob import glob

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from aidia import __appname__, __version__, PRETRAINED_DIR, LABEL_COLORMAP, HOME_DIR, LITE, EXTS, AI_DIR_NAME
from aidia import S_EPSILON, S_AREA_LIMIT
from aidia import qt
from aidia import utils
from aidia.image import imread, save_canvas_img
from aidia.config import get_config
from aidia.label_file import LabelFile
from aidia.label_file import LabelFileError
from aidia.shape import Shape
from aidia.widgets import Canvas
from aidia.widgets import LabelWidget
from aidia.widgets import SettingDialog
from aidia.widgets import CopyrightDialog
from aidia.widgets import CopyAnnotationsDialog
from aidia.widgets import DICOMDialog
from aidia.widgets import LabelEditDialog
from aidia.widgets import LabelListWidget
from aidia.widgets import LabelListWidgetItem
from aidia.widgets import ToolBar
from aidia.widgets import ZoomWidget
from aidia.dicom import DICOM, is_dicom

if not LITE:
    from aidia.widgets.ai_train_dialog import AITrainDialog
    from aidia.widgets.ai_eval_dialog import AIEvalDialog
    from aidia.ai.config import AIConfig
from aidia.widgets.ai_test_widget import AITestWidget


WEB_URL = "https://kottonhome.sakura.ne.jp/"

NO_DATA, EDIT = 0, 1

STATE_COLORS = {
    NO_DATA: QtGui.QBrush(QtCore.Qt.white),
    EDIT: QtGui.QBrush(QtCore.Qt.yellow),
}


class MainWindow(QtWidgets.QMainWindow):

    FIT_WINDOW_MODE, MANUAL_ZOOM = 0, 1
    DEFAULT_WIN_THRESH = 0.08

    def __init__(self, config=None):
        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        super().__init__()
        self.setWindowTitle(__appname__ + " " + __version__)

        self.dirty = False
        self._noSelectionSlot = False
        self.count_images = 0
        self.count_annotations = 0
        self.recentFiles = []
        self.maxRecent = 7
        self.zoom_level = None
        self.fit_window = False
        self.label_dialog_pos = None
        self.selected_polygon = None
        self.work_dir = None
        self.prev_dir = None
        self.dicom_data = None

        # mode of annotation
        self.create_mode = None

        # label search
        self.labels = {}
        self.data_labels = []
        self.target_label = None
        self.target_name = None

        # timer
        self.timer = QtCore.QElapsedTimer()
        self.elapsed_time = 0.0

        # label file values
        self.img_path = None
        self.lf_path = None
        self.note = ""

        # ONNX path for AI test
        self.model_dir = None

        # load application settings
        self.settings = QtCore.QSettings("aidia", "aidia")
        self.work_dir = self.settings.value("work_dir", True, str)
        self.approx_epsilon = self.settings.value("approx_epsilon", 0.03, float)
        self.area_limit = self.settings.value("area_limit", 50, int)
        self.is_polygon = self.settings.value("is_polygon", True, bool)
        self.is_rectangle = self.settings.value("is_rectangle", True, bool)
        self.is_linestrip = self.settings.value("is_linestrip", True, bool)
        self.is_line = self.settings.value("is_line", True, bool)
        self.is_point = self.settings.value("is_point", True, bool)
        self.is_submode = self.settings.value("is_submode", False, bool)
        self.text_ai_select = self.settings.value("ai_select", None, str)
        label_def = self.settings.value("label_def", ["label"], list)
        is_multi_label = self.settings.value("is_multi_label", False, bool)

        # initialize toolbar
        self.tools = self.toolbar("Tools")

        # initialize popup dialog
        self.copyrightDialog = CopyrightDialog(parent=self)
        self.settingDialog = SettingDialog(parent=self)
        self.dicomDialog = DICOMDialog(parent=self) # TODO
        self.labelEditDialog = LabelEditDialog(parent=self)
    
        if not LITE:
            self.ai_train_dialog = AITrainDialog(parent=self)
            self.ai_train_dialog.aiRunning.connect(self.callback_ai_train_running)
            self.ai_eval_dialog = AIEvalDialog(parent=self)
            self.ai_eval_dialog.aiRunning.connect(self.callback_ai_eval_running)

        # initialize AI test widget
        self.ai_test_widget = AITestWidget(self)

        # initialize label widget
        self.labelWidget = LabelWidget(self, label_def, is_multi_label)
        self.labelWidget.valueChanged.connect(self.update_label)

        # label list dock
        self.labelList = LabelListWidget()
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        # self.labelList.itemDoubleClicked.connect(self.edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        # filelist dock
        self.button_refresh = QtWidgets.QPushButton(self.tr("Refresh"))
        self.button_refresh.clicked.connect(self.update_dir)

        self.labelSearch = QtWidgets.QLineEdit()
        self.labelSearch.setPlaceholderText(self.tr("Search Label"))
        self.labelSearch.returnPressed.connect(self.labelSearchChanged)
        # self.labelSearch.textChanged.connect(self.labelSearchChanged)
        self.nameSearch = QtWidgets.QLineEdit()
        self.nameSearch.setPlaceholderText(self.tr('Search File'))
        # self.nameSearch.textChanged.connect(self.nameSearchChanged)
        self.nameSearch.returnPressed.connect(self.nameSearchChanged)

        self.fileListWidget = QtWidgets.QListWidget()
        # self.fileListWidget.itemClicked.connect(
        #     self.fileSelectionChanged)
        self.fileListWidget.currentRowChanged.connect(
            self.rowSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.nameSearch)
        fileListLayout.addWidget(self.labelSearch)
        fileListLayout.addWidget(self.fileListWidget)
        fileListLayout.addWidget(self.button_refresh)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName(u"Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        # summary widget layout
        self.label_sum = QtWidgets.QLabel(self)
        self.label_sum_edit = QtWidgets.QLabel(self)
        layout = QtWidgets.QVBoxLayout()
        layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        layout.addWidget(self.label_sum)
        layout.addWidget(self.label_sum_edit)
        summary_widget = QtWidgets.QWidget()
        summary_widget.setLayout(layout)
        summary_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        summary_scroll = QtWidgets.QScrollArea(self)
        summary_scroll.widgetResizable = True
        summary_scroll.setWidget(summary_widget)

        self.summary_dock = QtWidgets.QDockWidget(self.tr("Summary"), self)
        self.summary_dock.setObjectName("Summary")
        self.summary_dock.setWidget(summary_scroll)

        # DICOM infomation dock
        self.label_dicom_names = QtWidgets.QLabel(self)
        self.label_dicom_values = QtWidgets.QLabel(self)
        layout = QtWidgets.QHBoxLayout()
        layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        # layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        layout.addWidget(self.label_dicom_names, alignment=Qt.AlignLeft)
        layout.addWidget(self.label_dicom_values, alignment=Qt.AlignLeft)
        dicom_widget = QtWidgets.QWidget()
        dicom_widget.setLayout(layout)
        dicom_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        dicom_scroll = QtWidgets.QScrollArea(self)
        dicom_scroll.widgetResizable = True
        dicom_scroll.setWidget(dicom_widget)

        self.dicom_dock = QtWidgets.QDockWidget(self.tr("DICOM Info"), self)
        self.dicom_dock.setObjectName("DICOM Info")
        self.dicom_dock.setWidget(dicom_scroll)

        # connect canvas widget
        self.zoom_widget = ZoomWidget()
        self.canvas = self.labelList.canvas = Canvas(self, epsilon=self._config["epsilon"])
        self.canvas.zoomRequest.connect(self.zoom_request)
        self.canvas.fileOpenRequest.connect(self.file_open_request)

        # AI Button
        self.button_ai_test = QtWidgets.QPushButton(self)
        self.button_ai_test.setText(self.tr("Automatic Annotation"))
        self.button_ai_test.clicked.connect(self.ai_test)
        self.button_ai_test.setEnabled(False)

        self.ai_select = QtWidgets.QComboBox(self)
        self.ai_select.setStyleSheet("QComboBox{ text-align: center; }")
        def _validate(text):
            path = os.path.join(PRETRAINED_DIR, text)
            if os.path.exists(path):
                self.model_dir = path
            else:
                self.model_dir = None
        self.ai_select.currentTextChanged.connect(_validate)
        self.ai_select.setEnabled(False)

        self.ai_import = QtWidgets.QPushButton(self)
        self.ai_import.setText(self.tr("Import"))
        self.ai_import.clicked.connect(self.import_model)
            
        if not LITE:
            self.button_ai_train = QtWidgets.QPushButton(self)
            self.button_ai_train.setText(self.tr("AI Training"))
            self.button_ai_train.clicked.connect(self.ai_train_popup)
            self.button_ai_train.setEnabled(True)

            self.button_ai_eval = QtWidgets.QPushButton(self)
            self.button_ai_eval.setText(self.tr("AI Evaluation"))
            self.button_ai_eval.clicked.connect(self.ai_eval_popup)
            self.button_ai_eval.setEnabled(True)

            self.tag_is_submode = QtWidgets.QLabel(self.tr("from Parent Directory"))
            self.tag_is_submode.setToolTip(self.tr("""Find data from the parent directory."""))
            self.input_is_submode = QtWidgets.QCheckBox()
            self.input_is_submode.setChecked(self.is_submode)
            def _validate(state):
                if state == 2:
                    self.is_submode = True
                else:
                    self.is_submode = False
            self.input_is_submode.stateChanged.connect(_validate)

            submode_layout = QtWidgets.QHBoxLayout()
            submode_layout.addWidget(self.input_is_submode, alignment=QtCore.Qt.AlignRight)
            submode_layout.addWidget(self.tag_is_submode, alignment=QtCore.Qt.AlignLeft)
            submode_widget = QtWidgets.QWidget()
            submode_widget.setLayout(submode_layout)

        self.ai_dock = QtWidgets.QDockWidget(self.tr("AI"), self)
        self.ai_dock.setObjectName("AI")
        ai_layout = QtWidgets.QGridLayout()
        ai_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        # ai_layout.addWidget(self.auto_create_button)
        ai_layout.addWidget(self.button_ai_test, 0, 0, 1, 4)
        ai_layout.addWidget(self.ai_select, 1, 0, 1, 3)
        ai_layout.addWidget(self.ai_import, 1, 3, 1, 1)
        if not LITE:
            ai_layout.addWidget(self.button_ai_train, 2, 0, 1, 2)
            ai_layout.addWidget(self.button_ai_eval, 2, 2, 1, 2)
            ai_layout.addWidget(submode_widget, 3, 0, 1, 4, alignment=QtCore.Qt.AlignCenter)
            # ai_layout.addWidget(self.input_is_submode, 3, 0, 1, 1, QtCore.Qt.AlignRight)
            # ai_layout.addWidget(self.tag_is_submode, 3, 1, 1, 3, QtCore.Qt.AlignLeft)
        ai_widget = QtWidgets.QWidget()
        ai_widget.setLayout(ai_layout)
        self.ai_dock.setWidget(ai_widget)

        # note dock
        self.input_note = QtWidgets.QTextEdit(self)
        self.input_note.setPlaceholderText(self.tr('note'))
        self.input_note.textChanged.connect(self.update_note)
        self.input_note.setMaximumHeight(100)
        self.input_note.setEnabled(False)

        self.note_dock = QtWidgets.QDockWidget(self.tr("Note"), self)
        self.note_dock.setObjectName("Note")
        note_layout = QtWidgets.QVBoxLayout()
        note_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        note_layout.addWidget(self.input_note)
        note_widget = QtWidgets.QWidget()
        note_widget.setLayout(note_layout)
        self.note_dock.setWidget(note_widget)

        # timer dock
        self.interrupt_timer = QtCore.QTimer()
        self.interrupt_timer.setInterval(10)
        self.interrupt_timer.timeout.connect(self.timer_update)

        self.button_start_timer = QtWidgets.QPushButton(self.tr("Start"), self)
        self.button_start_timer.clicked.connect(self.timer_start)
        self.button_start_timer.setEnabled(False)

        self.button_stop_timer = QtWidgets.QPushButton(self.tr("Stop"), self)
        self.button_stop_timer.clicked.connect(self.timer_stop)
        self.button_stop_timer.setEnabled(False)

        self.label_timer = QtWidgets.QLabel("0.0 [s]", self)

        self.timer_dock = QtWidgets.QDockWidget(self.tr("Timer"), self)
        self.timer_dock.setObjectName("Timer")
       
        timer_layout = QtWidgets.QVBoxLayout()
        timer_layout.addWidget(self.label_timer, alignment=Qt.AlignRight)
        timer_layout.addWidget(self.button_start_timer)
        timer_layout.addWidget(self.button_stop_timer)
        timer_widget = QtWidgets.QWidget()
        timer_widget.setLayout(timer_layout)
        self.timer_dock.setWidget(timer_widget)

        # label dock
        self.ld_dock = QtWidgets.QDockWidget(self.tr("Labels"), self)
        self.ld_dock.setObjectName("Labels")
        ld_layout = QtWidgets.QVBoxLayout()
        ld_layout.addWidget(self.labelWidget)
        ld_widget = QtWidgets.QWidget()
        ld_widget.setLayout(ld_layout)
        self.ld_dock.setWidget(ld_widget)

        # set canvas functions
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scroll_request)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.shapeDoubleClicked.connect(self.edit_label_shape_selected)
        self.canvas.setDirty.connect(self.setDirty)
        self.canvas.updateStatus.connect(self.status)

        self.setCentralWidget(scrollArea)

        # set dock widgets
        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        features = features | QtWidgets.QDockWidget.DockWidgetClosable
        features = features | QtWidgets.QDockWidget.DockWidgetFloatable
        # features = features | QtWidgets.QDockWidget.DockWidgetMovable

        self.file_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.shape_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.ai_dock.setAllowedAreas(Qt.RightDockWidgetArea)

        self.ld_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.note_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.timer_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.summary_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.dicom_dock.setAllowedAreas(Qt.BottomDockWidgetArea)

        self.file_dock.setFeatures(features)
        self.shape_dock.setFeatures(features)
        self.ld_dock.setFeatures(features)
        self.ai_dock.setFeatures(features)
        self.note_dock.setFeatures(features)
        self.timer_dock.setFeatures(features)
        self.summary_dock.setFeatures(features)
        self.dicom_dock.setFeatures(features)

        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.ai_dock)

        self.addDockWidget(Qt.BottomDockWidgetArea, self.ld_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.note_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.timer_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.summary_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dicom_dock)

        self.tabifyDockWidget(self.note_dock, self.timer_dock)
        self.tabifyDockWidget(self.note_dock, self.summary_dock)
        self.tabifyDockWidget(self.note_dock, self.dicom_dock)

        action = functools.partial(qt.newAction, self)
        shortcuts = self._config["shortcuts"]

        toggle_show_label_action = action(
            self.tr("Toggle Labels Show and Hide"),
            self.canvas.toggle_show_label,
            shortcut="Space",
            icon="toggle-label",
            tip=self.tr("Toggle labels show and hide."),
            enabled=True
        )

        quit_action = action(
            self.tr("&Quit"), self.close, shortcuts["quit"], "quit",
            self.tr("Quit application.")
        )

        open_action = action(
            self.tr("&Load Image"),
            self.open_file,
            shortcuts["open"],
            "open",
            self.tr("Load image file.")
        )
   
        open_next_action = action(
            self.tr("&Next Image"),
            self.open_next_img,
            # shortcuts["open_next"],
            "Ctrl+Down",
            "next",
            self.tr("Open next image."),
            enabled=False
        )
        open_prev_action = action(
            self.tr("&Prev Image"),
            self.open_prev_img,
            # shortcuts["open_prev"],
            "Ctrl+Up",
            "prev",
            self.tr("Open previous image."),
            enabled=False
        )
        save_action = action(
            self.tr("&Save"),
            self.save_file, shortcuts["save"], "save",
            self.tr("Save labels to file."), enabled=False
        )
        save_as_action = action(
            self.tr("&Save As"), self.save_file_as,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file."),
            enabled=False
        )
        delete_file_action = action(
            self.tr("&Delete File"),
            self.delete_file,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file."),
            enabled=False
        )

        # export annotations
        export_anno_action = action(
            self.tr("&Export Annotations"),
            self.export_annotations,
            # shortcuts["delete_file"],
            # icon=None,
            self.tr("Export JSON annotation files."),
            enabled=True
        )

        # import pretrained model
        import_model_action = action(
            self.tr("&Import Pretrained Model"),
            self.import_model,
            # shortcuts["delete_file"],
            # icon=None,
            self.tr("Import pretrained models."),
            enabled=True
        )

        close_action = action(
            text=self.tr("&Close"),
            slot=self.close_file,
            shortcut=shortcuts["close"],
            icon="close",
            tip=self.tr("Close current file.")
        )
        create_mode_action = action(
            text=self.tr("Create Polygons"),
            slot=lambda: self.toggleDrawMode(False, create_mode='polygon'),
            # shortcut=shortcuts["create_polygon"],
            shortcut="N",
            icon="polygon",
            tip=self.tr("Start drawing polygons."),
            enabled=False,
        )
        create_rectangle_mode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, create_mode="rectangle"),
            # shortcuts["create_rectangle"],
            "R",
            "rectangle",
            self.tr("Start drawing rectangles."),
            enabled=False,
        )
        create_linestrip_mode = action(
            self.tr("Create Linestrip"),
            lambda: self.toggleDrawMode(False, create_mode="linestrip"),
            # shortcuts["create_linestrip"],
            "S",
            "line-strip",
            self.tr("Start drawing linestrips."),
            enabled=False,
        )
        create_line_mode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, create_mode="line"),
            # shortcuts["create_line"],
            "L",
            "line",
            self.tr("Start drawing a line."),
            enabled=False,
        )
        create_point_mode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, create_mode="point"),
            # shortcuts["create_point"],
            "P",
            "point",
            self.tr("Start drawing a point."),
            enabled=False,
        )
        edit_mode_action = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            # shortcuts["edit_polygon"],
            ["E", "ESC"],
            "edit",
            self.tr("Move and edit the selected polygons."),
            enabled=False
        )
        delete_action = action(
            self.tr("Delete Polygons"),
            self.delete_selected_shape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons."),
            enabled=False
        )
        copy_action = action(
            self.tr("Duplicate Polygons"),
            self.copySelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons."),
            enabled=False
        )
        undo_last_point_action = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point."),
            enabled=False
        )
        add_point_action = action(
            text=self.tr("Add Point to Edge"),
            slot=self.canvas.addPointToEdge,
            shortcut=None,
            icon="edit",
            tip=self.tr("Add point to the nearest edge."),
            enabled=False
        )
        remove_point_action = action(
            text=self.tr("Remove Selected Point"),
            slot=self.canvas.removeSelectedPoint,
            icon="edit",
            tip="Remove selected point from polygon.",
            enabled=False
        )
        undo_action = action(
            self.tr("Undo"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape."),
            enabled=False
        )
        hide_all_action = action(
            text=self.tr("&Hide\nPolygons"),
            slot=functools.partial(self.toggle_polygons, False),
            shortcut=shortcuts["hide_all"],
            icon="hide",
            tip=self.tr("Hide all polygons."),
            enabled=False
        )
        show_all_action = action(
            text=self.tr("&Show\nPolygons"),
            slot=functools.partial(self.toggle_polygons, True),
            shortcut=shortcuts["show_all"],
            icon="show",
            tip=self.tr("Show all polygons."),
            enabled=False
        )
        toggle_polygon_action = action(
            text=self.tr("&Toggle\nShow/Hide Selected Polygon"),
            slot=self.toggle_single_polygon,
            shortcut=shortcuts["toggle_polygon"],
            icon="eye",
            tip=self.tr("Toggle show/hide selected polygon."),
            enabled=False
        )

        reset_brightness_contrast_action = action(
            text=self.tr("&Reset Brightness and Contrast"),
            slot=self.canvas.reset_brightness_contrast,
            icon="contrast",
            tip=self.tr("Reset brightness and contrast."),
            enabled=False
        )

        help_action = action(
            self.tr("&Help"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show Aidia GitHub page.")
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoom_widget)
        self.zoom_widget.setWhatsThis(
            self.tr(
                "Zoom in or out of the image. Also accessible with "
                "{} and {} from the canvas."
            ).format(
                qt.fmtShortcut(
                    "{},{}".format(
                        shortcuts["zoom_in"], shortcuts["zoom_out"]
                    )
                ),
                qt.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoom_widget.setEnabled(False)

        zoom_in_action = action(
            self.tr("Zoom &In"),
            functools.partial(self.add_zoom, 1.1),
            shortcuts["zoom_in"], "zoom-in",
            self.tr("Increase zoom level."), enabled=False)
        zoom_out_action = action(
            self.tr("&Zoom Out"),
            functools.partial(self.add_zoom, 0.9),
            shortcuts["zoom_out"], "zoom-out",
            self.tr("Decrease zoom level."), enabled=False)
        zoom_org_action = action(
            self.tr("&Original size"),
            functools.partial(self.set_zoom, 100),
            shortcuts["zoom_to_original"], "zoom",
            self.tr("Zoom to original size."), enabled=False)
        fit_window_action = action(
            self.tr("&Fit Window"), self.set_fit_window,
            shortcuts["fit_window"], "fit-window",
            self.tr("Zoom follows window size."), checkable=True,
            enabled=False)

        # Group zoom controls into a list for easier toggling.
        zoom_actions = (self.zoom_widget, zoom_in_action, zoom_out_action,
                        zoom_org_action, fit_window_action)
        self.zoom_mode = self.FIT_WINDOW_MODE
        fit_window_action.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW_MODE: self.scale_fit_window,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit_action = action(
            self.tr("&Edit Label"), self.edit_label,
            # shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon."),
            enabled=False
        )

        popup_copyright_action = action(
            text=self.tr("&Copyright"),
            slot=self.popup_copyright,
            icon="copyright",
            tip=self.tr("Open copyright information."),
            enabled=True
        )

        popup_setting_action = action(
            text=self.tr("&Setting"),
            slot=self.popup_setting,
            icon="setting",
            tip=self.tr("Open setting dialog."),
            enabled=True
        )

        save_canvas_img_action = action(
            text=self.tr("&Export PNG"),
            slot=self.export_canvas_img,
            icon="save-as",
            tip=self.tr("Export the canvas image to PNG image."),
            enabled=True
        )

        delete_pretrained_model_action = action(
            text=self.tr("&Delete Pretrained Models"),
            slot=self.delete_pretrained_model,
            tip=self.tr("Delete pretrained model."),
            enabled=True
        )

        label_edit_action = action(
            text=self.tr("&Edit Label"),
            slot=self.edit_label_shape_selected,
            icon='pen',
            tip=self.tr("Edit labels directly"),
            enabled=True
        )

        # toggle view toolbar buttons
        def _func(value):
            self.tools.flags[2] = value
            self.tools.updateShowButtons()
            self.is_polygon = value
        show_polygon_mode_action = action(
            text=self.tr("&Show Polygon Mode"),
            slot=_func,
            checkable=True,
            enabled=True,
            checked=self.is_polygon,
        )

        def _func(value):
            self.tools.flags[3] = value
            self.tools.updateShowButtons()
            self.is_rectangle = value
        show_rectangle_mode_action = action(
            text=self.tr("&Show Rectangle Mode"),
            slot=_func,
            checkable=True,
            enabled=True,
            checked=self.is_rectangle,
        )

        # def _func(value):
        #     self.tools.flags[4] = value
        #     self.tools.updateShowButtons()
        #     self.is_linestrip = value
        # show_linestrip_mode_action = action(
        #     text=self.tr("&Show Linestrip Mode"),
        #     slot=_func,
        #     checkable=True,
        #     enabled=True,
        #     checked=self.is_linestrip,
        # )

        # def _func(value):
        #     self.tools.flags[5] = value
        #     self.tools.updateShowButtons()
        #     self.is_line = value
        # show_line_mode_action = action(
        #     text=self.tr("&Show Line Mode"),
        #     slot=_func,
        #     checkable=True,
        #     enabled=True,
        #     checked=self.is_line,
        # )

        # def _func(value):
        #     self.tools.flags[6] = value
        #     self.tools.updateShowButtons()
        #     self.is_point = value
        # show_point_mode_action = action(
        #     text=self.tr("&Show Point Mode"),
        #     slot=_func,
        #     checkable=True,
        #     enabled=True,
        #     checked=self.is_point,
        # )

        # label list right click menu.
        labelMenu = QtWidgets.QMenu()
        qt.addActions(labelMenu, [delete_action])
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = qt.struct(
            save=save_action,
            saveAs=save_as_action,
            open=open_action,
            close=close_action,
            deleteFile=delete_file_action,
            delete=delete_action,
            exportAnno=export_anno_action,
            importModel=import_model_action,
            edit=edit_action,
            popup_copyright=popup_copyright_action,
            popup_setting=popup_setting_action,
            copy=copy_action,
            undoLastPoint=undo_last_point_action,
            undo=undo_action,
            addPointToEdge=add_point_action,
            removePoint=remove_point_action,
            createMode=create_mode_action,
            createRectangleMode=create_rectangle_mode,
            # createLineStripMode=create_linestrip_mode,
            # createLineMode=create_line_mode,
            # createPointMode=create_point_mode,
            showPolygonMode=show_polygon_mode_action,
            showRectangleMode=show_rectangle_mode_action,
            # showLinestripMode=show_linestrip_mode_action,
            # showLineMode=show_line_mode_action,
            # showPointMode=show_point_mode_action,
            editMode=edit_mode_action,
            resetBrightnessContrast=reset_brightness_contrast_action,
            zoom=zoom, zoomIn=zoom_in_action,
            zoomOut=zoom_out_action,
            zoomOrg=zoom_org_action,
            fitWindow=fit_window_action,
            zoomActions=zoom_actions,
            openNextImg=open_next_action,
            openPrevImg=open_prev_action,
            toggle_show_label=toggle_show_label_action,
            fileMenuActions=(
                open_action,
                # opendir_action,
                save_action,
                save_as_action,
                close_action,
                quit_action),
            tool=(),
            editMenu=(
                # edit_action,
                # copy_action,
                delete_action,
                None,
                undo_action,
                undo_last_point_action,
                None,
                # add_point_action,
            ),
            # canvas right click menu.
            canvas_menu=(
                # add_point_action,
                # remove_point_action,
                # edit_action,
                # copy_action,
                label_edit_action,
                delete_action,
                undo_action,
                undo_last_point_action,
                toggle_polygon_action,
                # show_all_action,
                # hide_all_action,
                reset_brightness_contrast_action,
                save_canvas_img_action,
            ),
            onLoadActive=(
                close_action,
                create_mode_action,
                create_rectangle_mode,
                edit_mode_action,
            ),
            onShapesPresent=(save_as_action, hide_all_action,
                             show_all_action, toggle_polygon_action),
        )

        self.canvas.edgeSelected.connect(self.canvasShapeEdgeSelected)
        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        # menu bar
        self.menus = qt.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            tools=self.menu(self.tr("&Tools")),
            setting=self.menu(self.tr("&Option")),
            # help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        qt.addActions(
            self.menus.file,
            (
                open_action,
                # opendir_action,
                open_prev_action,
                open_next_action,
                self.menus.recentFiles,
                save_action,
                save_as_action,
                # close_action,
                None,
                delete_file_action,
                quit_action,
            ),
        )
        # utils.addActions(self.menus.help, (help_action,))
        qt.addActions(
            self.menus.setting,
            (
                popup_setting_action,
                help_action,
                popup_copyright_action,
            ),
        )
        qt.addActions(
            self.menus.view,
            (
                show_polygon_mode_action,
                show_rectangle_mode_action,
                # show_linestrip_mode_action,
                # show_line_mode_action,
                # show_point_mode_action,
                None,
                self.file_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.ld_dock.toggleViewAction(),
                self.ai_dock.toggleViewAction(),
                self.dicom_dock.toggleViewAction(),
                self.note_dock.toggleViewAction(),
                self.timer_dock.toggleViewAction(),
                self.summary_dock.toggleViewAction(),
                None,
                hide_all_action,
                show_all_action,
                toggle_polygon_action,
                toggle_show_label_action,
                None,
                zoom_in_action,
                zoom_out_action,
                zoom_org_action,
                fit_window_action,
                None,
                reset_brightness_contrast_action,
            ),
        )

        qt.addActions(
            self.menus.tools,
            (
                export_anno_action,
                import_model_action,
                delete_pretrained_model_action,
            )
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        qt.addActions(self.canvas.menus, self.actions.canvas_menu)

        # tool bar
        self.actions.tool = (
            open_action,
            # opendir_action,
            # open_prev_action,
            # open_next_action,
            save_action,
            # delete_file_action,
            create_mode_action,
            create_rectangle_mode,
            # create_linestrip_mode,
            # create_line_mode,
            # create_point_mode,
            edit_mode_action,
            # copy_action,
            # delete_action,
            undo_action,
            # zoom_in_action,
            # zoom,
            # zoom_out_action,
            fit_window_action,
        )

        # custom toolbar context menu
        self.tools.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tools.customContextMenuRequested.connect(self.toolbar_menu)

        # initialize process
        self.statusBar().showMessage(self.tr("{} started.").format(__appname__))
        self.statusBar().show()

        # Restore application settings.
        self.recentFiles = self.settings.value("recentFiles", []) or []
        # self.resize(QtCore.QSize(1280, 720))
        size = self.settings.value("window/size", QtCore.QSize(1280, 720))
        self.resize(size)
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        self.move(position)
        # self.restoreState(self.settings.value("window/state", QtCore.QByteArray()))

        # Populate the File menu dynamically.
        self.updateFileMenu()

        self.zoom_widget.valueChanged.connect(self.paint_canvas)

        # build toolbar and canvas menu
        self.populateModeActions()

        # restore toolbar status
        self.tools.flags[2] = self.is_polygon
        self.tools.flags[3] = self.is_rectangle
        # self.tools.flags[4] = self.is_linestrip
        # self.tools.flags[5] = self.is_line
        # self.tools.flags[6] = self.is_point
        self.tools.updateShowButtons()

        # set disable canvas by default
        self.canvas.setEnabled(False)

        # init directory
        self.queueEvent(self.init_dir)
        
        # set menubar on window top
        self.menuBar().setNativeMenuBar(False)

        # popup gpu info
        if not LITE and not AIConfig.is_gpu_available():
            self.info_message(self.tr('''No GPU is available.<br>Please keep in mind that many times takes in training AI.'''))

    def init_dir(self):
        if self.work_dir is not None and os.path.exists(self.work_dir):
            self.import_from_dir(self.work_dir)
        else:
            self.work_dir = HOME_DIR

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            qt.addActions(menu, actions)
        return menu
    
    def toolbar_menu(self):
        m = QtWidgets.QMenu()
        m.addActions([
            self.actions.showPolygonMode,
            self.actions.showRectangleMode,
            # self.actions.showLinestripMode,
            # self.actions.showLineMode,
            # self.actions.showPointMode,
        ])
        m.exec_(QtGui.QCursor.pos())


    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("{}ToolBar".format(title))
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        toolbar.setFloatable(False)
        toolbar.setMovable(False)
        if actions:
            qt.addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar

    ### Override Functions ###
    def closeEvent(self, event):
        if not self.may_continue_unsaved():
            event.ignore()
        if not LITE and not self.may_continue_ai_running():
            event.ignore()

        self.settings.setValue("filename", self.img_path if self.img_path else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        self.settings.setValue("work_dir", self.work_dir)
        self.settings.setValue("is_polygon", self.is_polygon)
        self.settings.setValue("is_rectangle", self.is_rectangle)
        self.settings.setValue("is_linestrip", self.is_linestrip)
        self.settings.setValue("is_line", self.is_line)
        self.settings.setValue("is_point", self.is_point)
        self.settings.setValue("is_multi_label", self.labelWidget.is_multi_label)
        self.settings.setValue("label_def", self.labelWidget.label_def)
        self.settings.setValue("approx_epsilon", self.approx_epsilon)
        self.settings.setValue("area_limit", self.area_limit)
        self.settings.setValue("is_submode", self.is_submode)
        self.settings.setValue("ai_select", self.ai_select.currentText())

        # ask the use for where to save the labels
        # self.settings.setValue("window/geometry", self.saveGeometry())

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.canvas_menu
        self.tools.clear()
        qt.addActions(self.tools, tool)
        self.canvas.menus.clear()
        qt.addActions(self.canvas.menus, menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            # self.actions.createLineStripMode,
            # self.actions.createLineMode,
            # self.actions.createPointMode,
            self.actions.editMode,
        )
        qt.addActions(self.menus.edit, actions + self.actions.editMenu)


    def setDirty(self):
        """Manage unsaved annotation files."""
        self.dirty = True
        self.actions.save.setEnabled(True)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        if self.img_path is not None:
            self.update_title(set_dirty=True)
         

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        # self.actions.createMode.setEnabled(True)
        # self.actions.createRectangleMode.setEnabled(True)
        # title = __appname__ + " " + __version__
        if self.img_path is not None:
            self.update_title()

        if self.has_label_file():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)


    def update_title(self, set_dirty=False):
        title = __appname__ + " " + __version__
        title += " - {}".format(self.img_path)
        if self.lf_path is not None and os.path.exists(self.lf_path):
            title += " - {}".format(self.lf_path)
        if set_dirty:
            title += '*'
        self.setWindowTitle(title)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def canvasShapeEdgeSelected(self, selected, shape):
        self.actions.addPointToEdge.setEnabled(
            selected and shape and shape.canAddPoint()
        )

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)


    def resetState(self):
        self.labelList.clear()
        self.labelWidget.reset()

        self.img_path = None
        self.dicom_data = None
        self.labelFile = None
        self.selected_polygon = None
        self.canvas.is_dicom = False

        self.update_state(NO_DATA)
        self.note = ""
        self.input_note.setText('')
        self.input_note.setEnabled(False)

        self.setWindowTitle(__appname__ + " " + __version__)

        self.interrupt_timer.stop()
        self.button_start_timer.setEnabled(False)
        self.button_stop_timer.setEnabled(False)
        self.update_label_timer(0.0)

        self.canvas.resetState()


    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None


    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)


    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)


    def tutorial(self):
        webbrowser.open(WEB_URL)


    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, create_mode='polygon'):
        # self.canvas.setEditing(edit)
        # self.actions.createMode.setEnabled(edit)
        # self.actions.editMode.setEnabled(not edit)
        self.canvas.setEditing(edit)
        self.canvas.createMode = create_mode
        if edit:
            self.create_mode = None
            self.actions.createMode.setEnabled(True)
            self.actions.createRectangleMode.setEnabled(True)
            # self.actions.createLineStripMode.setEnabled(True)
            # self.actions.createLineMode.setEnabled(True)
            # self.actions.createPointMode.setEnabled(True)
        else:
            self.create_mode = create_mode
            if create_mode == "polygon":
                self.actions.createMode.setEnabled(False)
                self.actions.createRectangleMode.setEnabled(True)
                # self.actions.createLineStripMode.setEnabled(True)
                # self.actions.createLineMode.setEnabled(True)
                # self.actions.createPointMode.setEnabled(True)
            elif create_mode == "rectangle":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(False)
                # self.actions.createLineStripMode.setEnabled(True)
                # self.actions.createLineMode.setEnabled(True)
                # self.actions.createPointMode.setEnabled(True)
            # elif create_mode == "linestrip":
            #     self.actions.createMode.setEnabled(True)
            #     self.actions.createRectangleMode.setEnabled(True)
            #     self.actions.createLineStripMode.setEnabled(False)
            #     self.actions.createLineMode.setEnabled(True)
            #     self.actions.createPointMode.setEnabled(True)
            # elif create_mode == "line":
            #     self.actions.createMode.setEnabled(True)
            #     self.actions.createRectangleMode.setEnabled(True)
            #     self.actions.createLineStripMode.setEnabled(True)
            #     self.actions.createLineMode.setEnabled(False)
            #     self.actions.createPointMode.setEnabled(True)
            # elif create_mode == "point":
            #     self.actions.createMode.setEnabled(True)
            #     self.actions.createRectangleMode.setEnabled(True)
            #     self.actions.createLineStripMode.setEnabled(True)
            #     self.actions.createLineMode.setEnabled(True)
            #     self.actions.createPointMode.setEnabled(False)
            else:
                raise ValueError(f"Unsupported createMode: {create_mode}")
        self.actions.editMode.setEnabled(not edit)


    def setEditMode(self):
        self.toggleDrawMode(True)


    def updateFileMenu(self):
        current = self.img_path

        def exists(filename):
            return os.path.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = qt.newIcon("file")
            action = QtWidgets.QAction(
                icon,
                "&{} {}".format(i + 1, QtCore.QFileInfo(f).fileName()),
                self
            )
            action.triggered.connect(functools.partial(self.load_recent, f))
            menu.addAction(action)


    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))


    def update_label(self):
        """Get current labels of the label widget, and update shape labels."""
        if not self.currentItem():
            return
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if item is None:
            return

        shape = item.shape()
        if shape is None:
            return

        shape.label = self.labelWidget.label
        # self.labelList.sort()
        r, g, b = self._get_rgb_by_label(shape)
        item.setText(
            '<font color="#{:02x}{:02x}{:02x}">●</font> {}'
            .format(r, g, b, self.labelWidget.label))
        self.update_shape_color(shape, color=(r, g, b))
        self.setDirty()


    def edit_label(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")

        if not self.canvas.editing():
            return
        if not item:
            item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return

        self.labelWidget.reset()
        self.labelWidget.update(shape.label)

    
    def popup_copyright(self):
        self.copyrightDialog.popUp()

        
    def popup_setting(self):
        new_setting_dict = self.settingDialog.popUp({
                S_EPSILON: self.approx_epsilon,
                S_AREA_LIMIT: self.area_limit,
            })
        if new_setting_dict:
            self.approx_epsilon = new_setting_dict[S_EPSILON]
            self.area_limit = new_setting_dict[S_AREA_LIMIT]

    
    def labelSearchChanged(self):
        """Call back function. Update target label and the shapes."""
        target = self.labelSearch.text().split('_')
        if not len(self.labelSearch.text()):
            self.target_label = None
            self.canvas.target_label = None
            self.canvas.loadShapes([item.shape() for item in self.labelList])
            self.import_from_workdir()
        else:
            self.target_label = target
            self.canvas.target_label = target
            self.canvas.loadShapes([item.shape() for item in self.labelList])
            self.import_from_workdir()


    def nameSearchChanged(self):
        if self.nameSearch.text():
            self.target_name = self.nameSearch.text()
            self.import_from_workdir()
        elif not len(self.nameSearch.text()):
            self.target_name = None
            self.import_from_workdir()


    def file_open_request(self, delta):
        if delta > 0:
            self.open_prev_img()
        else:
            self.open_next_img()


    def rowSelectionChanged(self, currentRow):
        if not self.may_continue_unsaved():
            return
        item = self.fileListWidget.item(currentRow)
        if not item:
            return
        img_path = os.path.join(self.work_dir, item.text())
        currIndex = self.image_list.index(img_path)
        if currIndex < len(self.image_list):
            filename = self.image_list[currIndex]
            if filename:
                self.load_file(filename, update_list=False)
                

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        self.labelWidget.reset()
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
            if len(self.canvas.selectedShapes) == 1:
                self.edit_label(item=item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)

    @staticmethod
    def has_duplicates(seq):
        # len(seq) != len(set(seq))
        dup_list = [k for k, v in collections.Counter(seq).items() if v > 1]
        if len(dup_list):
            return dup_list
        else:
            return None

    def add_label(self, shape):
        text = shape.label
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        # self.labelList.sort()
        r, g, b = self._get_rgb_by_label(shape)
        label_list_item.setText(
            '<font color="#{:02x}{:02x}{:02x}">●</font> {}'
            .format(r, g, b, text)
        )
        self.update_shape_color(shape, color=(r, g, b))

    def _get_rgb_by_label(self, shape):
        if self._config["shape_color"] == "auto":
            # item = self.labelList.findItemByShape(shape)
            # label_id = self.labelList.model().indexFromItem(item).row() + 1
            label_id = 1
            for x in shape.label.split("_"):
                if x in self.labelWidget.label_def:
                    label_id += self.labelWidget.label_def.index(x) + 1
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (self._config["shape_color"] == "manual" and
              self._config["label_colors"] and
              shape.label in self._config["label_colors"]):
            return self._config["label_colors"][shape.label]
    
    def update_shape_color(self, shape, color):
        r, g, b = color
        shape.line_color = QtGui.QColor(r, g, b)
        # shape.line_color = QtCore.Qt.yellow
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        # shape.vertex_fill_color = QtCore.Qt.yellow
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)
        shape.label_color = QtGui.QColor(r, g, b)

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.add_label(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape['shape_type']
            shape = Shape(label=label, shape_type=shape_type)
            [shape.addPoint(QtCore.QPointF(x, y)) for x, y in points]
            shape.close()
            s.append(shape)
        self.loadShapes(s)

    def saveLabels(self, lf_path):
        lf = LabelFile()

        def format_shape(shape):
            data = dict(
                label=shape.label,
                shape_type=shape.shape_type,
                points=[(int(p.x()), int(p.y())) for p in shape.points],
            )
            return data
        shapes = [format_shape(item.shape()) for item in self.labelList]

        try:
            filename = os.path.basename(self.img_path)
            br = 0.0 if self.canvas.brightness is None else self.canvas.brightness
            co = 1.0 if self.canvas.contrast is None else self.canvas.contrast
            lf.save(
                lf_path=lf_path,
                filename=filename,
                height=self.canvas.image.height(),
                width=self.canvas.image.width(),
                elapsed_time=self.elapsed_time,
                brightness=round(br, 4),
                contrast=round(co, 4),
                note=self.note,
                shapes=shapes,
            )
            self.labelFile = lf

            items = self.fileListWidget.findItems(
                self.img_path, Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.PartiallyChecked)
            return True
        except LabelFileError as e:
            self.error_message(
                self.tr("<p>Error saving label data.</p>"
                        "<b>{}</b>").format(e))
            return False

    def copySelectedShape(self):
        added_shapes = self.canvas.copySelectedShapes()
        self.labelList.clearSelection()
        [self.add_label(shape) for shape in added_shapes]
        self.setDirty()

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if not self.canvas.editing():
            self.setEditMode()
        if self.canvas.editing():
            selected_shapes = [item.shape()
                               for item in self.labelList.selectedItems()]
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    
    def add_label_data(self):
        item_name = os.path.basename(self.img_path)
        self.labels[item_name] = self.get_all_labels(self.labelFile)

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        # text = self.labelDialog.popUp(pos=self.label_dialog_pos)
        # if text:
        self.labelList.clearSelection()
        shape = self.canvas.setLastLabel(self.labelWidget.default_label)
        self.add_label(shape)
        self.actions.editMode.setEnabled(True)
        self.actions.undoLastPoint.setEnabled(False)
        self.actions.undo.setEnabled(True)
        self.setDirty()
        # else:
            # self.canvas.undoLastLine()
            # self.canvas.shapesBackups.pop()
    
    def edit_label_shape_selected(self):
        item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        
        result = self.labelEditDialog.popup(shape.label)
        if not result:
            return
        shape.label = result

        # if multi label
        if len(result.split('_')) > 1:
            self.labelWidget.is_multi_label_checkbox.setChecked(True)
        
        # update label dock
        self.labelWidget.reset()
        self.labelWidget.update(shape.label)

        # update label list
        r, g, b = self._get_rgb_by_label(shape)
        item.setText(
            '<font color="#{:02x}{:02x}{:02x}">●</font> {}'
            .format(r, g, b, shape.label))
        self.update_shape_color(shape, color=(r, g, b))
        self.setDirty()

    def scroll_request(self, delta, orientation):
        units = - delta * 0.03  # revirse scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.set_scroll(orientation, value)

    def set_scroll(self, orientation, value):
        self.scrollBars[orientation].setValue(value)

    def set_zoom(self, value):
        self.actions.fitWindow.setChecked(False)
        self.zoom_mode = self.MANUAL_ZOOM
        self.zoom_widget.setValue(value)
        self.zoom_level = value

    def add_zoom(self, increment=1.1):
        self.set_zoom(self.zoom_widget.value() * increment)

    def zoom_request(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.add_zoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.set_scroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.set_scroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def set_fit_window(self, value=True):
        self.zoom_mode = self.FIT_WINDOW_MODE if value else self.MANUAL_ZOOM
        self.adjust_scale()

    def toggle_polygons(self, value):
        [item.setCheckState(Qt.Checked if value else Qt.Unchecked)
         for item in self.labelList]

    def toggle_single_polygon(self):
        selected_items = [item for item in self.labelList.selectedItems()]
        if selected_items:
            for item in selected_items:
                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                self.selected_polygon = item
        elif self.selected_polygon:
            item = self.selected_polygon
            if item.checkState() == Qt.Checked:
                item.setCheckState(Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)
        else:
            return

    def load_file(self, img_path=None, update_list=True):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if (img_path in self.image_list and
                self.fileListWidget.currentRow() !=
                self.image_list.index(img_path) and
                update_list):
            self.fileListWidget.setCurrentRow(self.image_list.index(img_path))
            self.fileListWidget.repaint()
            return False

        if img_path == self.img_path:
            return False

        # print('Loading {}'.format(img_path))
        self.resetState()
        self.canvas.setEnabled(False)

        # img path processing
        if img_path is None:
            img_path = self.settings.value("filename", "")
        img_path = str(img_path)
        if not QtCore.QFile.exists(img_path):
            self.error_message(self.tr("No such file: <b>{}</b>").format(img_path))
            return False

        self.wait_cursor()

        # assumes same name, but json extension
        self.status(self.tr("Loading {}...").format(os.path.basename(str(img_path))))

        # Load label file.
        self._load_labelfile(img_path)

        # load image
        if not self._load_image(img_path):
            return False

        self.img_path = img_path
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
        self.setClean()
        self.canvas.setEnabled(True)

        # enable canvas widget
        self.actions.resetBrightnessContrast.setEnabled(True)

        # enable timer
        self.button_start_timer.setEnabled(True)
        self.button_stop_timer.setEnabled(False)

        # dicom info
        self.update_dicom_info()

        if self.zoom_level and self.zoom_mode == self.MANUAL_ZOOM:
            self.set_zoom(self.zoom_level)
        else:
            self.adjust_scale(initial=True)
        self.addRecentFile(self.img_path)
        self.toggleActions(True)
        # self.setEditMode()
        if self.create_mode is None:
            self.setEditMode()
        else:
            self.toggleDrawMode(edit=False, create_mode=self.create_mode)
        self.status(self.tr("Loaded {}").format(self.img_path))

        self.reset_cursor()

        # init AI functions
        if not LITE:
            self.ai_train_dialog.reset_state()
            self.ai_eval_dialog.reset_state()

        return True

    
    def _load_labelfile(self, img_path):
        self.lf_path = self.make_lfpath(img_path)
        try:
            if os.path.exists(self.lf_path):
                self.labelFile = LabelFile()
                self.labelFile.load(self.lf_path)
            else:
                self.labelFile = None
                self.update_state(NO_DATA)
        except LabelFileError as e:
            self.error_message(self.tr(
                "<p><b>{}</b></p><p>Make sure <i>{}</i> is a valid label file.</p>").format(e, self.lf_path))
            self.labelFile = None
            self.update_state(NO_DATA)
        except FileNotFoundError:
            self.labelFile = None
            self.update_state(NO_DATA)
        else:
            if self.labelFile:
                self.update_state(EDIT)
                self.elapsed_time = self.labelFile.elapsed_time
                self.update_label_timer(self.labelFile.elapsed_time)
                self.canvas.brightness = self.labelFile.brightness
                self.canvas.contrast = self.labelFile.contrast
                self.note = self.labelFile.note
                self.input_note.setPlainText(self.note)


    def _load_image(self, img_path):
        if is_dicom(img_path) or utils.extract_ext(img_path) == ".dcm":
            try:
                dicom_data = DICOM(img_path)
                img = dicom_data.load_image()
            except Exception as e:
                self.reset_cursor()
                self.error_message(
                    self.tr(
                        "<p>Cannot open DICOM file.</p>"
                        "<p>Error Details:<br/>{}</p>"
                    ).format(e))
                self.setClean()
                return False
            else:
                # mff = MFF(img)
                # img = mff.run()
                # self.canvas.mff = mff
                self.dicom_data = dicom_data
                if self.canvas.wc == 0 and self.canvas.ww == 0:
                    self.canvas.wc = dicom_data.wc
                    self.canvas.ww = dicom_data.ww
                self.canvas.original_wc = dicom_data.wc
                self.canvas.original_ww = dicom_data.ww
                self.canvas.pixel_spacing = dicom_data.pixel_spacing[0]
                self.canvas.is_dicom = True

            if not self.canvas.loadPixmap(img, dicom_data.bits):
                self.reset_cursor()
                self.error_message(self.tr("<p>Cannot open image file.</p>"))
                self.setClean()
                return False
        elif utils.extract_ext(img_path) in EXTS:
            img = imread(img_path)
            if img is None:
                self.reset_cursor()
                self.error_message(self.tr("<p>Cannot load image file.</p>"))
                self.status(self.tr("Error reading {}").format(img_path))
                self.setClean()
                return False
            if not self.canvas.loadPixmap(img):
                self.reset_cursor()
                self.error_message(self.tr("<p>Cannot open image file.</p>"))
                self.setClean()
                return False
        else:
            return False

        return True


    def resize_event(self, event):
        if self.canvas and not self.canvas.image.isNull()\
           and self.zoom_mode != self.MANUAL_ZOOM:
            self.adjust_scale()
        super(MainWindow, self).resize_event(event)

    def paint_canvas(self):
        assert not self.canvas.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoom_widget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjust_scale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW_MODE
                             if initial else self.zoom_mode]()
        # print(value)
        value = int(100 * value)
        self.zoom_widget.setValue(value)
        self.zoom_level = value

    def scale_fit_window(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap"s aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        # print(w1, h1, a1, w2, h2, a2, self.canvas.pixmap.width(), self.centralWidget().width())
        return w1 / w2 if a2 >= a1 else h1 / h2

    def load_recent(self, filename):
        if self.may_continue_unsaved():
            self.load_file(filename)

    def open_prev_img(self, _value=False):
        if not self.may_continue_unsaved():
            return
        if len(self.image_list) <= 0:
            return
        if self.img_path is None:
            return
        currIndex = self.image_list.index(self.img_path)
        if currIndex - 1 >= 0:
            filename = self.image_list[currIndex - 1]
            if filename:
                self.load_file(filename)

    def open_next_img(self, _value=False, load=True):
        if not self.may_continue_unsaved():
            return
        
        if len(self.image_list) <= 0:
            self.resetState()
            self.canvas.setEnabled(False)
            return
        
        filename = None
        if self.img_path is None:
            filename = self.image_list[0]
        else:
            currIndex = self.image_list.index(self.img_path)
            if currIndex + 1 < len(self.image_list):
                filename = self.image_list[currIndex + 1]
            else:
                filename = self.image_list[-1]
        # self.img_path = filename
        if filename and load:
            self.load_file(filename)

    def open_file(self, _value=False):
        if not self.may_continue_unsaved():
            return
        default_dir = os.path.dirname(str(self.img_path)) if self.img_path else HOME_DIR

        # get extension filter
        _exts = [f"*{e}" for e in EXTS]
        _exts = " ".join(_exts)
        filter = self.tr("All files (*);;Image files ({})").format(_exts)

        # get file name
        while True:
            filename = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Choose image file"),
                default_dir,
                filter=filter,)
            filename = str(filename[0]).replace("/", os.sep)
            ext = os.path.splitext(filename)[1].lower()
            if len(filename) == 0:
                return
            elif len(ext) and ext == LabelFile.SUFFIX:  # if json files are selected
                _ret = False
                for ext in EXTS:
                    _img_path = os.path.splitext(filename)[0] + ext
                    if os.path.exists(_img_path):
                        filename = _img_path
                        _ret = True
                        break
                if _ret:
                    break
                else:
                    self.error_message(self.tr("The image file was not found."))
            elif not len(ext) and is_dicom(filename):   # dicom file has no ext
                break
            elif len(ext) and ext in EXTS:     # supported image file
                break
            else:
                self.error_message(self.tr("This file format is not supported."))

        # open directory
        target_path = utils.get_parent_path(filename)
        if target_path != self.work_dir:
            self.labels = {}
            self.import_from_dir(target_path)

        self.load_file(filename, update_list=True)

    def save_file(self):
        assert not self.canvas.image.isNull(), "cannot save empty image"

        # if self.has_labels():
        if self.labelFile:
            self._save_file(self.labelFile.lf_path)
        elif self.img_path:
            self._save_file(self.lf_path)
        else:
            self._save_file(self.save_file_dialog())
            
        self.add_label_data()
        self.update_state(EDIT)
        self.update_sum()
        return True


    def save_file_as(self, _value=False):
        assert not self.canvas.image.isNull(), "cannot save empty image"
        # if self.has_labels():
        self._save_file(self.save_file_dialog())

    def save_file_dialog(self):
        caption = self.tr("{} - Choose File").format(__appname__)
        filters = self.tr("Label files (*{})").format(LabelFile.SUFFIX)
        dlg = QtWidgets.QFileDialog(
            self, caption, self.current_path(), filters
        )
        dlg.setDefaultSuffix(LabelFile.SUFFIX[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = os.path.basename(os.path.splitext(self.img_path)[0])
        default_labelfile_name = os.path.join(
            self.current_path(), basename + LabelFile.SUFFIX
        )
        filename = dlg.getSaveFileName(
            self, self.tr("Choose File"), default_labelfile_name,
            self.tr("Label files (*{})").format(LabelFile.SUFFIX))
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _save_file(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def close_file(self, _value=False):
        if not self.may_continue_unsaved():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)
        self.input_note.setEnabled(False)
        self.button_start_timer.setEnabled(False)
        self.button_stop_timer.setEnabled(False)
        # self.input_editor.setEnabled(False)
        # self.input_checker.setEnabled(False)

    def delete_file(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr("You are about to permanently delete this label file, "
                      "proceed anyway?")
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return
        if os.path.exists(self.lf_path):
            os.remove(self.lf_path)
            item = self.fileListWidget.currentItem()
            if item:
                item.setCheckState(Qt.Unchecked)
            self.resetState()
            img_path = os.path.join(self.work_dir, item.text())
            self.load_file(img_path)
        self.update_sum()


    def has_labels(self):
        if self.noShapes():
            self.error_message(
                "You must label at least one object to save the file.")
            return False
        return True


    def has_label_file(self):
        if self.img_path is None:
            return False
        return os.path.exists(self.lf_path)
    

    def may_continue_unsaved(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr("Save annotations to '{}' before closing?").format(
            os.path.basename(self.img_path))
        answer = mb.question(self,
                             self.tr("Save annotations?"),
                             msg,
                             mb.Save | mb.Discard | mb.Cancel,
                             mb.Save)
        if answer == mb.Discard:
            self.dirty = False
            return True
        elif answer == mb.Save:
            if self.save_file():
                return True
            else:
                return False
        else:  # answer == mb.Cancel
            return False
    
    
    def may_continue_ai_running(self):
        if (self.ai_train_dialog.ai.isRunning() or
            self.ai_eval_dialog.ai.isRunning()):
            mb = QtWidgets.QMessageBox
            msg = self.tr("AI thread is running. Terminate?")
            answer = mb.question(self,
                                self.tr("AI thread is running!"),
                                msg,
                                mb.Ok, mb.Cancel)
            if answer == mb.Ok:
                # terminate processing
                if self.ai_train_dialog.ai.isRunning():
                    self.ai_train_dialog.ai.terminate()  # TODO:need more secure process
                if self.ai_eval_dialog.ai.isRunning():
                    self.ai_eval_dialog.ai.terminate()
                return True
            else:
                return False
        else:
            return True


    def may_continue(self, message="Continue?"):
        mb = QtWidgets.QMessageBox
        answer = mb.question(self,
                             self.tr("Confirmation"),
                             message,
                             mb.Yes | mb.No,
                             mb.Yes)
        if answer == mb.Yes:
            return True
        elif answer == mb.No:
            return False
        else:  # answer == mb.Cancel
            return False

    def error_message(self, message):
        return QtWidgets.QMessageBox.critical(
            self, self.tr("Error"),
            "<p>{}</p>".format(message))

    def info_message(self, message):
        return QtWidgets.QMessageBox.information(
            self, self.tr("Confirmation"),
            "<p>{}</p>".format(message))

    def current_path(self):
        return os.path.dirname(str(self.img_path)) if self.img_path else "."

    def toggle_keep_prev_mode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def delete_selected_shape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        if len(self.canvas.selectedShapes) == 1:
            shape = self.canvas.selectedShapes[0]
            msg = self.tr(
                "You are about to permanently delete '{}' polygon, "
                "proceed anyway?"
            ).format(shape.label)
            if yes == QtWidgets.QMessageBox.warning(
                    self, self.tr("Attention"), msg,
                    yes | no):
                self.remLabels(self.canvas.deleteSelected())
                self.setDirty()
        elif len(self.canvas.selectedShapes) > 1:
            msg = self.tr(
                "You are about to permanently delete polygon you selected, "
                "proceed anyway?"
            )
            if yes == QtWidgets.QMessageBox.warning(
                    self, self.tr("Attention"), msg,
                    yes | no):
                self.remLabels(self.canvas.deleteSelected())
                self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)
        self.labelWidget.disable_buttons()

    def copy_shape(self):
        self.canvas.endMove(copy=True)
        self.labelList.clearSelection()
        [self.add_label(shape) for shape in self.canvas.selectShapes]
        # self.labelList.sort()
        self.setDirty()

    def move_shape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()


    @property
    def image_list(self):
        count = self.fileListWidget.count()
        lst = [os.path.join(self.work_dir, self.fileListWidget.item(i).text())
            for i in range(count)]
        return lst


    def update_dir(self):
        self.labelSearch.setText('')
        self.import_from_workdir()


    def wait_cursor(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)


    def reset_cursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()


    def make_lfpath(self, img_path):
        # if utils.get_basedir(img_path) == 'image':
        #     parent_dir = os.path.dirname(img_path)
        #     lf_dir = os.path.join(os.path.dirname(parent_dir), 'annotation')
        #     img_name = utils.get_basename(img_path)
        #     return os.path.join(lf_dir, img_name) + LabelFile.SUFFIX
        # else:
        return os.path.splitext(img_path)[0] + LabelFile.SUFFIX


    def get_all_labels(self, lf: LabelFile):
        _labels = []
        for shape in lf.shapes:
            l = shape['label'].split('_')
            if l == ['']: # skip shapes have no labels
                continue
            else:
                _labels.append(l)
        return _labels
    

    def is_search(self):
        if self.target_label or self.target_name:
            return True
        return False


    def label_search(self, item_name):
        """If the target condition label hits, return True."""
        label_list = self.labels.get(item_name)
        for label in label_list:
            if utils.target_in_list(self.target_label, label):
                return True
        return False
        

    def import_from_dir(self, dirpath):
        if not os.path.exists(dirpath):
            self.error_message(self.tr("{} does not exists.").format(dirpath))
            return
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)
        if not self.may_continue_unsaved():
            return
        self.wait_cursor()

        # check whether can get image list
        img_list = self.scan_all_imgs(dirpath)
        if img_list is None or len(img_list) == 0:
            self.reset_cursor()
            self.error_message(self.tr("No images in the directory."))
            return
        
        self.canvas.reset_params()

        is_get_all_labels = False
        if len(self.labels) == 0:
            is_get_all_labels = True

        self.img_path = None
        self.fileListWidget.clear()
        self.count_images = 0
        self.count_annotations = 0

        for img_path in img_list:
            self.count_images += 1
            item_name = os.path.basename(img_path)

            if self.target_name and self.target_name not in item_name:
                continue
            if self.target_label and not self.labels.get(item_name):
                continue
            if self.target_label and self.labels.get(item_name) \
                and not self.label_search(item_name):
                # skip data when the target label is not in the annotation
                continue
                
            item = QtWidgets.QListWidgetItem(os.path.basename(img_path))
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            lf_path = self.make_lfpath(img_path)
            lf = None
            try:
                lf = LabelFile()
                lf.load(lf_path)
            except Exception as e:
                lf = None
            if lf is None:
                item.setCheckState(Qt.Unchecked)
                self.fileListWidget.addItem(item)
            elif lf is None and self.is_search(): # skip no data while searching
                continue
            else:
                if is_get_all_labels:
                    self.labels[item_name] = self.get_all_labels(lf)
                self.count_annotations += 1
                item.setCheckState(Qt.PartiallyChecked)
                item.setBackground(STATE_COLORS[EDIT])
                self.fileListWidget.addItem(item)
        
        self.work_dir = dirpath
        self.update_sum(only_text=True)
        # self.open_next_img(load=load)
        self.update_ai_select()
        self.resetState()
        self.fileListWidget.setCurrentRow(0)
        self.fileListWidget.repaint()

        self.reset_cursor()


    def import_from_workdir(self):
        self.import_from_dir(self.work_dir)


    def scan_all_imgs(self, dirpath=None):
        number_sort = True
        img_paths = []
        target_dir = dirpath if dirpath is not None else self.work_dir
        for fp in glob(os.path.join(target_dir, '*')):
            if is_dicom(fp) or utils.extract_ext(fp) in EXTS:
                img_paths.append(fp)
                key = re.sub(r"\D", "", os.path.basename(fp))
                if key == "":
                    number_sort = False
            # if fname.lower().endswith(tuple(extensions)):
            #     img_paths.append(fp)
                # key = re.search(r'\d+', os.path.basename(fp))
                # if key is None:
                    # number_sort = False
        img_paths.sort(key=lambda x: x.lower())
        if number_sort:
            # img_paths.sort(key=lambda s: int(re.search(r'\d+', os.path.basename(s)).group()))
            img_paths.sort(key=lambda s: int(re.sub(r"\D", "", os.path.basename(s))))
        return img_paths


    def update_note(self):
        if self.img_path is None:
            return
        self.note = self.input_note.toPlainText()
        self.setDirty()

    
    def update_state(self, state):
        row = self.fileListWidget.currentRow()
        item = self.fileListWidget.item(row)
        if state == NO_DATA:
            self.input_note.setEnabled(True)
            if item:
                item.setCheckState(Qt.Unchecked)
                item.setBackground(STATE_COLORS[NO_DATA])
        elif state == EDIT:
            self.input_note.setEnabled(True)
            if item:
                item.setCheckState(Qt.PartiallyChecked)
                item.setBackground(STATE_COLORS[EDIT])


    def update_sum(self, only_text=False):
        if not only_text:
            self.count_images = 0
            self.count_annotations = 0
            for img_path in self.scan_all_imgs():
                self.count_images += 1
                lf_path = self.make_lfpath(img_path)
                if QtCore.QFile.exists(lf_path) and LabelFile.is_label_file(lf_path):
                    self.count_annotations += 1
        text = self.tr("[Images Total] {}").format(self.count_images)
        self.label_sum.setText(text)
        text = self.tr("[Annotations Total] {}").format(self.count_annotations)
        self.label_sum_edit.setText(text)

    
    def update_dicom_info(self):
        if self.dicom_data is None:
            self.label_dicom_names.setText("")
            self.label_dicom_values.setText("")
        else:
            t1, t2 = self.dicomDialog.get_info_as_text(self.dicom_data)
            self.label_dicom_names.setText(t1)
            self.label_dicom_values.setText(t2)


    def timer_start(self):
        self.timer.restart()   
        self.interrupt_timer.start()
        self.button_start_timer.setEnabled(False)
        self.button_stop_timer.setEnabled(True)


    def timer_stop(self):
        self.interrupt_timer.stop()
        etime = self.timer.elapsed() / 1000
        self.update_label_timer(etime)
        self.elapsed_time = etime
        self.setDirty()
        self.button_start_timer.setEnabled(True)
        self.button_stop_timer.setEnabled(False)


    def timer_update(self):
        etime = self.timer.elapsed() / 1000
        self.update_label_timer(etime)
    

    def update_label_timer(self, etime):
        self.label_timer.setText(f"{etime:.2f} [s]")


    def ai_train_popup(self):
        if self.is_submode:    
            target_dir = utils.get_parent_path(self.work_dir)
        else:
            target_dir = self.work_dir

        try:
            self.ai_train_dialog.popup(target_dir,
                                       self.is_submode,
                                       self.labelWidget.label_def)
        except Exception as e:
            self.error_message(e)
            return
            
        # get all labels
        # self.wait_cursor()
        # dir_list = glob(os.path.join(parent_dir, "**"))
        
        # for subdir_path in dir_list:
        #     if utils.get_basename(subdir_path) == AI_DIR_NAME:
        #         continue
        #     subdir_jsons = glob(os.path.join(subdir_path, "*.json"))
        #     for p in subdir_jsons:
        #         lf = LabelFile()
        #         lf.load(p)
        #         if lf.shapes is None:
        #             continue
        #         _labels = self.get_all_labels(lf)
        #         _labels = sum(_labels, [])
        #         labels.extend(_labels)
        # labels = sorted(list(set(labels)))
        # self.reset_cursor()

        # get all labels
        # self.wait_cursor()
        # for p in glob(os.path.join(self.work_dir, "*.json")):
        #     lf = LabelFile()
        #     lf.load(p)
        #     if lf.shapes is None:
        #         continue
        #     _labels = self.get_all_labels(lf)
        #     _labels = sum(_labels, [])
        #     labels.extend(_labels)
        # labels = sorted(list(set(labels)))
        # self.reset_cursor()

    def ai_eval_popup(self):
        if self.is_submode:
            parent_dir = utils.get_parent_path(self.work_dir)
            self.ai_eval_dialog.popup(parent_dir)
        else:
            self.ai_eval_dialog.popup(self.work_dir)


    def update_ai_select(self):
        self.ai_select.clear()
        if not os.path.exists(PRETRAINED_DIR):
            self.button_ai_test.setEnabled(False)
            self.ai_select.setEnabled(False)
            return
        
        # get directory names in data directory
        targets = [name for name in os.listdir(PRETRAINED_DIR) if os.path.isdir(os.path.join(PRETRAINED_DIR, name))]

        for t in targets:
            logdir = os.path.join(PRETRAINED_DIR, t)
            config_path = os.path.join(logdir, "config.json")
            onnx_path = os.path.join(logdir, "model.onnx")
            if (os.path.exists(config_path) and
                os.path.exists(onnx_path)):
                self.ai_select.addItem(t)
        
        if self.ai_select.count() < 1:
            self.button_ai_test.setEnabled(False)
            self.ai_select.setEnabled(False)
        else:
            self.button_ai_test.setEnabled(True)
            self.ai_select.setEnabled(True)
            if self.text_ai_select is not None:
                idx = self.ai_select.findText(self.text_ai_select)
                if idx > -1:
                    self.ai_select.setCurrentIndex(idx)


    def ai_test(self):
        """Generate annotations with AI prediction results."""
        if not self.img_path:
            return
        
        if not self.model_dir or not os.path.exists(self.model_dir):
            self.error_message(self.tr("AI Model was not found."))
            return

        if not self.noShapes():
            message = self.tr("Are you sure you want to overwrite annotations?")
            answer = self.may_continue(message=message)
            if not answer:
                return

        self.wait_cursor()
        self.status(self.tr("AI Testing ..."))
        self.setEnabled(False)

        try:
            shapes = self.ai_test_widget.generate_shapes(
                self.canvas.img_array,
                self.model_dir,
                self.approx_epsilon,
                self.area_limit)
        except Exception as e:
            self.reset_cursor()
            self.error_message(e)
            self.setEnabled(True)
            return
        else:
            if len(shapes):
                self.labelList.clear()
                self.labelWidget.reset()
                self.loadLabels(shapes)
                self.setDirty()
                self.reset_cursor()
                self.setEnabled(True)
                return
            else:
                self.reset_cursor()
                self.info_message(self.tr("No detections."))
                self.setEnabled(True)
                return
            
    def callback_ai_train_running(self, is_running):
        if is_running:
            self.button_ai_eval.setEnabled(False)
            self.input_is_submode.setEnabled(False)
        else:
            self.button_ai_eval.setEnabled(True)
            self.input_is_submode.setEnabled(True)
    
    def callback_ai_eval_running(self, is_running):
        if is_running:
            self.button_ai_train.setEnabled(False)
            self.input_is_submode.setEnabled(False)
        else:
            self.button_ai_train.setEnabled(True)
            self.input_is_submode.setEnabled(True)

    def export_annotations(self):
        # confirm unsaved annotations
        if not self.may_continue_unsaved():
            return
        
        # open directory dialog
        target_dir = self.select_dir_dialog(HOME_DIR)
        if target_dir is None:
            return
        
        pd = CopyAnnotationsDialog(self, self.work_dir, target_dir, self.is_submode)
        pd.popup()

        self.info_message(self.tr("Exported annotation files to {}").format(target_dir))
        return


    def import_model(self):
        if self.work_dir is None:
            return
        
        # open directory dialog
        target_dir = self.select_dir_dialog(HOME_DIR, custom_text=self.tr("Select Pretrained Model Directory"))
        if target_dir is None:
            return

        config_path = os.path.join(target_dir, "config.json")
        onnx_path = os.path.join(target_dir, "model.onnx")

        if not os.path.exists(config_path) :
            self.error_message(self.tr("{} does not exists.").format(config_path))
            return
        
        if not os.path.exists(onnx_path):
            self.error_message(self.tr("{} does not exists.").format(onnx_path))
            return
        
        self.wait_cursor()
        dir_path = os.path.join(PRETRAINED_DIR, os.path.basename(target_dir))
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        try:
            shutil.copy(config_path, dir_path)
            shutil.copy(onnx_path, dir_path)
        except shutil.SameFileError: # if src and dist are same.
            self.reset_cursor()
            self.error_message(self.tr(
                """The same model has already exists."""
            ))
            return

        self.reset_cursor()
        
        self.info_message(self.tr("Imported {}").format(target_dir))
        self.update_ai_select()
        return
    

    def select_dir_dialog(self, default_dir=None, custom_text=None):
        opendir = HOME_DIR
        if default_dir is not None:
            opendir = default_dir
        
        opendir = HOME_DIR
        target_path = str(QtWidgets.QFileDialog.getExistingDirectory(
            self,
            custom_text if custom_text else self.tr("Select Directory"),
            opendir,
            QtWidgets.QFileDialog.ShowDirsOnly |
            QtWidgets.QFileDialog.DontResolveSymlinks))
        if not target_path:
            return None
        target_path = target_path.replace('/', os.sep)
        return target_path


    def export_canvas_img(self):
        opendir = HOME_DIR if self.prev_dir is None else self.prev_dir
        target_path = self.select_dir_dialog(opendir, self.tr("Select Output Directory"))
        if target_path is None:
            return
        self.prev_dir = target_path

        save_path = os.path.join(target_path, utils.get_basename(self.img_path) + ".png")
        save_canvas_img(self.canvas.img_array, save_path)


    def delete_pretrained_model(self):
        r = self.may_continue(self.tr(
            """Are you sure you want to delete it?"""
        ))
        if not r:
            return

        for target_path in glob(os.path.join(PRETRAINED_DIR, "*")):
            shutil.rmtree(target_path)
        self.update_ai_select()
