import os
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt

from aidia import qt
from aidia import CLEAR, ERROR
from aidia import ModelTypes
from aidia.ai.config import AIConfig


class AIAugmentDialog(QtWidgets.QDialog):
    """Dialog for configuring AI data augmentation parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(Qt.Window
                            | Qt.CustomizeWindowHint
                            | Qt.WindowTitleHint
                            | Qt.WindowCloseButtonHint)
        self.setWindowTitle(self.tr("Advanced Data Augmentation Settings"))
        self.setMinimumSize(QtCore.QSize(500, 600))

        self.config = None

        # Create main layout
        self._layout = QtWidgets.QVBoxLayout()

        # Title
        title = qt.head_text(self.tr("Advanced Data Augmentation Settings"))
        title.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self._layout.addWidget(title)

        # Description
        description = QtWidgets.QLabel(self.tr(
            "Fine-tune data augmentation factors. These settings control the intensity "
            "of random transformations applied to training images to improve model generalization."
        ))
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet("color: gray; margin: 10px;")
        self._layout.addWidget(description)

        # Scroll area for parameters
        scroll_area = QtWidgets.QScrollArea()
        scroll_widget = QtWidgets.QWidget()
        self.param_layout = QtWidgets.QFormLayout()

        # Initialize parameter widgets
        self.init_parameter_widgets()

        scroll_widget.setLayout(self.param_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        self._layout.addWidget(scroll_area)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()

        # Reset to defaults button
        self.button_reset = QtWidgets.QPushButton(self.tr("Reset to Defaults"))
        self.button_reset.setAutoDefault(False)
        self.button_reset.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.button_reset)

        button_layout.addStretch()

        # OK and Cancel buttons
        self.button_ok = QtWidgets.QPushButton(self.tr("OK"))
        self.button_ok.clicked.connect(self.accept_changes)
        self.button_cancel = QtWidgets.QPushButton(self.tr("Cancel"))
        self.button_cancel.clicked.connect(self.reject)

        button_layout.addWidget(self.button_ok)
        button_layout.addWidget(self.button_cancel)

        self._layout.addLayout(button_layout)
        self.setLayout(self._layout)

    def init_parameter_widgets(self):
        """Initialize all parameter input widgets."""
        
        # Rotation parameter
        self.rotation_spinbox = QtWidgets.QDoubleSpinBox()
        self.rotation_spinbox.setRange(0.0, 0.5)
        self.rotation_spinbox.setSingleStep(0.1)
        self.rotation_spinbox.setDecimals(2)
        self.rotation_spinbox.setToolTip(self.tr("Random rotation factor"))
        self.param_layout.addRow(self.tr("Rotation Factor:"), self.rotation_spinbox)

        # Scale parameter
        self.scale_spinbox = QtWidgets.QDoubleSpinBox()
        self.scale_spinbox.setRange(0.0, 0.5)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setDecimals(2)
        self.scale_spinbox.setToolTip(self.tr("Random scale variation factor"))
        self.param_layout.addRow(self.tr("Scale Factor:"), self.scale_spinbox)

        # Shift parameter
        self.shift_spinbox = QtWidgets.QDoubleSpinBox()
        self.shift_spinbox.setRange(0.0, 0.5)
        self.shift_spinbox.setSingleStep(0.1)
        self.shift_spinbox.setDecimals(2)
        self.shift_spinbox.setToolTip(self.tr("Random translation factor"))
        self.param_layout.addRow(self.tr("Shift Factor:"), self.shift_spinbox)

        # Shear parameter
        self.shear_spinbox = QtWidgets.QDoubleSpinBox()
        self.shear_spinbox.setRange(0.0, 0.5)
        self.shear_spinbox.setSingleStep(0.1)
        self.shear_spinbox.setDecimals(2)
        self.shear_spinbox.setToolTip(self.tr("Random shear factor"))
        self.param_layout.addRow(self.tr("Shear Factor:"), self.shear_spinbox)

        # Brightness parameter
        self.brightness_spinbox = QtWidgets.QDoubleSpinBox()
        self.brightness_spinbox.setRange(0.0, 0.5)
        self.brightness_spinbox.setSingleStep(0.1)
        self.brightness_spinbox.setDecimals(2)
        self.brightness_spinbox.setToolTip(self.tr("Random brightness adjustment factor"))
        self.param_layout.addRow(self.tr("Brightness Factor:"), self.brightness_spinbox)

        # Contrast parameter
        self.contrast_spinbox = QtWidgets.QDoubleSpinBox()
        self.contrast_spinbox.setRange(0.0, 0.5)
        self.contrast_spinbox.setSingleStep(0.1)
        self.contrast_spinbox.setDecimals(2)
        self.contrast_spinbox.setToolTip(self.tr("Random contrast adjustment factor"))
        self.param_layout.addRow(self.tr("Contrast Factor:"), self.contrast_spinbox)

        # Blur parameter
        self.blur_spinbox = QtWidgets.QDoubleSpinBox()
        self.blur_spinbox.setRange(0.0, 0.5)
        self.blur_spinbox.setSingleStep(0.1)
        self.blur_spinbox.setDecimals(2)
        self.blur_spinbox.setToolTip(self.tr("Random blur factor"))
        self.param_layout.addRow(self.tr("Blur Factor:"), self.blur_spinbox)

        # Noise parameter
        self.noise_spinbox = QtWidgets.QDoubleSpinBox()
        self.noise_spinbox.setRange(0.0, 0.5)
        self.noise_spinbox.setSingleStep(0.1)
        self.noise_spinbox.setDecimals(2)
        self.noise_spinbox.setToolTip(self.tr("Random noise factor"))
        self.param_layout.addRow(self.tr("Noise Factor:"), self.noise_spinbox)

    def load_config(self, config: AIConfig):
        """Load configuration values into the dialog."""
        self.config = config

        # Load numeric parameters
        self.rotation_spinbox.setValue(config.RANDOM_ROTATE)
        self.scale_spinbox.setValue(config.RANDOM_SCALE)
        self.shift_spinbox.setValue(config.RANDOM_SHIFT)
        self.shear_spinbox.setValue(config.RANDOM_SHEAR)
        self.brightness_spinbox.setValue(config.RANDOM_BRIGHTNESS)
        self.contrast_spinbox.setValue(config.RANDOM_CONTRAST)
        self.blur_spinbox.setValue(config.RANDOM_BLUR)
        self.noise_spinbox.setValue(config.RANDOM_NOISE)

        # Disable contrast, blur, and noise for Ultralytics models
        is_ultralytics = config.is_ultralytics()
        self.contrast_spinbox.setEnabled(not is_ultralytics)
        self.blur_spinbox.setEnabled(not is_ultralytics)
        self.noise_spinbox.setEnabled(not is_ultralytics)
        
        # Update tooltips for disabled parameters
        if is_ultralytics:
            disabled_tooltip = self.tr("This parameter is not supported for Ultralytics models")
            self.contrast_spinbox.setToolTip(disabled_tooltip)
            self.blur_spinbox.setToolTip(disabled_tooltip)
            self.noise_spinbox.setToolTip(disabled_tooltip)

    def save_config(self):
        """Save dialog values back to the configuration."""
        if self.config is None:
            return

        # Save numeric parameters
        self.config.RANDOM_ROTATE = self.rotation_spinbox.value()
        self.config.RANDOM_SCALE = self.scale_spinbox.value()
        self.config.RANDOM_SHIFT = self.shift_spinbox.value()
        self.config.RANDOM_SHEAR = self.shear_spinbox.value()
        self.config.RANDOM_BRIGHTNESS = self.brightness_spinbox.value()
        
        # For Ultralytics models, force contrast, blur, and noise to 0
        is_ultralytics = self.config.is_ultralytics()
        self.config.RANDOM_CONTRAST = 0.0 if is_ultralytics else self.contrast_spinbox.value()
        self.config.RANDOM_BLUR = 0.0 if is_ultralytics else self.blur_spinbox.value()
        self.config.RANDOM_NOISE = 0.0 if is_ultralytics else self.noise_spinbox.value()

    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        # Default values based on AIConfig initialization
        self.rotation_spinbox.setValue(0.1)
        self.scale_spinbox.setValue(0.1)
        self.shift_spinbox.setValue(0.1)
        self.shear_spinbox.setValue(0.1)
        self.brightness_spinbox.setValue(0.1)
        
        # Check if current model is Ultralytics
        is_ultralytics = self.config.is_ultralytics()
        
        # Set values and enable/disable based on model type
        contrast_value = 0.0 if is_ultralytics else 0.1
        blur_value = 0.0 if is_ultralytics else 0.1
        noise_value = 0.0 if is_ultralytics else 0.1
        
        self.contrast_spinbox.setValue(contrast_value)
        self.blur_spinbox.setValue(blur_value)
        self.noise_spinbox.setValue(noise_value)
        
        self.contrast_spinbox.setEnabled(not is_ultralytics)
        self.blur_spinbox.setEnabled(not is_ultralytics)
        self.noise_spinbox.setEnabled(not is_ultralytics)

    def accept_changes(self):
        """Accept changes and close dialog."""
        self.save_config()
        self.accept()

    def popup(self, config: AIConfig):
        """Show the dialog with the given configuration."""
        self.load_config(config)
        return self.exec_()
