import json
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Qt

from aidia import qt
from aidia import aidia_logger


class AILabelReplaceDialog(QtWidgets.QDialog):
    """Label replacement settings dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(self.tr("Label Replacement Settings"))
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | 
                           Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setMinimumSize(QtCore.QSize(600, 400))
        
        self.replace_dict = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI"""
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        title = qt.head_text(self.tr("Label Replacement Settings"))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        description = QtWidgets.QLabel(
            self.tr("Enter old labels (left) and new labels (right), one per line.\n"
                   "The number of lines must match between both text boxes.")
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: gray; font-size: 12px; margin: 10px;")
        layout.addWidget(description)
        
        # Text editor layout
        text_layout = QtWidgets.QHBoxLayout()
        
        # Left side (original labels)
        left_layout = QtWidgets.QVBoxLayout()
        left_label = QtWidgets.QLabel(self.tr("Original Labels"))
        left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(left_label)
        
        self.old_labels_edit = QtWidgets.QTextEdit()
        self.old_labels_edit.setPlaceholderText(
            self.tr("old_label1\nold_label2\nold_label3")
        )
        self.old_labels_edit.setMinimumHeight(200)
        left_layout.addWidget(self.old_labels_edit)
        
        # Arrow
        arrow_layout = QtWidgets.QVBoxLayout()
        arrow_layout.addStretch()
        arrow_label = QtWidgets.QLabel("→")
        arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        arrow_layout.addWidget(arrow_label)
        arrow_layout.addStretch()
        
        # Right side (new labels)
        right_layout = QtWidgets.QVBoxLayout()
        right_label = QtWidgets.QLabel(self.tr("New Labels"))
        right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(right_label)
        
        self.new_labels_edit = QtWidgets.QTextEdit()
        self.new_labels_edit.setPlaceholderText(
            self.tr("new_label1\nnew_label2\nnew_label3")
        )
        self.new_labels_edit.setMinimumHeight(200)
        right_layout.addWidget(self.new_labels_edit)
        
        text_layout.addLayout(left_layout)
        text_layout.addLayout(arrow_layout)
        text_layout.addLayout(right_layout)
        layout.addLayout(text_layout)
        
        # Validation status display
        self.status_label = QtWidgets.QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        # Validate button
        # validate_button = QtWidgets.QPushButton(self.tr("Validate"))
        # validate_button.clicked.connect(self.validate_labels)
        # button_layout.addWidget(validate_button)
        
        # Clear button
        clear_button = QtWidgets.QPushButton(self.tr("Clear"))
        clear_button.clicked.connect(self.clear_text)
        button_layout.addWidget(clear_button)
        
        button_layout.addStretch()
        
        # OK/Cancel buttons
        ok_button = QtWidgets.QPushButton(self.tr("OK"))
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)
        
        cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Validate on text change
        self.old_labels_edit.textChanged.connect(self.on_text_changed)
        self.new_labels_edit.textChanged.connect(self.on_text_changed)
        
    def on_text_changed(self):
        """Process when text is changed"""
        old_text = self.old_labels_edit.toPlainText().strip()
        new_text = self.new_labels_edit.toPlainText().strip()
        
        if not old_text and not new_text:
            self.status_label.setText("")
            self.status_label.setStyleSheet("")
            return
        
        old_labels = [label.strip() for label in old_text.split('\n') if label.strip()]
        new_labels = [label.strip() for label in new_text.split('\n') if label.strip()]
        
        # Check line count
        if len(old_labels) != len(new_labels):
            self.status_label.setText(
                self.tr("✗ Line count mismatch: {} old labels, {} new labels").format(
                    len(old_labels), len(new_labels)
                )
            )
            self.status_label.setStyleSheet("color: red;")
        elif len(old_labels) == 0:
            self.status_label.setText(self.tr("✗ No labels entered"))
            self.status_label.setStyleSheet("color: red;")
        else:
            # Check for duplicates
            if len(set(old_labels)) != len(old_labels):
                self.status_label.setText(self.tr("✗ Duplicate old labels found"))
                self.status_label.setStyleSheet("color: red;")
            else:
                self.status_label.setText(
                    self.tr("✓ {} replacement rules ready").format(len(old_labels))
                )
                self.status_label.setStyleSheet("color: green;")
    
    def validate_labels(self):
        """Validate labels"""
        old_text = self.old_labels_edit.toPlainText().strip()
        new_text = self.new_labels_edit.toPlainText().strip()
        
        if not old_text or not new_text:
            QtWidgets.QMessageBox.warning(
                self, 
                self.tr("Warning"), 
                self.tr("Please enter both old and new labels.")
            )
            return False
        
        old_labels = [label.strip() for label in old_text.split('\n') if label.strip()]
        new_labels = [label.strip() for label in new_text.split('\n') if label.strip()]
        
        # Check line count
        if len(old_labels) != len(new_labels):
            QtWidgets.QMessageBox.warning(
                self, 
                self.tr("Warning"), 
                self.tr("The number of old labels ({}) must match the number of new labels ({}).").format(
                    len(old_labels), len(new_labels)
                )
            )
            return False
        
        # Check for empty labels
        if len(old_labels) == 0:
            QtWidgets.QMessageBox.warning(
                self, 
                self.tr("Warning"), 
                self.tr("Please enter at least one label pair.")
            )
            return False
        
        # Check for duplicates
        if len(set(old_labels)) != len(old_labels):
            duplicates = [label for label in set(old_labels) if old_labels.count(label) > 1]
            QtWidgets.QMessageBox.warning(
                self, 
                self.tr("Warning"), 
                self.tr("Duplicate old labels found: {}").format(", ".join(duplicates))
            )
            return False
        
        # Check for empty strings
        empty_old = [i for i, label in enumerate(old_labels) if not label]
        empty_new = [i for i, label in enumerate(new_labels) if not label]
        
        if empty_old or empty_new:
            QtWidgets.QMessageBox.warning(
                self, 
                self.tr("Warning"), 
                self.tr("Empty labels are not allowed.")
            )
            return False
        
        QtWidgets.QMessageBox.information(
            self, 
            self.tr("Success"), 
            self.tr("Label validation successful!\n{} replacement rules are valid.").format(len(old_labels))
        )
        return True
    
    def clear_text(self):
        """Clear text"""
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("Confirm"),
            self.tr("Clear all text?"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.old_labels_edit.clear()
            self.new_labels_edit.clear()
    
    def get_replace_dict(self):
        """Get replacement dictionary"""
        old_text = self.old_labels_edit.toPlainText().strip()
        new_text = self.new_labels_edit.toPlainText().strip()
        
        if not old_text or not new_text:
            return {}
        
        old_labels = [label.strip() for label in old_text.split('\n') if label.strip()]
        new_labels = [label.strip() for label in new_text.split('\n') if label.strip()]
        
        # Return empty dictionary if line counts don't match
        if len(old_labels) != len(new_labels):
            return {}
        
        # Create dictionary
        replace_dict = {}
        for old_label, new_label in zip(old_labels, new_labels):
            if old_label and new_label:  # Skip empty strings
                replace_dict[old_label] = new_label
        
        return replace_dict
    
    def set_replace_dict(self, replace_dict):
        """Set replacement dictionary"""
        if isinstance(replace_dict, dict) and replace_dict:
            old_labels = list(replace_dict.keys())
            new_labels = list(replace_dict.values())
            
            self.old_labels_edit.setPlainText('\n'.join(old_labels))
            self.new_labels_edit.setPlainText('\n'.join(new_labels))
        else:
            self.old_labels_edit.clear()
            self.new_labels_edit.clear()
    
    def accept(self):
        """Process when OK button is pressed"""
        old_text = self.old_labels_edit.toPlainText().strip()
        new_text = self.new_labels_edit.toPlainText().strip()
        
        # Accept as is if nothing is entered
        if not old_text and not new_text:
            self.replace_dict = {}
            super().accept()
            return
        
        # Validate if there is input
        if old_text or new_text:
            if not self.validate_labels():
                return  # Don't close if validation fails
        
        self.replace_dict = self.get_replace_dict()
        super().accept()
    
    def popup(self, replace_dict=None):
        """Show dialog"""
        if replace_dict:
            self.set_replace_dict(replace_dict)
        
        return self.exec_()
