from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from aidia import qt
from aidia.dicom import DICOM

class DICOMDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint)

        layout = QtWidgets.QVBoxLayout()

        self.text1 = QtWidgets.QLabel()
        self.text2 = QtWidgets.QLabel()
        sub_layout = QtWidgets.QHBoxLayout()
        sub_layout.addWidget(self.text1, alignment=Qt.AlignLeft)
        sub_layout.addWidget(self.text2, alignment=Qt.AlignLeft)
        sub_widget = QtWidgets.QWidget()
        sub_widget.setLayout(sub_layout)
        layout.addWidget(sub_widget)

        # accept and reject button
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok, QtCore.Qt.Horizontal, self)
        bb.button(bb.Ok).setIcon(qt.newIcon('done'))
        bb.accepted.connect(self.accept)
        layout.addWidget(bb)

        self.setLayout(layout)

    
    def popUp(self, dicom:DICOM):
            tags, names, values = dicom.get_info(self.target_names)
            t1 = ""
            t2 = ""
            for name, value in zip(names, values):
                t1 += f"{name}\n"
                t2 += f"{value}\n"
            self.text1.setText(t1)
            self.text2.setText(t2)
            if self.exec_():
                return True
            else:
                return False


    def get_info_as_text(self, d:DICOM):
            t1 = ""
            t2 = ""

            t1 += self.tr("PatientID") + "\n"
            t1 += self.tr("Modality") + "\n"
            t1 += self.tr("Manufacturer") + "\n"
            t1 += self.tr("InstanceNumber") + "\n"
            t1 += self.tr("Rows") + "\n"
            t1 += self.tr("Columns") + "\n"
            t1 += self.tr("BitsAllocated") + "\n"
            t1 += self.tr("PixelSpacing x") + "\n"
            t1 += self.tr("PixelSpacing y") + "\n"
            t1 += self.tr("ImagePositionPatient x") + "\n"
            t1 += self.tr("ImagePositionPatient y") + "\n"
            t1 += self.tr("ImagePositionPatient z") + "\n"
            t1 += self.tr("SliceThickness") + "\n"
            t1 += self.tr("SpacingBetweenSlices") + "\n"
            t1 += self.tr("RescaleSlope") + "\n"
            t1 += self.tr("RescaleIntercept") + "\n"
            t1 += self.tr("WindowCenter") + "\n"
            t1 += self.tr("WindowWidth") + "\n"
            t1 += self.tr("PhotometricInterpretation")

            t2 += f"{d.id}\n"
            t2 += d.modality + "\n"
            t2 += d.manufacturer + "\n"
            t2 += str(d.slice_number) + "\n"
            t2 += str(d.rows) + "\n"
            t2 += str(d.columns) + "\n"
            t2 += str(d.bits) + "\n"
            x, y = d.pixel_spacing
            t2 += f"{x:.4f} mm" + "\n"
            t2 += f"{y:.4f} mm" + "\n"
            x, y, z = d.image_pos
            t2 += f"{x:.4f} mm" + "\n" 
            t2 += f"{y:.4f} mm" + "\n" 
            t2 += f"{z:.4f} mm" + "\n" 
            t2 += f"{d.slice_thickness:.4f} mm" + "\n"
            t2 += f"{d.slice_spacing:.4f} mm" + "\n"
            t2 += str(d.rescale_slope) + "\n"
            t2 += str(d.rescale_intercept) + "\n"
            t2 += f"{int(d.wc):d}" + "\n"
            t2 += f"{int(d.ww):d}" + "\n"
            t2 += str(d.photo_interpret)
    
            return t1, t2