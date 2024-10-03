import pydicom
import numpy as np
import os

def is_dicom(path):
    """Return True if the path is DICOM file by pydicom fucntion."""
    try:
        res = pydicom.misc.is_dicom(path)
    except Exception as e:
        # print(e)
        return False
    else:
        return res


class DICOM():

    def __init__(self, param) -> None:
        self.data = None
        self.path = None

        if isinstance(param, str):
            self.load_data_from_path(param)
        else:
            self.load_data(param)
        
        if self.data is not None:
            self.wc = 2048
            if hasattr(self.data, "WindowCenter"):
                wc = self.data.WindowCenter
                if isinstance(wc, pydicom.multival.MultiValue):
                    self.wc = wc[0]
                else:
                    self.wc = wc
            
            self.ww = 4096
            if hasattr(self.data, "WindowWidth"):
                ww = self.data.WindowWidth
                if isinstance(ww, pydicom.multival.MultiValue):
                    self.ww = ww[0]
                else:
                    self.ww = ww

            self.id = 0
            if hasattr(self.data, "PatientID"):
                self.id = str(self.data.PatientID)
            
            self.modality = ""
            if hasattr(self.data, "Modality"):
                self.modality = self.data.Modality
            
            self.manufacturer = ""
            if hasattr(self.data, "Manufacturer"):
                self.manufacturer = self.data.Manufacturer
            
            self.rows = 0
            if hasattr(self.data, "Rows"):
                self.rows = self.data.Rows
            
            self.columns = 0
            if hasattr(self.data, "Columns"):
                self.columns = self.data.Columns
            
            self.bits = 0
            if hasattr(self.data, "BitsAllocated"):
                self.bits = self.data.BitsAllocated
            
            self.slice_number = 0
            if hasattr(self.data, "InstanceNumber"):
                self.slice_number = self.data.InstanceNumber
            
            self.image_pos = [0, 0, 0]
            if hasattr(self.data, "ImagePositionPatient"):
                self.image_pos = self.data.ImagePositionPatient
            
            self.slice_thickness = 0
            if hasattr(self.data, "SliceThickness"):
                self.slice_thickness = self.data.SliceThickness
            
            self.slice_spacing = self.slice_thickness
            if hasattr(self.data, "SpacingBetweenSlices"):
                self.slice_spacing = self.data.SpacingBetweenSlices
            
            self.pixel_spacing = [0, 0]
            if hasattr(self.data, "PixelSpacing"):
                self.pixel_spacing = self.data.PixelSpacing

            self.rescale_slope = 1
            if hasattr(self.data, "RescaleSlope"):
                self.rescale_slope = self.data.RescaleSlope

            self.rescale_intercept = 0
            if hasattr(self.data, "RescaleIntercept"):
                self.rescale_intercept = self.data.RescaleIntercept

            self.photo_interpret = 1
            if hasattr(self.data, "PhotometricInterpretation"):
                self.photo_interpret = self.data.PhotometricInterpretation


    def load_data(self, data):
        self.data = data

    
    def load_data_from_path(self, path):
        if os.path.exists(path):
            self.path = path
        else:
            raise FileNotFoundError('{} is not found.'.format(path))
        
        try:
            self.data = pydicom.dcmread(path, force=True)
        except Exception as e:
            print(e)
            self.data = None


    def load_image(self):
        if not hasattr(self.data.file_meta, 'TransferSyntaxUID'):
            self.data.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        img = self.data.pixel_array.astype(float)

        if not len(img):
            raise ValueError('Error in loading image')
        
        if self.rescale_intercept is not None and self.rescale_slope is not None:
            img = self.rescale_slope * img + self.rescale_intercept
        
        if self.photo_interpret is not None and self.photo_interpret == 'MONOCHROME1':
            img = img - np.min(img) # slide min pixel to 0
            img = np.max(img) - img # flip pixels
            img = img + np.min(img) # to original
        
        return img

    
    def get_info(self):
        tags = []
        names = []
        values = []
        for elem in self.data:
            tags.append(elem.tag)
            names.append(elem.name)
            values.append(elem.repval)
        return tags, names, values


    