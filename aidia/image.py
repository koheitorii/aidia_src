import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from aidia import LABEL_COLORMAP
from aidia import dicom
from aidia import utils
from qtpy import QtGui

EXTS = [".{}".format(fmt.data().decode("ascii").lower()) for fmt in QtGui.QImageReader.supportedImageFormats()]

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    """Load a RGB image with OpenCV.
    
    This function supports the file name including 2-bytes codes.
    """
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        return None
    
def imwrite(img, filename):
    """ Write an image with OpenCV.
    
    This function supports the file name including 2-bytes codes.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    is_success, im_buf_arr = cv2.imencode(".png", img)
    if is_success:
        im_buf_arr.tofile(filename)
    return is_success

def read_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Failed to load {img_path}")

    if dicom.is_dicom(img_path) or utils.extract_ext(img_path) == ".dcm":
        dicom_data = dicom.DICOM(img_path)
        img = dicom_data.load_image()
        img = dicom_transform(
            img,
            dicom_data.wc,
            dicom_data.ww,
            dicom_data.bits
        )
        img = convert_dtype(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif utils.extract_ext(img_path) in EXTS:
        img = imread(img_path)
    else:
        img = None
    return img

def preprocessing(img, is_tensor=False):
    img = np.array(img, dtype=np.float32)
    img = img / 255
    if is_tensor:
        img = np.expand_dims(img, axis=0)
    return img

def convert_dtype(img: np.ndarray):
    """Convert image dtype between uint8 and uint16."""
    if img.dtype == np.uint8:
        pre_max = 255.0
        new_max = 65535.0
        result_dtype = np.uint16
    elif img.dtype == np.uint16:
        pre_max = 65535.0
        new_max = 255.0
        result_dtype = np.uint8
    else:
        raise TypeError(f"Unsupported image dtype: {img.dtype}")
    return (img / pre_max * new_max).astype(result_dtype)


def equalize_hist_16bit(image):
    hist = cv2.calcHist([image], [0], None, [65536], [0, 65536])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min()) * 65535.0
    cdf = np.ma.filled(cdf_m, 0).astype(np.uint16)
    return cdf[image]


def normalize_image(image, mode='minmax', dtype=np.uint8):
    if image.dtype == np.uint16:
        max_value = 65535.0
    elif image.dtype == np.uint8:
        max_value = 255.0
    else:
        raise NotImplementedError
    if mode == 'local':
        float_gray = image.astype(np.float32) / max_value
        mean = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=22.0)
        num = float_gray - mean
        variance = cv2.GaussianBlur(num * num, (0, 0), sigmaX=40.0)
        std_deviation = cv2.pow(variance, 0.5)
        result = num / std_deviation
    elif mode == 'std':
        float_gray = image.astype(np.float32) / max_value
        result = (float_gray - np.mean(float_gray)) / np.std(float_gray)
    elif mode == 'minmax':
        maxp = np.max(image)
        minp = np.min(image)
        result = (image - minp) / (maxp - minp) * max_value
    else:
        raise ValueError
    return result.astype(dtype)


def adjust_img(img_arr, c, w):
    if img_arr is None:
        return None
    img_arr = img_arr.astype(np.float32)
    std = (img_arr - (c - 0.5)) / (w - 1) + 0.5
    std = std.clip(0, 1)
    img_arr = std * 65535
    img_arr = clip_uint16(img_arr)
    return img_arr.astype(np.uint16)


def calc_window_params(image):
    hist = cv2.calcHist([image],[0], None, [65536], [0, 65536])
    cdf = hist.cumsum()
    wc = int((cdf.argmax() + cdf.argmin() + 1) / 2)
    ww = int(cdf.argmax() - cdf.argmin() + 1)
    return wc, ww


def gamma_correct(image, gamma=1.0):
    if not 0.0 < gamma < 5.0:
        return image

    if image.dtype == np.uint8:
        return (255 * ((image / 255) ** (1 / gamma))).astype(np.uint8)
    elif image.dtype == np.uint16:
        return (65535 * ((image / 65535) ** (1 / gamma))).astype(np.uint16)
    else:
        raise TypeError(f"Unsupported {image.dtype}")


def change_contrast(image, contrast=0.0):
    if not - 1.0 < contrast < 1.0:
        return image

    if image.dtype == np.uint8:
        result = np.clip(image + image * contrast, 0, 255)
        return result.astype(np.uint8)
    elif image.dtype == np.uint16:
        result = np.clip(image + image * contrast, 0, 65535)
        return result.astype(np.uint16)
    else:
        raise TypeError(f"Unsupported {image.dtype}")
    
def graylevel_transform(img, brightness, contrast):
    if img.dtype == np.uint8:
        result = np.clip(img * contrast + brightness, 0, 255)
        return result.astype(np.uint8)
    elif img.dtype == np.uint16:
        result = np.clip(img * contrast + brightness, 0, 65535)
        return result.astype(np.uint16)
    else:
        raise TypeError(f"Unsupported {img.dtype}")


def dicom_transform(img, wc, ww, bits):
    if img is None:
        return None

    low = int(wc - ww / 2)
    high = int(wc + ww / 2)
    std = (img - low) / ww
    std = std.clip(0, 1)

    if bits == 8:
        img = std * 255
        img = clip_uint8(img)
    elif bits == 16:
        img = std * 65535
        img = clip_uint16(img)
    else:
        raise NotImplementedError("Unsupported data type.")
    
    return img


def clip_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def clip_uint16(image):
    return np.clip(image, 0, 65535).astype(np.uint16)


class MFF():
    def __init__(self, image, sigma=75, filter_num=8, filter_size=9, subst_ce=1, hf_ce=5, drc_filter_size=5, drc_grad=0.0):
        self.original_image = image.astype(np.float32)
        self.sigma = sigma
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.subst_ce = subst_ce
        self.hf_ce = hf_ce
        self.drc_filter_size = drc_filter_size
        self.drc_grad = drc_grad

        self.build()
    
    def build(self):
        # Create low frequency filter and calculate substractions.
        _buf = None
        subst = None
        for _ in range(self.filter_num):
            if _buf is None:
                blur = cv2.bilateralFilter(self.original_image, self.filter_size, self.sigma, self.sigma)
                subst = (self.original_image - blur) * self.subst_ce
                _buf = blur
            else:
                blur = cv2.bilateralFilter(_buf, self.filter_size, self.sigma, self.sigma)
                subst += (_buf - blur) * self.subst_ce
                _buf = blur
        self.subst = subst

        # High frequency mask.
        self.hf_mask = self.subst * self.hf_ce

        # Dynamic range compression masks.
        _img = self.original_image - self.subst
        _img = self.blur_img = cv2.blur(_img, (self.drc_filter_size, self.drc_filter_size))
        mean = self.pixcel_mean = np.mean(_img)
        self.drc_mask = np.where(_img > mean,
                                self.drc_grad * (_img - self.pixcel_mean), 
                                -self.drc_grad * (mean - _img))


    def run(self, custom_grad=None):
        if custom_grad:
            drc_mask = np.where(self.blur_img > self.pixcel_mean,
                                custom_grad * (self.blur_img - self.pixcel_mean), 
                                -custom_grad * (self.pixcel_mean - self.blur_img))
        else:
            drc_mask = self.drc_mask

        # Apply masks.
        result_image = self.original_image + drc_mask + self.hf_mask
        result_image = clip_uint16(result_image)
        return result_image.astype(np.uint16)


def mask2polygon(masks, labels, approx_epsilon=0.003, area_limit=50):
    """ masks: 0 or 255 pix masks, shape (h, w, c)"""
    shapes = []
    for i in range(masks.shape[2] - 1):
        binary = masks[:, :, i + 1]
        binary = cv2.dilate(binary, (9, 9))
        # binary = np.array(np.where(masks[:, :, i + 1], 255, 0), dtype=np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not len(contours):
            continue
        for cnt in contours:
            # skip figures have small area.
            area = cv2.contourArea(cnt)
            if area < area_limit:
                continue
            # Detect points of a polygon.
            approx = cv2.approxPolyDP(
                curve=cnt,
                epsilon=approx_epsilon * cv2.arcLength(cnt, True),
                closed=True)
            # Skip polygons have less than 3 points.
            if len(approx) < 3:
                continue
            approx = approx.astype(int).reshape((-1, 2)).tolist()

            shape = {}
            shape["label"] = labels[i - 1]
            shape["points"] = approx
            shape["shape_type"] = "polygon"
            shapes.append(shape)
    return shapes


def mask2merge(src_img, pred, class_names, gt=None, thresh=0.5):
    """Return the RGB image merged AI prediction masks and the original image."""
    fonttype = cv2.FONT_HERSHEY_DUPLEX
    fontsize = 0.5
    fontcolor = (255, 255, 255) if np.mean(src_img[:, :100, :]) < 128 else (0, 0, 0)  # select color depending on background
    fontweight = 1

    src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)

    merge = src_img.astype(float)
    for c in range(pred.shape[2] - 1):
        color = LABEL_COLORMAP[c + 2][::-1]
        m = pred[:, :, c + 1]
        indexes = np.where(m >= thresh)
        if indexes[0].size == 0 or indexes[1].size == 0:
            continue

        mask = np.zeros_like(src_img)
        for x in range(3):
            binary = np.zeros_like(m, dtype=float)
            binary[indexes] = float(color[x])
            mask[..., x] = binary
        merge += mask.astype(float)

        # add label text
        label = class_names[c]
        y_idx, x_idx = indexes
        x1 = np.min(x_idx)
        y1 = np.min(y_idx) - 5
        cv2.putText(merge, label, (x1, y1), fonttype, fontsize, color, 1, cv2.LINE_AA)
    merge[merge > 255] = 255.0
    merge = merge.astype(np.uint8)
    cv2.putText(merge, "AI", (0, 10), fonttype, fontsize, fontcolor, fontweight, cv2.LINE_AA)

    if gt is not None:
        gt_merge = src_img.astype(float)
        for c in range(pred.shape[2] - 1):
            color = LABEL_COLORMAP[c + 2][::-1]
            gt_m = gt[:, :, c + 1]
            indexes = np.where(gt_m >= thresh)
            if indexes[0].size == 0 or indexes[1].size == 0:
                continue

            gt_mask = np.zeros_like(src_img)
            for x in range(3):
                gt_mask[..., x] = gt_m * float(color[x])
           
            # add label text
            label = class_names[c]
            y_idx, x_idx = indexes
            x1 = np.min(x_idx)
            y1 = np.min(y_idx) - 5
            cv2.putText(gt_merge, label, (x1, y1), fonttype, fontsize, color, 1, cv2.LINE_AA)
            gt_merge += gt_mask.astype(float)
        gt_merge[gt_merge > 255] = 255.0
        gt_merge = gt_merge.astype(np.uint8)
        cv2.putText(gt_merge, "human", (0, 10), fonttype, fontsize, fontcolor, fontweight, cv2.LINE_AA)

    cv2.putText(src_img, "original", (0, 10), fonttype, fontsize, fontcolor, fontweight, cv2.LINE_AA)

    if gt is not None:
        concat = np.concatenate([src_img, merge, gt_merge], axis=1)
    else:
        concat = np.concatenate([src_img, merge], axis=1)

    concat = cv2.cvtColor(concat, cv2.COLOR_BGR2RGB)
    return concat


def det2merge(src_img, pred, gt=None):
    fonttype = cv2.FONT_HERSHEY_DUPLEX
    fontsize = 1
    fontcolor = (255, 255, 255) if np.mean(src_img[:, :100, :]) < 128 else (0, 0, 0)  # select color depending on background
    fontweight = 1

    merge = np.copy(src_img)
    merge = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    for p in pred:
        class_id = p["class_id"]
        class_name = p["class_name"]
        xmin, ymin, xmax, ymax = list(map(int, p["bbox"]))
        color = LABEL_COLORMAP[class_id + 2][::-1]
        cv2.rectangle(merge, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)
 
        cv2.putText(merge, class_name, (xmin, ymin+5), fonttype, 1, color, 1, cv2.LINE_AA)

    merge = cv2.cvtColor(merge, cv2.COLOR_BGR2RGB)
    return merge


def mask2rect(mask):
    m = np.copy(mask)
    if np.max(m) == 1:
        m *= 255
    m = np.array(m, dtype=np.uint8)

    rect_list = []
    contours, hierarchy = cv2.findContours(
        m, 
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    ) 
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 3 or h < 3:  # skip tiny box
            continue
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        rect_list.append([x1, y1, x2, y2])
    return rect_list


def fig2img(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib figure to an RGB image."""
    if fig is None:
        return None
    if not isinstance(fig, plt.Figure):
        raise TypeError("Expected a matplotlib Figure object.")
    if fig.canvas is None:
        raise ValueError("Figure canvas is not initialized.")
    fig.canvas.draw()
    data = fig.canvas.tostring_argb()
    w, h = fig.canvas.get_width_height()
    c = len(data) // (w * h)
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)[..., 1:]  # remove alpha channel


def save_canvas_img(canvas_img: np.ndarray, path: str) -> bool:
    if canvas_img.dtype == np.uint16:
        canvas_img = convert_dtype(canvas_img)
    if canvas_img.ndim == 2:
        canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2RGB)
    return imwrite(canvas_img, path)