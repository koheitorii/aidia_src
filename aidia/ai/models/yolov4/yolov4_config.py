import numpy as np

from aidia.ai.models.yolov4.yolov4_utils import get_anchors


class YOLO_Config(object):
    def __init__(self):
        # YOLO options
        self.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
        # self.ANCHORS              = [2,2, 4,4, 6,6, 8,8, 10,10, 12,12, 14,14, 16,16, 18,18]
        self.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
        self.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
        self.STRIDES              = [8, 16, 32]
        self.STRIDES_TINY         = [16, 32]
        self.XYSCALE              = [1.2, 1.1, 1.05]
        self.XYSCALE_TINY         = [1.05, 1.05]
        self.ANCHOR_PER_SCALE     = 3
        self.IOU_LOSS_THRESH      = 0.5
        self.MAX_BBOX_PER_SCALE   = 150

        # TEST options
        self.SCORE_THRESHOLD      = 0.25
        self.IOU_THRESHOLD        = 0.5

    def get_yolo_params(self, is_tiny=False, version=4):
        assert version in [3, 4]
        if is_tiny:
            STRIDES = np.array(self.STRIDES_TINY)
            ANCHORS = get_anchors(self.ANCHORS_TINY, is_tiny)
            XYSCALE = self.XYSCALE_TINY if version == 4 else [1, 1]
        else:
            STRIDES = np.array(self.STRIDES)
            if version == 4:
                ANCHORS = get_anchors(self.ANCHORS, is_tiny)
            elif version == 3:
                ANCHORS = get_anchors(self.ANCHORS_V3, is_tiny)
            XYSCALE = self.XYSCALE if version == 4 else [1, 1, 1]
        return STRIDES, ANCHORS, XYSCALE
