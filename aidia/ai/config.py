import os
import json
import torch

from aidia import LOCAL_DATA_DIR_NAME


class AIConfig(object):
    def __init__(self, dataset_dir=None):
        """Common config class."""
        if dataset_dir is not None and not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"{dataset_dir} is not found.")
        self.dataset_dir = dataset_dir
        self.log_dir = None
        self.gpu_num = 0
        self.image_size = (0, 0)
        self.num_classes = 0
        self.total_batchsize = 0

        # self.USE_MULTI_GPUS = False
        # self.SAVE_BEST = True
        self.NAME = 'test'
        self.TASK = "Segmentation"
        self.DATASET_NUM = 1
        self.SEED = 12345

        self.SUBMODE = False
        self.DIR_SPLIT = False
        self.EARLY_STOPPING = False

        # training setting
        self.INPUT_SIZE = 256
        self.BATCH_SIZE = 8
        self.TRAIN_STEP = None
        self.VAL_STEP = None
        self.EPOCHS = 10
        self.LEARNING_RATE = 0.001
        self.N_SPLITS = 5

        self.LABELS = []
        self.REPLACE_DICT = {}

        self.MODEL = "YOLOv4"

        # self.DEPTH = 8
        # self.CROP_MODE = 'polygon'
        # self.SQUARE = False

        # image augmentatin
        self.RANDOM_HFLIP = True
        self.RANDOM_VFLIP = True
        self.RANDOM_ROTATE = 0.1
        self.RANDOM_SCALE = 0.1
        self.RANDOM_SHIFT = 0.1
        self.RANDOM_SHEAR = 0.1
        self.RANDOM_BRIGHTNESS = 0.1
        self.RANDOM_CONTRAST = 0.1
        self.RANDOM_BLUR = 0.1  # 0 to n
        self.RANDOM_NOISE = 0.1

        # inference setting
        self.SHOW_LABELS = True
        self.SHOW_CONF = True

        self.build_params()
            
    def build_params(self):
        self.gpu_num = torch.cuda.device_count()
        # if self.USE_MULTI_GPUS and self.gpu_num > 1:
        #     self.total_batchsize = self.BATCH_SIZE * self.gpu_num
        # else:
        #     self.total_batchsize = self.BATCH_SIZE
        self.total_batchsize = self.BATCH_SIZE
        if self.dataset_dir is not None:
            self.log_dir = os.path.join(self.dataset_dir, LOCAL_DATA_DIR_NAME, self.NAME)
        self.image_size = (self.INPUT_SIZE, self.INPUT_SIZE)
        self.num_classes = len(self.LABELS)

    def load(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} is not found.")
        try:
            with open(json_path, encoding="utf-8") as f:
                dic = json.load(f)
                for key, value in dic.items():
                    if key == "dataset_dir":
                        continue
                    setattr(self, key, value)
        except Exception as e:
            try:    #  not UTF-8 json file handling
                with open(json_path) as f:
                    dic = json.load(f)
                    for key, value in dic.items():
                        if key == "dataset_dir":
                            continue
                        setattr(self, key, value)
            except Exception as e:
                raise ValueError(f"Failed to load config.json: {e}")
        
        if self.RANDOM_ROTATE < 0.0 or self.RANDOM_ROTATE > 0.5:
            self.RANDOM_ROTATE = 0.1
        if self.RANDOM_SCALE < 0.0 or self.RANDOM_SCALE > 0.5:
            self.RANDOM_SCALE = 0.1
        if self.RANDOM_SHIFT < 0.0 or self.RANDOM_SHIFT > 0.5:
            self.RANDOM_SHIFT = 0.1
        if self.RANDOM_SHEAR < 0.0 or self.RANDOM_SHEAR > 0.5:
            self.RANDOM_SHEAR = 0.1
        if self.RANDOM_BRIGHTNESS < 0.0 or self.RANDOM_BRIGHTNESS > 0.5:
            self.RANDOM_BRIGHTNESS = 0.1
        if self.RANDOM_CONTRAST < 0.0 or self.RANDOM_CONTRAST > 0.5:
            self.RANDOM_CONTRAST = 0.1
        if self.RANDOM_BLUR < 0.0 or self.RANDOM_BLUR > 0.5:
            self.RANDOM_BLUR = 0.1
        if self.RANDOM_NOISE < 0.0 or self.RANDOM_NOISE > 0.5:
            self.RANDOM_NOISE = 0.1

        self.build_params()

    def save(self, json_path):
        with open(json_path, mode='w', encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)


    @staticmethod
    def is_gpu_available():
        if torch.cuda.device_count() > 0:
            return True
        return False
    
    @staticmethod
    def get_gpu_count():
        return torch.cuda.device_count()
    
    @staticmethod
    def get_gpu_names():
        names = []
        for i in range(torch.cuda.device_count()):
            names.append(torch.cuda.get_device_name(i))
        return names
    
    def is_ultralytics(self) -> bool:
        """Check if the model is from Ultralytics."""
        if self.MODEL.startswith("Ultralytics_"):
            return True
        return False
