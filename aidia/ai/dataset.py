import os
import json
import numpy as np
import glob
import cv2

from aidia import CLS, DET, SEG, EXTS
from aidia import aidia_logger
from aidia import dicom
from aidia import image
from aidia import utils
from aidia import errors
from aidia.ai.config import AIConfig


class Dataset(object):
    def __init__(self, config:AIConfig, load=False):

        self.config = config
        np.random.seed(self.config.SEED)

        self.dataset_num = config.DATASET_NUM

        if self.config.TASK == CLS:
            self.target_shape = [
                "polygon",
                "rectangle",
                "linestrip",
                "line",
                "point",
            ]
        elif self.config.TASK == DET:
            self.target_shape = ["polygon", "rectangle"]
        elif self.config.TASK == SEG:
            self.target_shape = ["polygon"]

        self.image_info = []
        self.class_info = []
        self.num_per_class = []
        self.train_per_class = []
        self.val_per_class = []
        self.test_per_class = []

        self.subdir_dict = {}
        self.subdir_ids = []

        self.train_ids = []
        self.test_ids = []
        self.val_ids = []

        self.num_shapes = 0
        self.num_train = 0
        self.num_test = 0
        self.num_val = 0
        self.train_steps = 0
        self.val_steps = 0

        if load:
            p = os.path.join(self.config.log_dir, "dataset.json")
            self.load(p)
        else:
            self.load_classes()
            self.load_data()
            self.prepare()

    ### General Dataset Function ###
    def add_class(self, class_id, class_name):
        self.class_info.append({
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, image_id, path, **kwargs):
        info = {
            "id": image_id,
            "path": path,
        }
        info.update(kwargs)
        self.image_info.append(info)

    def prepare(self):
        if not len(self.image_info):
            raise errors.DataLoadingError('image infomation is empty')
        self.num_images = len(self.image_info)
        self.num_classes = len(self.class_info)
        self.image_ids = np.arange(self.num_images)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]

        self.split_data()

        self.train_per_class = [0] * self.num_classes
        self.val_per_class = [0] * self.num_classes
        self.test_per_class = [0] * self.num_classes
        for i in self.train_ids:
            a = self.image_info[i]["annotations"]
            for s in a:
                label = s["label"]
                label_id = self.class_names.index(label)
                self.train_per_class[label_id] += 1
        for i in self.val_ids:
            a = self.image_info[i]["annotations"]
            for s in a:
                label = s["label"]
                label_id = self.class_names.index(label)
                self.val_per_class[label_id] += 1
        for i in self.test_ids:
            a = self.image_info[i]["annotations"]
            for s in a:
                label = s["label"]
                label_id = self.class_names.index(label)
                self.test_per_class[label_id] += 1

    def load(self, json_path):
        # p = os.path.join(self.config.log_dir, "dataset.json")
        try:
            with open(json_path, encoding="utf-8") as f:
                dic = json.load(f)
                for key, value in dic.items():
                    setattr(self, key, value)
        except Exception as e:
            try:    # not UTF-8 json file handling
                with open(json_path) as f:
                    dic = json.load(f)
                    for key, value in dic.items():
                        setattr(self, key, value)
            except:
                raise errors.DataLoadingError(f"Failed to load dataset.json: {e}")

    def save(self, json_path):
        # p = os.path.join(self.config.log_dir, "dataset.json")
        save_dict = self.__dict__.copy()
        save_dict.pop("config")
        for k, v in save_dict.items():
            if isinstance(v, np.ndarray):
                save_dict[k] = v.tolist()
        with open(json_path, mode='w', encoding="utf-8") as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        
        #TODO: add functions of saving only dataset information

    def load_classes(self):
        for idx, label in enumerate(self.config.LABELS):
            self.add_class(class_id=idx, class_name=str(label))
        self.num_per_class = np.zeros(len(self.class_info), np.uint)

    def load_data(self):
        if self.config.SUBMODE:
            _id = 0
            dir_list = glob.glob(os.path.join(self.config.dataset_dir, "**"))
            json_paths = []
            for subdir_path in dir_list:
                if utils.get_basename(subdir_path) == "data":
                    continue
                subdir_jsons = glob.glob(os.path.join(subdir_path, "*.json"))
                for p in subdir_jsons:
                    json_paths.append(p)
                    self.subdir_dict[p] = _id
                _id += 1
        else:
            json_paths = glob.glob(os.path.join(self.config.dataset_dir, "*.json"))
        json_paths = sorted(json_paths)

        image_id = 0
        for json_path in json_paths:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("shapes") is None:
                continue

            new_shapes = []
            for shape in data["shapes"]:
                if not shape["shape_type"] in self.target_shape:
                    continue
                label = shape["label"].split("_")
                label = list(set(label))
                new_label = []
                for l in label:
                    if l in self.config.LABELS:
                        new_label.append(l)

                # skip multi label data and no label data
                if len(new_label) > 1 or len(new_label) == 0:
                    continue

                new_label = new_label[0]
                label_index = self.config.LABELS.index(new_label)
                self.num_per_class[label_index] += 1
                shape["label"] = new_label
                new_shapes.append(shape)
                self.num_shapes += 1

            # skip no shapes data
            if len(new_shapes) == 0:
                continue

            name = os.path.splitext(json_path)[0]
            img_path = None
            if os.path.exists(name) and dicom.is_dicom(name):
                img_path = name
            else:
                for ext in EXTS:
                    p = name + ext
                    if os.path.exists(p):
                        img_path = p
                        break
            if not img_path:
                continue
            
            _id = 0
            if self.config.SUBMODE:
                _id = self.subdir_dict[json_path]
                if _id not in self.subdir_ids:
                    self.subdir_ids.append(_id)

            self.add_image(image_id=image_id,
                           subdir_id=_id,
                           path=img_path,
                           height=data["height"],
                           width=data["width"],
                           annotations=new_shapes)
            image_id += 1

    def split_data(self):
        """Get ids of data splited to train, validation, and test."""
        i = self.dataset_num

        if self.config.SUBMODE and self.config.DIR_SPLIT:
            subdir_ids = np.copy(self.subdir_ids)
            np.random.shuffle(subdir_ids)
            train_subdir_ids = np.array_split(subdir_ids, self.config.N_SPLITS)
            test_subdir_ids = train_subdir_ids.pop(-i)
            train_subdir_ids = np.concatenate(train_subdir_ids)
            split_pos = int(train_subdir_ids.size * 0.95)
            train_subdir_ids, val_subdir_ids = np.split(train_subdir_ids, [split_pos])

            self.num_subdir = len(self.subdir_ids)
            self.num_train_subdir = len(train_subdir_ids)
            self.num_test_subdir = len(test_subdir_ids)
            self.num_val_subdir = len(val_subdir_ids)

            for image_id in self.image_ids:
                subdir_id = self.image_info[image_id]["subdir_id"]
                if subdir_id in train_subdir_ids:
                    self.train_ids.append(image_id)
                elif subdir_id in test_subdir_ids:
                    self.test_ids.append(image_id)
                elif subdir_id in val_subdir_ids:
                    self.val_ids.append(image_id)
                else:
                    raise ValueError("Subdirectory split processing error.")
            self.train_ids = np.array(self.train_ids)
            self.test_ids = np.array(self.test_ids)
            self.val_ids = np.array(self.val_ids)
            np.random.shuffle(self.train_ids)
            np.random.shuffle(self.test_ids)
            np.random.shuffle(self.val_ids)

        else:
            ids = np.copy(self.image_ids)
            np.random.shuffle(ids)
            self.train_ids = np.array_split(ids, self.config.N_SPLITS)
            self.test_ids = self.train_ids.pop(-i)
            self.train_ids = np.concatenate(self.train_ids)
            split_pos = int(self.train_ids.size * 0.95)
            self.train_ids, self.val_ids = np.split(self.train_ids, [split_pos])
        
        if len(self.val_ids) == 0 or len(self.test_ids) == 0:
            raise errors.DataFewError

        self.num_train = len(self.train_ids)
        self.num_test = len(self.test_ids)
        self.num_val = len(self.val_ids)
        self.train_steps = self.num_train // self.config.total_batchsize
        self.val_steps = self.num_val // self.config.total_batchsize
        if self.train_steps == 0 or self.val_steps == 0:
            raise errors.DataFewError
        
    ### Data Loading Function ###
    def load_image(self, image_id, is_resize=True):
        img_path = self.image_info[image_id]["path"]

        # if img_path was not found, get relative path
        if not os.path.exists(img_path):
            if self.config.SUBMODE:
                filename = os.path.basename(img_path)
                dirname = os.path.basename(os.path.dirname(img_path))
                img_path = os.path.join(self.config.dataset_dir, dirname, filename)
            else:
                filename = os.path.basename(img_path)
                img_path = os.path.join(self.config.dataset_dir, filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Failed to load {img_path}")
        
        img = image.read_image(img_path)

        if is_resize:
            img = cv2.resize(img, self.config.image_size)
        return img

    def load_masks(self, image_id):
        image_info = self.image_info[image_id]
        annotations = self.image_info[image_id]["annotations"]

        masks = []
        if self.num_classes > 1:
            h = image_info["height"]
            w = image_info["width"]
            foreground = np.zeros(shape=self.config.image_size, dtype=np.uint8)
            mask_per_class = []
            for i in range(self.num_classes + 1):
                mask_per_class.append(np.zeros(shape=self.config.image_size, dtype=np.uint8))
            for a in annotations:
                _m = np.zeros(shape=(h, w, 3), dtype=np.uint8)
                points = a["points"]
                points = [[int(p[0]), int(p[1])] for p in points] # points to int
                points = np.array(points, np.int32)
                _m = cv2.fillPoly(_m, [points], color=(1, 1, 1))
                _m = cv2.resize(_m, self.config.image_size,
                                interpolation=cv2.INTER_NEAREST)
                _m = _m[:, :, 0]
                foreground += _m  # to create background mask

                label = a["label"]
                if isinstance(label, list):  # TODO: for old system
                    label = label[0]
                class_id = self.config.LABELS.index(label) + 1
                if class_id not in list(range(self.num_classes + 1)):
                    raise IndexError(f"{class_id} is out of range. {a['label'][0]}")
                mask_per_class[class_id] += _m
            
            # create background mask
            _foreground = np.where(foreground >= 1, 1, 0).astype(np.uint8)
            background = np.where(_foreground == 1, 0, 1).astype(np.uint8)

            # marge masks, (background, class 1, class 2, ...)
            mask_per_class[0] += background
            for i in range(self.num_classes):
                mask_per_class[i + 1] = np.where(mask_per_class[i + 1] >= 1, 1, 0).astype(np.uint8)
            masks = np.stack(mask_per_class, axis=2)
            return masks
        else: # 1 class (2 class including background)
            h = image_info["height"]
            w = image_info["width"]
            foreground = np.zeros(shape=self.config.image_size, dtype=np.uint8)
            for a in annotations:
                _m = np.zeros(shape=(h, w, 3), dtype=np.uint8)
                points = a["points"]
                points = [[int(p[0]), int(p[1])] for p in points] # points to int
                points = np.array(points, np.int32)
                _m = cv2.fillPoly(_m, [points], color=(1, 1, 1))
                _m = cv2.resize(_m, self.config.image_size,
                                interpolation=cv2.INTER_NEAREST)
                _m = _m[:, :, 0]
                foreground += _m
            foreground = np.where(foreground >= 1, 1, 0).astype(np.uint8)
            background = np.where(foreground == 1, 0, 1).astype(np.uint8)
            masks.append(background)
            masks.append(foreground)
            masks = np.stack(masks, axis=2)
            return masks

    def get_yolo_bboxes(self, image_id):
        """Get YOLO bounding boxes.

        YOLO format example:
        image_dir/001.jpg x_min, y_min, x_max, y_max, class_id x_min2, y_min2, x_max2, y_max2, class_id2

        Returns
        -------
        np.array([x_min, y_min, x_max, y_max, class_id], [...], ...)
        """
        annotations = self.image_info[image_id]["annotations"]
        bboxes = []
        for a in annotations:
            shape_type = a["shape_type"]
            if shape_type in ['polygon', 'rectangle']:
                points = a["points"]
                label = a["label"]
                if isinstance(label, list):  # TODO: for old system
                    label = label[0]
                class_id = self.class_names.index(label)
                if shape_type == 'polygon':
                    xmin = np.min(points)
                    ymin = np.min(points)
                    xmax = np.max(points)
                    ymax = np.max(points)
                else:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    xmin = min(x1, x2)
                    ymin = min(y1, y2)
                    xmax = max(x1, x2)
                    ymax = max(y1, y2)
                bboxes.append([
                    float(xmin),
                    float(ymin),
                    float(xmax),
                    float(ymax),
                    float(class_id)
                ])
            else:
                continue
        bboxes = np.array(bboxes)
        # h = self.image_info[image_id]["height"]
        # w = self.image_info[image_id]["width"]
        # bboxes = bboxes * np.array([w, h, w, h, 1])
        bboxes = bboxes.astype(np.int64)
        return bboxes
