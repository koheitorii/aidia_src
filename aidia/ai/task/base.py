import os
import random
import cv2
import time
import albumentations as A
from aidia import APP_DIR
from aidia.ai.config import AIConfig
from aidia.ai.dataset import Dataset

class Model(object):
    """Base class for AI models."""
    def __init__(self, config: AIConfig) -> None:
        """Initialize the model with the given configuration."""
        self.config = config
        self.dataset = None
        self.model = None
    
    def set_config(self, config: AIConfig):
        """Set the configuration for the model."""
        self.config = config

    def build_dataset(self):
        """Override this method to build the dataset."""
        self.dataset = Dataset(self.config)

    def load_dataset(self):
        """Override this method to load the dataset."""
        self.dataset = Dataset(self.config, load=True)
    
    def build_model(self, mode, weights_path=None):
        """Override this method to build the model."""
        assert mode in ["train", "test"]
        if mode == "train":
            self.model = None
        elif mode == "test":
            self.model = None
        else:
            raise ValueError("Mode must be 'train' or 'test'.")
        pass

    def train(self, custom_callbacks=None):
        pass

    def predict(self, images):
        pass
    
    def convert(self):
        pass


class CustomAugmentation(object):
    """Custom augmentation class using Albumentations."""
    def __init__(self, config: AIConfig):
        self.config = config
        self.augmentation = self.build_augmentation()

    def build_augmentation(self):
        """Build the augmentation pipeline based on the configuration."""
        augmentations = []
        if self.config.RANDOM_HFLIP:
            augmentations.append(A.HorizontalFlip(p=0.5))
        if self.config.RANDOM_VFLIP:
            augmentations.append(A.VerticalFlip(p=0.5))
        
        # Combine geometric transformations to preserve image size
        if self.config.RANDOM_ROTATE > 0.0 or self.config.RANDOM_SHIFT > 0.0 or self.config.RANDOM_SCALE > 0.0:
            augmentations.append(A.ShiftScaleRotate(
                shift_limit=self.config.RANDOM_SHIFT if self.config.RANDOM_SHIFT > 0.0 else 0.0,
                scale_limit=self.config.RANDOM_SCALE if self.config.RANDOM_SCALE > 0.0 else 0.0,
                rotate_limit=self.config.RANDOM_ROTATE * 90.0 if self.config.RANDOM_ROTATE > 0.0 else 0.0,
                border_mode=0,  # cv2.BORDER_CONSTANT
                fill=128,
                fill_mask=0,
                p=0.5
            ))
        
        if self.config.RANDOM_SHEAR > 0.0:
            augmentations.append(A.Affine(
                shear=(-self.config.RANDOM_SHEAR * 45, self.config.RANDOM_SHEAR * 45),
                border_mode=0,  # cv2.BORDER_CONSTANT
                fill=128,
                fill_mask=0,
                p=0.5
            ))
        if self.config.RANDOM_BRIGHTNESS > 0.0:
            augmentations.append(A.RandomBrightnessContrast(
                brightness_limit=self.config.RANDOM_BRIGHTNESS,
                contrast_limit=0,
                p=0.5
            ))
        if self.config.RANDOM_CONTRAST > 0.0:
            augmentations.append(A.RandomBrightnessContrast(
                brightness_limit=0,
                contrast_limit=self.config.RANDOM_CONTRAST,
                p=0.5
            ))
        if self.config.RANDOM_BLUR > 0.0:
            augmentations.append(A.Blur(blur_limit=(3, int(self.config.RANDOM_BLUR * 10 + 2)), p=self.config.RANDOM_BLUR))
        if self.config.RANDOM_NOISE > 0.0:
            augmentations.append(A.GaussNoise(std_range=(0, self.config.RANDOM_NOISE * 0.5), p=0.5))

        return A.Compose(augmentations, seed=self.config.SEED)
