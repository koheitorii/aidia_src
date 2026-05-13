import os
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
