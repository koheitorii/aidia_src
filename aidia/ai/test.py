import os
import numpy as np
import tensorflow as tf

from aidia import APP_DIR
from aidia.ai.config import AIConfig

class TestModel():
    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self.dataset = None
        self.model = None
    
    def set_config(self, config):
        self.config = config

    def build_dataset(self, mode=None):
        # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path=os.path.join(APP_DIR, 'ai', 'data'))
        with np.load(os.path.join(APP_DIR, 'ai', 'data', 'mnist.npz'), allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]
        self.dataset = [(x_train, y_train), (x_test, y_test)]

    def build_model(self, mode=None):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        optim = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        self.model.compile(
            optimizer=optim,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    def train(self, custom_callbacks=None):
        callbacks = []
        if custom_callbacks:
            for c in custom_callbacks:
                callbacks.append(c)

        self.model.fit(
            self.dataset[0][0],
            self.dataset[0][1],
            batch_size=self.config.total_batchsize,
            epochs=self.config.EPOCHS,
            verbose=0,
            callbacks=callbacks,
            validation_split=0.2)

    def save(self):
        pass

    def stop_training(self):
        self.model.stop_training = True