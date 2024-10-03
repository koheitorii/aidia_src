import tensorflow as tf
import os

from aidia.ai.config import AIConfig

class TestModel():
    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self.dataset = None
        self.model = None
    
    def set_config(self, config):
        self.config = config

    def build_dataset(self, mode=None):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        self.dataset = [(train_images, train_labels), (test_images, test_labels)]

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