import tensorflow as tf
import tensorflow_addons as tfa 
from tqdm import tqdm
import os

import sys
sys.path.append('../')
from models.model import ClassificationNetwork
from data_processing.augmentations import DataAug, DataAugSmall

class Evaluation():
    def __init__(self, saved_model_path, config) -> None:
        self.network = ClassificationNetwork(
            saved_model_path, 
            config
            )
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
            )
        # self.data_aug = DataAugSmall(batch_size=config.batch_size)

    def get_gradients(self, batch, labels):
        with tf.GradientTape() as tape:
            x = self.network(batch)
            loss_value = self.loss(x, labels)
            
        grads = tape.gradient(
            loss_value, 
            self.network.classification_layer
            )
        del tape
        return loss_value, grads


    def train(self, 
        ds: tf.data.Dataset,
        ds_val: tf.data.Dataset,
        num_examples,
        batch_size,
        epochs=100,
        verbose=True):

        self.network.compile(num_examples, batch_size, epochs)
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            for batch, labels in ds:
                loss_value, grads = self.get_gradients(xbatch, labels)
                self.network.optimizer.apply_gradients(
                    zip(
                        grads, 
                        self.network.classification_layer.trainable_variables
                        )
                    )
                epoch_loss_avg.update_state(loss_value)

            eval_acc = evaluate(ds_val)
            if verbose:
                print("Epoch {:03d}: Loss: {:.3f}, Acc: {:.3f}".format(
                    epoch, epoch_loss_avg.result(), eval_acc))

    def evaluate(self, ds):
        results = self.network.evaluate(ds)
        return results