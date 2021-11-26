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
            loss_value = self.loss(labels, x)
            
        grads = tape.gradient(
            loss_value, 
            self.network.classification_layer.trainable_variables
            )
        del tape
        return loss_value, grads, x


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
            acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
            for batch, labels in tqdm(ds):
                loss_value, grads, x = self.get_gradients(batch, labels)
                self.network.optimizer.apply_gradients(
                    zip(
                        grads, 
                        self.network.classification_layer.trainable_variables
                        )
                    )
                epoch_loss_avg.update_state(loss_value)
                acc_metric.update_state(labels, x)

            train_acc = acc = acc_metric.result().numpy()
            eval_acc = self.evaluate(ds_val)
            if verbose:
                print("Epoch {:03d}: Loss: {:.3f}, Train acc: {:.3f}, Val acc: {:.3f}".format(
                    epoch, epoch_loss_avg.result(), train_acc, eval_acc))

    def evaluate(self, ds):
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        for img, lbl, in ds:
            results = self.network.predict(img)
            acc_metric.update_state(lbl, results)
        
        acc = acc_metric.result().numpy()
        return acc