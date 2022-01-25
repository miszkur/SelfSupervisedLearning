import tensorflow as tf
import tensorflow_addons as tfa 
from tqdm import tqdm
import os

import sys
sys.path.append('../')
from models.model import ClassificationNetwork

class Evaluation():
    def __init__(
        self, saved_encoder_path, config, saved_projection_path=None) -> None:
        self.network = ClassificationNetwork(
            config,
            saved_encoder_path,
            saved_projection_path=saved_projection_path
            )
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
            )

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
        verbose=True,
        saved_model_path='saved_model/classifier'):

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

            train_acc = acc_metric.result().numpy()
            eval_acc = self.evaluate(ds_val)
            if verbose:
                print("Epoch {:03d}: Loss: {:.3f}, Train acc: {:.3f}, Val acc: {:.3f}".format(
                    epoch, epoch_loss_avg.result(), train_acc, eval_acc))
        self.network.save_model(saved_model_path)

    def evaluate(self, ds):
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        for img, lbl, in ds:
            results = self.network.predict(img)
            acc_metric.update_state(lbl, results)
        
        acc = acc_metric.result().numpy()
        return acc