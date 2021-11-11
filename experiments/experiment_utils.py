import tensorflow as tf
import tensorflow_addons as tfa 
from tqdm import tqdm
import numpy as np

import sys
sys.path.append('../')
from models.model import SiameseNetwork

class Experiment():
    def __init__(self, config) -> None:
        self.online_network = SiameseNetwork()
        self.online_network.compile(config.optimizer_params)
        self.target_network = SiameseNetwork(target=True)
        self.target_network.compile(config.optimizer_params)
        self.tau = config.tau
        self.eigenspace_experiment = config.eigenspace_experiment

    def augment(self, x):
        x_aug = tfa.image.gaussian_filter2d(x)
        return x_aug

    def update_target_network(self, tau):

        # update encoder
        for x, y in zip(self.target_network.encoder.variables, self.online_network.encoder.variables):
            x.assign(x + (1 - tau) * (y - x))

        # update projector
        target = self.target_network.projector.get_weights()
        online = self.online_network.projector.get_weights()
        weights = [x + (1 - tau) * (y - x) for x, y in zip(target, online)]
        self.online_network.projector.set_weights(weights)



    def grad(self, input, input_aug):
        y = self.target_network(input)
        y_aug = self.target_network(input_aug)
        with tf.GradientTape() as tape:
            x, projector_output = self.online_network(input)
            y = tf.stop_gradient(y)
            x_aug, projector_output_aug = self.online_network(input_aug)
            y_aug = tf.stop_gradient(y_aug)
            loss_value = self.online_network.loss(x, x_aug, y, y_aug)
        return loss_value, tape.gradient(
            loss_value, self.online_network.model.trainable_variables), x

    def train(
        self, 
        ds: tf.data.Dataset,  
        save_path: str,
        epochs=100,  
        show_history=True):

        # Create target network.
        for x in ds.take(1):
            self.online_network(x)
            self.target_network(x)
        self.update_target_network(tau=0)

        train_loss_results = []
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            for x in tqdm(ds):

                # Optimize the model
                x_aug = self.augment(x)
                loss_value, grads, x = self.grad(x, x_aug)
                self.online_network.model.optimizer.apply_gradients(
                    zip(grads, self.online_network.model.trainable_variables))

                # Eignespace alignment experiment
                if self.eigenspace_experiment:
                    F = self.online_network.model.layers[3].output
                    # print(F)
                    corr = tf.matmul(tf.expand_dims(F, 2), tf.expand_dims(F, 1))
                    sth = tf.linalg.eigvals(corr[0,:,:])
                    # print(sth)

                # Update target network
                self.update_target_network(self.tau)

                # Track progress
                epoch_loss_avg.update_state(loss_value)  

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())

            if epoch % 1 == 0:
                print("Epoch {:03d}: Loss: {:.3f}".format(
                    epoch, epoch_loss_avg.result()))

        self.online_network.save_model(save_path)

