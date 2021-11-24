import tensorflow as tf
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_batch_ops import batch
import tensorflow_addons as tfa 
from tqdm import tqdm
import numpy as np
import pickle
import os

import sys
sys.path.append('../')
from models.model import SiameseNetwork
from data_processing.augmentations import DataAug, DataAugSmall

class Experiment():
    def __init__(self, config) -> None:
        self.online_network = SiameseNetwork(
            image_size=config.image_size, 
            predictor_hidden_size=config.predictor_hidden_size)
        self.online_network.compile(config.optimizer_params)

        self.target_network = SiameseNetwork(target=True, image_size=config.image_size)
        self.target_network.compile(config.optimizer_params)

        self.name = config.name
        self.tau = config.tau
        self.lambda_ = config.lambda_

        self.eigenspace_experiment = config.eigenspace_experiment
        self.symmetry_regularisation = config.symmetry_reg
        self.eps = config.eps
        self.F = None
        self.F_eigenval = []
        self.wp_eigenval = []
        self.allignment = []
        self.symmetry = []
        self.should_compute_F = self.eigenspace_experiment or self.name in ['DirectPred', 'DirectCopy']
        
        if config.image_size == (32, 32):
            self.data_aug = DataAugSmall(batch_size=config.batch_size)
        else:
            self.data_aug = DataAug(batch_size=config.batch_size)

        self.cosine_sim = tf.keras.losses.CosineSimilarity(
            axis=0,
            reduction=tf.keras.losses.Reduction.NONE
            )

    def update_target_network(self, tau):

        # update encoder
        for x, y in zip(self.target_network.encoder.variables, self.online_network.encoder.variables):
            x.assign(x + (1 - tau) * (y - x))

        # update projector
        target = self.target_network.projector.get_weights()
        online = self.online_network.projector.get_weights()
        weights = [x + (1 - tau) * (y - x) for x, y in zip(target, online)]
        self.target_network.projector.set_weights(weights)

    def update_f(self, corr):
        if self.F is None:
            self.F = corr
        else:
            self.F = self.lambda_ * self.F + (1 - self.lambda_) * corr


    def grad(self, input_aug1, input_aug2):
        y = self.target_network(input_aug1, training=True)
        y_aug = self.target_network(input_aug2, training=True)
        with tf.GradientTape() as tape:
            x, projector_output = self.online_network(
                input_aug1, 
                training=True
                )
            y = tf.stop_gradient(y)
            x_aug, projector_output_aug = self.online_network(
                input_aug2, 
                training=True
                )
            y_aug = tf.stop_gradient(y_aug)
            loss_value = self.online_network.loss(x, x_aug, y, y_aug)
        grads = tape.gradient(
            loss_value, self.online_network.model.trainable_variables)
        self.online_network.model.optimizer.apply_gradients(
                    zip(grads, self.online_network.model.trainable_variables))
        del tape 

        return (
            loss_value,
            projector_output, 
            projector_output_aug
        )


    def cosine_similarity(self, x, y):
        x = tf.math.l2_normalize(x, axis=0)
        y = tf.math.l2_normalize(y, axis=0)
        xy = tf.multiply(x, y)
        return tf.math.reduce_sum(xy, axis=0)

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
                x_aug1 = self.data_aug(x)
                x_aug2 = self.data_aug(x)
                loss_value, h1, h2 = self.grad(x_aug1, x_aug2)

                # Update target network
                self.update_target_network(self.tau)

                if self.should_compute_F:
                    corr_1 = tf.matmul(tf.expand_dims(h1, 2), tf.expand_dims(h1, 1))
                    corr_2 = tf.matmul(tf.expand_dims(h2, 2), tf.expand_dims(h2, 1))
                    corr = tf.concat([corr_1, corr_2], axis=0)
                    corr = tf.reduce_mean(corr, axis=0)
                    self.update_f(corr)

                    if self.name in ['DirectPred', 'DirectCopy']:
                        self.online_network.predictor.update_predictor(
                            F_=self.F, 
                            eps=self.eps,
                            method=self.name
                            )

                # Track progress
                epoch_loss_avg.update_state(loss_value)  

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())

            if epoch % 1 == 0:
                print("Epoch {:03d}: Loss: {:.3f}".format(
                    epoch, epoch_loss_avg.result()))

            # Eignespace alignment experiment
            if self.eigenspace_experiment and epoch % 5 == 0:
                # get F eignvalues
                eigval, eigvec = tf.linalg.eigh(self.F)
                self.F_eigenval.append(eigval)

                # get predictor head
                self.online_network.predictor.update_wp()
                wp = self.online_network.predictor.get_wp()
                self.symmetry.append(
                    self.online_network.predictor.symmetry())
                wp_eigval = tf.linalg.eigvals(wp)
                wp_eigval = tf.math.real(wp_eigval)
                self.wp_eigenval.append(wp_eigval)
                
                if self.symmetry_regularisation:
                    self.online_network.predictor.symmetry_reg()

                wp_v = tf.matmul(wp, eigvec)
                cosine = self.cosine_sim(eigvec, wp_v)
                self.allignment.append(cosine)

        self.online_network.save_model(save_path)

        if self.eigenspace_experiment:
            self.save_eigenspace_experiment_results('results_eigenspace')

    def save_eigenspace_experiment_results(self, path):
        with open(os.path.join(path, 'F_eigenval.pkl'), 'wb') as f:
            pickle.dump(self.F_eigenval, f)
        with open(os.path.join(path, 'wp_eigenval.pkl'), 'wb') as f:
            pickle.dump(self.wp_eigenval, f)
        with open(os.path.join(path, 'allignment.pkl'), 'wb') as f:
            pickle.dump(self.allignment, f)
        with open(os.path.join(path, 'symmetry.pkl'), 'wb') as f:
            pickle.dump(self.symmetry, f)

