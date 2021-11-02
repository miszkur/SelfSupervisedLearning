import tensorflow as tf
import tensorflow_addons as tfa 
from tqdm import tqdm

import sys
sys.path.append('../')
from models.model import SiameseNetwork

class Experiment():
    def __init__(self, config=None) -> None:
        self.online_network = SiameseNetwork()
        self.target_network = SiameseNetwork()

    def augment(self, x):
        x_aug = tfa.image.gaussian_filter2d(x)
        return x_aug

    def grad(self, input, input_aug):
        y = self.target_network(input)
        y_aug = self.target_network(input_aug)
        with tf.GradientTape() as tape:
            x = self.online_network(input)
            y = tf.stop_gradient(y)
            x_aug = self.online_network(input_aug)
            y_aug = tf.stop_gradient(y_aug)
            loss_value = self.online_network.loss(x, x_aug, y, y_aug)
        return loss_value, tape.gradient(
            loss_value, self.online_network.model.trainable_variables)

    def train(
        self, 
        ds: tf.data.Dataset,  
        save_path: str,
        epochs=100,  
        show_history=True):

        train_loss_results = []
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            print('lets start')
            for x in tqdm(ds):

                # Optimize the model
                x_aug = self.augment(x)
                print('hello!')
                loss_value, grads = self.grad(x, x_aug)
                print('after grad:', loss_value)
                self.online_network.model.optimizer.apply_gradients(
                    zip(grads, self.online_network.model.trainable_variables))
                print('after update')

                # Track progress
                epoch_loss_avg.update_state(loss_value)  
                print("step done")

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())

            if epoch % 1 == 0:
                print("Epoch {:03d}: Loss: {:.3f}".format(
                    epoch, epoch_loss_avg.result()))

        # history = self.model.fit(ds, epochs=epochs, verbose=1)
        self.online_network.save_model(save_path)
        print('bye bye!')

