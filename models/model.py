import tensorflow as tf
from tensorflow.python.eager.def_function import run_functions_eagerly
from tensorflow.python.ops.nn_impl import moments
import tensorflow_addons as tfa
from models.resnet18 import ResNet18
from models.mlp_head import MLPHead
import matplotlib.pyplot as plt
tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math

class SiameseNetwork(tf.keras.Model):
    def __init__(self, target=False, image_size=224):
        super(SiameseNetwork, self).__init__()
        self.encoder = ResNet18()
        self.projector = MLPHead()
        self.target = target
        if not self.target:
            self.predictor = MLPHead()
        self.flatten = tfkl.Flatten()
        self.image_size = image_size
        self.model = self.build_model()
        # TODO: check axis
        self.cosine_sim = tf.keras.losses.CosineSimilarity(
            axis=1
        )
    
    def build_model(self):
        input = tf.keras.layers.Input(shape=(self.image_size, self.image_size, 3))
        x = self.encoder(input)
        x = self.flatten(x)
        projection = self.projector(x)
        if self.target:
            return tfk.models.Model(inputs=input, outputs=projection)

        y = self.predictor(projection)
        return tfk.models.Model(inputs=input, outputs=(y, projection))

    def call(self, x, training=False):
        y = self.model.call(x, training=False)
        return y

    def compile(self, config):
        optimizer = tfa.optimizers.SGDW(
            learning_rate = config.lr, 
            momentum = config.momentum,
            weight_decay = config.weight_decay)
        self.optimizer = optimizer
        self.model.compile(optimizer=optimizer, loss=[self.loss])

    @tf.function
    def l2_loss(self, x, y):
        # TODO: check axis for debugging
        x_norm = tfm.l2_normalize(x, axis=-1)
        y_norm = tfm.l2_normalize(y, axis=-1)
        output = (x_norm - y_norm) ** 2
        return tfm.reduce_sum(output)

    @tf.function
    def loss(self, x, x_aug, y, y_aug):
        loss = self.cosine_sim(x, y_aug)
        loss += self.cosine_sim(x_aug, y)
        return loss

    def save_model(self, saved_model_path):
        self.model.save(saved_model_path)

    def load_model(self, saved_model_path):
        model = tf.keras.models.load_model(
            saved_model_path, compile=False)
        self.model = model
        return model
