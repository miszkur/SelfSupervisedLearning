import tensorflow as tf
from tensorflow.python.eager.def_function import run_functions_eagerly
from tensorflow.python.ops.nn_impl import moments
import tensorflow_addons as tfa
from models.resnet18 import ResNet18
from models.mlp_head import MLPHead
from typing import Tuple
import matplotlib.pyplot as plt

tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math

class SiameseNetwork(tf.keras.Model):
    def __init__(self, image_size: Tuple[int], target=False, predictor_hidden_size=None, deeper_proj=False):
        super(SiameseNetwork, self).__init__()
        self.encoder = ResNet18(image_size)
        self.projector = [MLPHead()]
        if deeper_proj:
            self.projector.insert(0, MLPHead(output_size=None))
        self.target = target
        if not self.target:
            self.predictor = MLPHead(hidden_size=predictor_hidden_size)
        self.image_size = image_size
        self.model = self.build_model()
        self.cosine_sim = tf.keras.losses.CosineSimilarity(
            axis=1
        )
    
    def build_model(self):
        input = tf.keras.layers.Input(shape=(self.image_size[0], self.image_size[1], 3))
        x = self.encoder(input)
        for layer in self.projector:
            x = layer(x)
        if self.target:
            return tfk.models.Model(inputs=input, outputs=x)

        y = self.predictor(x)
        return tfk.models.Model(inputs=input, outputs=(y, x))

    def call(self, x, training=False):
        y = self.model.call(x, training=training)
        return y

    def compile(self, config):
        if config.use_SGDW:
            optimizer = tfa.optimizers.SGDW(
                learning_rate = config.lr, 
                momentum = config.momentum,
                weight_decay = config.weight_decay)
        else:
            optimizer = tfk.optimizers.SGD(
                learning_rate = config.lr, 
                momentum = config.momentum)
        self.optimizer = optimizer
        self.use_L2_weight_decay = config.use_L2_weight_decay
        self.only_predictor = config.only_predictor
        self.model.compile(optimizer=optimizer, loss=[self.loss])

    @tf.function
    def loss(self, x, x_aug, y, y_aug):
        loss = self.cosine_sim(x, y_aug)
        loss += self.cosine_sim(x_aug, y)
        if self.use_L2_weight_decay:
            if self.only_predictor:
                loss += tf.add_n(
                    [tf.nn.l2_loss(v) for v in self.predictor.trainable_variables
                    if 'bias' not in v.name]) * 0.0004 
            else:
                loss += tf.add_n(
                    [tf.nn.l2_loss(v) for v in self.model.trainable_variables
                    if 'bias' not in v.name]) * 0.0004 
        return loss

    def save_encoder(self, path):
        assert path[-3:] == '.h5', \
            'Path for saving weights should have .h5 extension'
        self.encoder.save_weights(path)

    def save_projector(self, path):
        if path is None: 
            return

        assert path[-3:] == '.h5', \
            'Path for saving weights should have .h5 extension'
        self.projector[0].save_weights(path)


class ClassificationNetwork(tf.keras.Model):
    def __init__(
        self, 
        config, 
        saved_encoder_path, 
        saved_projection_path=None, 
        num_classes=10):
        super(ClassificationNetwork, self).__init__()

        self.image_size = config.image_size
        self.encoder = ResNet18(self.image_size)
        self.projection_head = None
        if saved_projection_path is not None:
            self.projection_head = MLPHead(output_size=None)

        self.classification_layer = tfkl.Dense(
            num_classes, 
            use_bias=True
            )
        self.model = self.build_model()
        self.encoder.load_weights(saved_encoder_path)
        if saved_projection_path is not None:
            self.projection_head.load_weights(saved_projection_path)

    def build_model(self):
        input = tf.keras.layers.Input(
            shape=(self.image_size[0], 
                self.image_size[1], 
                3)
            )
        x = self.encoder(input, training=False)
        if self.projection_head is not None:
            x = self.projection_head(x, training=False)
        x = self.classification_layer(x)
        return tfk.models.Model(inputs=input, outputs=x)

    def call(self, x):
        y = self.model.call(x)
        return y

    def compile(self, num_examples, batch_size, epochs):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
            )
        # Define optimizer
        batches_per_epoch = num_examples // batch_size
        update_steps = epochs * batches_per_epoch
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            5e-2, 
            update_steps, 
            5e-4, 
            power=2
            )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(
            optimizer=self.optimizer, 
            loss=loss 
            )

    def save_model(self, saved_model_path):
        self.model.save(saved_model_path)

    def load_model(self, saved_model_path):
        model = tf.keras.models.load_model(
            saved_model_path, compile=False)
        self.model = model
        return model

    def evaluate(self,ds):
        return self.model.evaluate(ds,batch_size=128)
