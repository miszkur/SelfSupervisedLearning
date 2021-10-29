import tensorflow as tf
from models.resnet18 import ResNet18
import matplotlib.pyplot as plt
tfk = tf.keras


class DirectPred(tf.keras.Model):
    def __init__(self, num_classes, image_size=224):
        super(DirectPred, self).__init__()
        self.resnet18 = ResNet18()
        self.fc = tfk.layers.Dense(
            units=num_classes, 
            activation=tf.keras.activations.softmax)
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        x = tf.keras.layers.Input(shape=(self.image_size, self.image_size, 3))
        y = self.resnet18(x)

        if self.num_classes:
            y = self.fc(y)

        return tfk.models.Model(inputs=x, outputs=y)

    def compile(self):
        optimizer = tfk.optimizers.SGD()
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')


    def train(self, X, y, training_samples=100, epochs=100, batch_size=24, show_history=True):
        steps_per_epoch = training_samples // batch_size
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1,
                                 steps_per_epoch=steps_per_epoch)

        if show_history:
            plt.figure()
            plt.plot(history.history["loss"], label="training loss")
            plt.title(f"Loss for {self.name} model")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
