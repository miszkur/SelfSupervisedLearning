import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tfk.layers 

class MLPHead(tfkl.Layer):

    def __init__(self, hidden_size=512, output_size=128) -> None:
        super(MLPHead, self).__init__()

        self.hidden_size = hidden_size

        if self.hidden_size is not None:
            self.hidden_layer = tfkl.Dense(
                hidden_size, 
                use_bias=True
                )
            # center and scale are set differently in Byol and DiretPred paper
            self.batch_norm = tfkl.BatchNormalization(
                momentum=0.9, 
                epsilon=1e-5, 
                center=False, 
                scale=False
                )
            self.relu = tf.nn.relu

        self.output_layer = tfkl.Dense(
            output_size, 
            use_bias=False
            )

    def symmetry(self):
        w = self.output_layer.get_weights()[0]
        return np.sum(np.abs(w-w.T))/np.sum(np.abs(w))

    def call(self, x):
        if self.hidden_size is not None:
            x = self.hidden_layer(x)
            x = self.batch_norm(x)
            x = self.relu(x)
        
        y = self.output_layer(x)
        return y
