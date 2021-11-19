import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tfk.layers 

class MLPHead(tfkl.Layer):

    def __init__(self, hidden_size=512, output_size=128) -> None:
        super(MLPHead, self).__init__()

        self.hidden_size = hidden_size
        self.wp = None

        if self.hidden_size is not None:
            self.hidden_layer = tfkl.Dense(
                hidden_size, 
                use_bias=True
                )
            # center and scale are set differently in Byol and DiretPred paper
            self.batch_norm = tfkl.BatchNormalization(
                momentum=0.9, 
                epsilon=1e-5, 
                center=False # diables beta, in DirectPred affine=False for projector
                )
            self.relu = tf.nn.relu

        self.output_layer = tfkl.Dense(
            output_size, 
            use_bias=False
            )

    def get_wp(self):
        return self.wp

    def update_wp(self):
        w1 = self.hidden_layer.get_weights()[0]
        w2 = self.output_layer.get_weights()[0]
        self.wp = tf.matmul(w1,w2)
        self.wp = tf.transpose(self.wp)

    def update_predictor(self, F, eps, method):
        # TODO runs, but check if correct
        assert self.hidden_size is None, \
            'Predictor in DirectPred should have 1 layer!'

        if method == 'DirectPred':
            eigval, eigvec = tf.linalg.eigh(F)
            eigval = tf.math.real(eigval)
            max_eigval = tf.math.reduce_max(eigval)
            eigval = tf.divide(eigval, max_eigval)
            eigval = tf.clip_by_value(
                eigval, 
                clip_value_min=0, 
                clip_value_max=max_eigval
                )
            
            p = tf.math.pow(eigval, 0.5) + eps
            p = tf.clip_by_value(
                p, 
                clip_value_min=1e-4, 
                clip_value_max=tf.math.reduce_max(p)
                )
            p_diag = tf.linalg.diag(p)
            w = tf.matmul(eigvec, p_diag)
            w = tf.matmul(w, eigvec, transpose_b=True)
        elif method == 'DirectCopy':
            pass

        self.output_layer.set_weights([w])
        
        
    def symmetry(self):
        return tf.math.reduce_sum(
            tf.math.abs(
                self.wp-tf.transpose(self.wp)
            )) / tf.math.reduce_sum(tf.math.abs(self.wp))

    def call(self, x, training=None):
        if self.hidden_size is not None:
            x = self.hidden_layer(x)
            x = self.batch_norm(x, training=training)
            x = self.relu(x)
        
        y = self.output_layer(x)
        return y
