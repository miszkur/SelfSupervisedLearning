"""
Data augmentations used during training BYOL.
For details, see Appendix B in https://arxiv.org/pdf/2006.07733v3.pdf
"""

import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers

class RandomCropByol(tfkl.Layer):
    """Crop the image according to BYOL paper.

    A random patch of the image is selected, with an area uniformly sampled 
    between 8% and 100% of that of the original image, and an aspect ratio 
    logarithmically sampled between 3/4 and 4/3. This patch is then resized to 
    the target size of 224 × 224 using bicubic interpolation.
    """

    def __init__(self, width, height) -> None:
        super(RandomCropByol, self).__init__()
        self.width = width
        self.height = height
        self.current_area = self.width * self.height

    def call(self, x):
        new_area = tf.random.uniform(
            [], 0.08, 1.0, dtype=tf.float32) * self.current_area
        min_ratio = tf.math.log(3 / 4)
        max_ratio = tf.math.log(4 / 3)
        aspect_ratio = tf.math.exp(tf.random.uniform(
            [], min_ratio, max_ratio, dtype=tf.float32)) 
        
        w = tf.cast(tf.round(tf.sqrt(new_area * aspect_ratio)), tf.int32)
        h = tf.cast(tf.round(tf.sqrt(new_area / aspect_ratio)), tf.int32)

        w = tf.minimum(w, self.width)
        h = tf.minimum(h, self.height)

        # TODO: check batch size
        x = tf.image.random_crop(x, size=[h, w, 3]) 
        x = tf.image.resize(x, [224, 224], method='bicubic')
        return x
