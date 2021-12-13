"""
Data augmentations used during training BYOL.
For details, see Appendix B in https://arxiv.org/pdf/2006.07733v3.pdf
"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class DataAugSmall():
    """Perform augmentation according to DirectPred implementation for CIFAR10.
    
    A random patch of the image is selected, with an area uniformly sampled 
    between 80% and 100% of that of the original image, and an aspect ratio 
    logarithmically sampled between 3/4 and 4/3. This patch is then resized to 
    the target size of initial size using bicubic interpolation.
    """

    def __init__(self, width=32, height=32, batch_size=128) -> None:
        self.width = width
        self.height = height
        self.current_area = self.width * self.height
        self.batch_size = batch_size
        # Variance of CIFAR10 for each channel.
        var = tf.constant([0.2023, 0.1994, 0.2010])
        self.var = self.reshape_variance(batch_size, var)
        # Mean of CIFAR10 for each channel.
        self.mean = tf.constant([0.4914, 0.4822, 0.4465])

    def reshape_variance(self, batch_size, var):
        # Variance of CIFAR10 for each channel.
        if batch_size is None:
            # If we do not have batches the data is 3D.
            var = tf.reshape(var, shape=[1,1,3])
            var = tf.repeat(var,axis=0,repeats=self.width)
            var = tf.repeat(var,axis=1,repeats=self.height)
        else:
            var = tf.reshape(var, shape=[1,1,1,3])
            var = tf.repeat(var,axis=0,repeats=batch_size)
            var = tf.repeat(var,axis=1,repeats=self.width)
            var = tf.repeat(var,axis=2,repeats=self.height)
        return var

    def normalize(self, x):
        x = tf.math.subtract(x, self.mean)
        x = tf.math.divide(x, self.var)
        return x

    def augment(self, x, s=1.0):
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

        x = tf.image.random_flip_left_right(x)
        if self.batch_size is None:
            x = tf.image.random_crop(x, size=[h, w, 3]) 
        else:
            x = tf.image.random_crop(x, size=[self.batch_size, h, w, 3]) 
        x = tf.image.resize(x, [self.width, self.height], method='bicubic')

        if tf.random.uniform([], minval=0.0, maxval=1.0) < 0.8:
            brightness_factor = tf.random.uniform([], minval=0.6, maxval=1.4)
            x = tf.multiply(x, brightness_factor)
            # x = tf.image.random_brightness(x,max_delta=0.8)
            # This might not work as in torch ColorJitter 
            x = tf.image.adjust_contrast(x, 0.8)
            x = tf.image.adjust_saturation(x, 0.8)
            x = tf.image.random_hue(x, max_delta=0.2)
            # x = tf.clip_by_value(x,0,1)

        if tf.random.uniform([1], minval=0.0, maxval=1.0) < 0.2:
            x = tf.repeat(tf.image.rgb_to_grayscale(x), 3, axis=-1)
        x = self.normalize(x)
        return x


class DataAug():
    """Data augmentation for bigger images e.g. STL10."""

    def __init__(self, width=96, height=96, batch_size=128) -> None:
        self.width = width
        self.height = height
        self.current_area = self.width * self.height
        self.batch_size = batch_size

    def augment(self, x, s=1.0):
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

        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_crop(x, size=[self.batch_size, h, w, 3]) 
        x = tf.image.resize(x, [self.width, self.height], method='bicubic')


        if tf.random.uniform([], minval=0.0, maxval=1.0) < 0.8:
            x = tf.image.random_brightness(x,max_delta=0.8*s)
            x = tf.image.random_contrast(x,lower=1-0.8*s,upper=1+0.8*s)
            x = tf.image.random_saturation(x,lower=1-0.8*s,upper=1+0.8*s)
            x = tf.image.random_hue(x,max_delta=0.2*s)
            x = tf.clip_by_value(x,0,1)

        if tf.random.uniform([1], minval=0.0, maxval=1.0) < 0.2:
            x = tf.repeat(tf.image.rgb_to_grayscale(x), 3, axis=-1)

        x = tfa.image.gaussian_filter2d(
            x, 
            filter_shape=9, # 0.1 * STL10 width 
            sigma=np.random.uniform(low=0.1, high=2.0))

        return x
