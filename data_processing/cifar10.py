import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple
from data_processing.augmentations import DataAugSmall

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    # TODO: Check if per_image_standarization is needed.
    return ((tf.cast(image, tf.float32) / 255.),
        label)

def get_cifar10(
    split: str, 
    batch_size=128, 
    include_labels=False) -> Tuple[tf.data.Dataset, int]:
    """Get STL10 dataset for ResNet model.

    Args:
        split (str, required): defines subset of STL10 to get. Possible options:
        'train', 'test'.
        batch_size (int, optional): Size of batches. Defaults to 128.
        include_labels (bool, optional): If true output dataset consists of 
        tuples (image, label), otherwise it only includes only labels. 
        Defaults to False.

    Returns:
        Dataset: preprocessed CIFAR10
    """

    ds, ds_info = tfds.load(
        'cifar10', split=split, with_info=True, as_supervised=True)
    ds = ds.map(normalize_img,  num_parallel_calls=tf.data.AUTOTUNE)
    # Map to return only images:
    if not include_labels:
        ds = ds.map(lambda img, _: img,  num_parallel_calls=tf.data.AUTOTUNE)
    else: 
        data_aug = DataAugSmall(batch_size=None)
        ds = ds.map(lambda x, y: (data_aug.augment(x), y), 
            num_parallel_calls=tf.data.AUTOTUNE)

    if split == 'test':
        ds = ds.batch(batch_size)
        ds = ds.cache()
    else:
        ds = ds.cache()
        # For true randomness, set the shuffle buffer to the full dataset size.
        # if it fits into memory, uncomment the following line.
        # ds = ds.shuffle(ds_info.splits[split].num_examples)
        ds = ds.shuffle(1000)
        ds = ds.batch(batch_size)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, ds_info.splits[split].num_examples

    

