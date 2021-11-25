import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple
from data_processing.augmentations import DataAug

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def resize_to_resnet_input(image, label):
    """Resizes images to resnet compatible size (224x224)."""
    return tf.image.resize(image, [224, 224]), label

def get_stl10(
    split: str, 
    batch_size=128, 
    include_labels=False) -> Tuple[tf.data.Dataset, int]:
    """Get STL10 dataset for ResNet model.

    Args:
        split (str, required): defines subset of STL10 to get. Possible options:
        'train', 'unlabelled', 'test'.
        batch_size (int, optional): Size of batches. Defaults to 128.
        include_labels (bool, optional): If true output dataset consists of 
        tuples (image, label), otherwise it only includes only labels. 
        Defaults to False.

    Returns:
        Dataset: preprocessed STL10
    """

    ds, ds_info = tfds.load(
        'stl10', split=split, with_info=True, as_supervised=True)
    ds = ds.map(normalize_img,  num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(resize_to_resnet_input,  num_parallel_calls=tf.data.AUTOTUNE)
    # Map to return only images:
    if not include_labels:
        ds = ds.map(lambda img, _: img,  num_parallel_calls=tf.data.AUTOTUNE)
    else: 
        data_aug = DataAug(batch_size=batch_size)
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

    

