import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def resize_to_resnet_input(image, label):
    """Resizes images to resnet compatible size (224x224)."""
    return tf.image.resize(image, [224, 224]), label

def get_stl10(split: str, batch_size=128) -> Tuple[tf.data.Dataset, int]:
    """Get STL10 dataset for ResNet model.

    Args:
        split (str, required): defines subset of STL10 to get. Possible options:
        'train', 'unlabelled', 'test'.
        batch_size (int, optional): Size of batches. Defaults to 128.

    Returns:
        Dataset: preprocessed STL10
    """
    ds, ds_info = tfds.load(
        'stl10', split=split, with_info=True, as_supervised=True)
    ds = ds.map(normalize_img,  num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(resize_to_resnet_input,  num_parallel_calls=tf.data.AUTOTUNE)
    # TODO: for now that's only for pretraining.
    ds = ds.map(lambda img, _: img,  num_parallel_calls=tf.data.AUTOTUNE)
    if split == 'test':
        ds = ds.batch(batch_size)
        ds = ds.cache()
    else:
        ds = ds.cache()
        # For true randomness, set the shuffle buffer to the full dataset size.
        ds = ds.shuffle(ds_info.splits[split].num_examples)
        ds = ds.batch(batch_size)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, ds_info.splits[split].num_examples

    

