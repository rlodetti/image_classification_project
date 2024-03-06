from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import matplotlib.pyplot as plt 

import tensorflow as tf
import numpy as np
import random

# Set the random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Average aspect ratio for image resizing
AVG_RATIO = 1.3675285705588607

def create_dataset(directory, ratio=AVG_RATIO):
    """
    Creates a dataset from a directory of images, resizing them to a consistent size.

    Parameters:
    - directory: Path to the directory where the image files are located.
    - ratio: Aspect ratio to determine the width of the images based on a fixed height.

    Returns:
    A tf.data.Dataset object.
    """
    height = 64
    width = int(height * ratio)
    dataset = image_dataset_from_directory(
        directory,
        label_mode='binary',
        color_mode="grayscale",
        batch_size=32,
        image_size=(height, width),
        shuffle=True,
        crop_to_aspect_ratio=True,
        seed=42
    )
    return dataset

def process_dataset(dataset):
    """
    Processes the dataset by rescaling, caching, shuffling, and prefetching.

    Parameters:
    - dataset: A tf.data.Dataset object to process.

    Returns:
    A processed tf.data.Dataset object.
    """
    # Normalizing the dataset by rescaling
    rescale = Rescaling(1./255)
    dataset = dataset.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Cache the dataset to improve performance
    dataset = dataset.cache()

    # Shuffle and prefetch to optimize loading
    dataset = dataset.shuffle(buffer_size=1000, seed=42)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def show_images(train_ds):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i],cmap='gray')
            if labels[i] == 0:
                label = 'Normal'
            else:
                label = 'Pneumonia'
            plt.title(label)
            plt.axis("off")