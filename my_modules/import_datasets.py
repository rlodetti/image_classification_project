from tensorflow import random, compat, data
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# Set random seed and ignore warnings for consistent and clean output
random.set_seed(42)
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)  # Use TensorFlow's way to ignore warnings

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
    dataset = dataset.map(lambda x, y: (rescale(x), y), num_parallel_calls=data.AUTOTUNE)

    # Cache the dataset to improve performance
    dataset = dataset.cache()

    # Shuffle and prefetch to optimize loading
    dataset = dataset.shuffle(buffer_size=1000, seed=42)
    dataset = dataset.prefetch(data.AUTOTUNE)

    return dataset

