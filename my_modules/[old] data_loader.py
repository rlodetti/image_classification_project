import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
tf.random.set_seed(42)

def create_dataset(directory):
    dataset = image_dataset_from_directory(directory,
                                        label_mode='binary',
                                        color_mode="grayscale", # will save memory as images are already in grayscale
                                        batch_size=32, # selecting default value
                                        image_size=(256,256), # selecting default value
                                        shuffle=True,
                                        seed=42)
    return dataset
    
def process_dataset(dataset):
    # Define the rescaling layer
    rescale = Rescaling(1./255)
    
    # Normalizing the dataset
    dataset = dataset.map(lambda x, y: (rescale(x), y))
    
    # improves speed by only having to read the dataset for the first epoch
    dataset = dataset.cache()
    
    # increases generalization by shuffling elements each epoch
    dataset = dataset.shuffle(buffer_size=1000, seed=42)
    
    # this automatically adjusts the number of batches
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
