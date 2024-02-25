import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_and_preprocess_dataset(self, directory):
        dataset = image_dataset_from_directory(
            directory,
            label_mode=self.config["label_mode"],
            batch_size=self.config["batch_size"],
            image_size=self.config["image_size"],
            color_mode=self.config["color_mode"]
        )
        return dataset.cache().shuffle(buffer_size=self.config["shuffle_buffer_size"]).prefetch(tf.data.AUTOTUNE)

def get_datasets(directories, config, preprocessing_layer=None):
    data_loader = DataLoader(config)
    train_ds = data_loader.load_and_preprocess_dataset(directories["train"], preprocessing_layer)
    test_ds = data_loader.load_and_preprocess_dataset(directories["test"], preprocessing_layer)
    val_ds = data_loader.load_and_preprocess_dataset(directories["val"], preprocessing_layer)
    return train_ds, test_ds, val_ds