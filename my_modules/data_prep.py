import os
import shutil
from pathlib import Path
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf
import numpy as np
import random

from PIL import Image


# Set the random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def create_directory_structure(base_path, dir_names, labels):
    directories = {}
    for dir_name in dir_names:
        dir_path = os.path.join(base_path, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for label in labels:
            label_dir = os.path.join(dir_path, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
        directories[dir_name] = dir_path
    return directories

def collect_and_combine_data(source_dirs, dest_dir,labels):
    for label in labels:
        files = []
        for dir_path in source_dirs:
            label_dir = os.path.join(dir_path, label)
            files.extend([os.path.join(label_dir, f) for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
        
        combined_dir = os.path.join(dest_dir, label)
        for file in files:
            try:
                shutil.copy(file, combined_dir)
            except IOError as e:
                print(f"Unable to copy file. {e}")
            except:
                print("Unexpected error:", sys.exc_info())

def split_and_distribute_data(source_dir, target_dirs, labels, split_ratio=(0.8, 0.1, 0.1)):
    for label in labels:
        files = [os.path.join(source_dir, label, f) for f in os.listdir(os.path.join(source_dir, label)) if os.path.isfile(os.path.join(source_dir, label, f))]
        
        train_files, test_files = train_test_split(files, test_size=split_ratio[1] + split_ratio[2], random_state=42)
        val_files, test_files = train_test_split(test_files, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=42)
        
        for files, dir_name in zip([train_files, val_files, test_files], ['new_train', 'new_val', 'new_test']):
            copy_files(files, target_dirs[dir_name], label)
    
    # delete temporary directory
    shutil.rmtree(source_dir)

def copy_files(files, dest_dir, label):
    for file in files:
        try:
            shutil.copy(file, os.path.join(dest_dir, label))
        except IOError as e:
            print(f"Unable to copy file. {e}")
        except:
            print("Unexpected error:", sys.exc_info())

def run_redistribution():
    # Base directory
    base_dir = "data/chest_xray"
    
    # Labels
    labels = ["NORMAL", "PNEUMONIA"]
    
    new_dir_names = ['new_train', 'new_val', 'new_test', 'temp_combined']
    
    # Initialize directories
    new_dirs = create_directory_structure(base_dir, new_dir_names, labels)
    
    # Original directories
    original_dirs = [os.path.join(base_dir, name) for name in ['train', 'test', 'val']]
    
    # Collect and combine data into a temporary directory
    collect_and_combine_data(original_dirs, new_dirs['temp_combined'], labels)
    
    # Split and distribute data
    split_and_distribute_data(new_dirs['temp_combined'], new_dirs, labels, split_ratio=(0.8, 0.1, 0.1))


def pie(old_df, new_df):
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    
    # Create a GridSpec with 2 rows and 4 columns
    # The first column will be used for the row titles
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 3, 3, 3])
    
    # Row titles
    row_titles = ['Before', 'After']
    col_titles = ['Train', 'Validation', 'Test']
    
    # Create an empty subplot for each row title and set the title
    for i, title in enumerate(row_titles):
        ax = fig.add_subplot(gs[i, 0])
        ax.text(0.5, 0.5, title, va='center', ha='center', fontsize=14, fontweight='bold')
        ax.axis('off')  # Hide the axes
    
    # Now create the actual plots for the remaining cells
    for row in range(2):
        for col in range(1, 4):  # Start from 1 to skip the title columns
            ax = fig.add_subplot(gs[row, col])
            n = col-2
            if row == 0:
                ax.set_title(col_titles[n],fontsize=12, fontweight='bold')
                ax.pie(list(old_df.iloc[n,1:3]), labels=['Normal','Pneumonia'], autopct='%1.f%%',
                       textprops={'fontsize': 12, 'color': 'white', 'fontweight':'bold'})
            else:
                ax.pie(list(new_df.iloc[n,1:3]), labels=['Normal','Pneumonia'], autopct='%1.f%%',
                       textprops={'fontsize': 12, 'color': 'white', 'fontweight':'bold'})
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def compare_bar(old_df, new_df):
    old_df['total_rate'] = round((old_df['Total'] / old_df['Total'].sum()) * 100, 1)
    new_df['total_rate'] = round((new_df['Total'] / new_df['Total'].sum()) * 100)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    
    # Plotting for 'Before' distribution
    bars_before = ax[0].bar(old_df['Dataset'], old_df['Normal'], label='Normal')
    ax[0].bar(old_df['Dataset'], old_df['Pneumonia'], bottom=old_df['Normal'], label='Pneumonia')
    ax[0].set_ylabel('Number of Images')
    ax[0].set_title('Before')
    
    # Plotting for 'After' distribution
    bars_after = ax[1].bar(new_df['Dataset'], new_df['Normal'], label='Normal')
    ax[1].bar(new_df['Dataset'], new_df['Pneumonia'], bottom=new_df['Normal'], label='Pneumonia')
    ax[1].set_title('After')
    
    plt.legend()

    # Adding text annotations for 'Before' distribution
    for dataset, rate in zip(old_df['Dataset'], old_df['total_rate']):
        total_height = old_df[old_df['Dataset'] == dataset]['Normal'].values[0] + old_df[old_df['Dataset'] == dataset]['Pneumonia'].values[0]
        ax[0].text(old_df[old_df['Dataset'] == dataset].index[0], total_height, f'{rate}%', ha='center', va='bottom')
    
    # Adding text annotations for 'After' distribution
    for dataset, rate in zip(new_df['Dataset'], new_df['total_rate']):
        total_height = new_df[new_df['Dataset'] == dataset]['Normal'].values[0] + new_df[new_df['Dataset'] == dataset]['Pneumonia'].values[0]
        ax[1].text(new_df[new_df['Dataset'] == dataset].index[0], total_height, f'{rate}%', ha='center', va='bottom')

# Average aspect ratio for image resizing
AVG_RATIO = 1.3675285705588607

def create_dataset(directory, ratio=AVG_RATIO, process=False, is_training=True):
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
        shuffle=is_training,  # Shuffle only if it's the training dataset
        crop_to_aspect_ratio=True,
        seed=42
    )
    if process:
        return process_dataset(dataset)
    else:
        return dataset

def process_dataset(dataset, is_training=True):
    """
    Processes the dataset by rescaling, (optionally shuffling for validation/test), caching, and prefetching.

    Parameters:
    - dataset: A tf.data.Dataset object to process.
    - is_training: A boolean flag to indicate if the dataset is for training. True by default.

    Returns:
    A processed tf.data.Dataset object.
    """
    # Normalizing the dataset by rescaling
    rescale = Rescaling(1./255)
    dataset = dataset.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        # Shuffle only if it's the training dataset
        dataset = dataset.shuffle(buffer_size=1000, seed=42)

    # Cache the dataset to improve performance
    dataset = dataset.cache()

    # Prefetch to optimize loading
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset



def show_images(train_ds):
    plt.figure(figsize=(10, 6))
    for images, labels in train_ds.take(1):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i],cmap='gray')
            if labels[i] == 0:
                label = 'Normal'
            else:
                label = 'Pneumonia'
            plt.title(label)
            plt.axis("off")

def get_image_sizes(directory):
    """Collect widths and heights of images with given extensions in directory."""
    sizes = [(img.size) for root, dirs, files in os.walk(directory)
             for file in files if file.lower().endswith('jpeg')
             for img in (Image.open(os.path.join(root, file)),)]
    return zip(*sizes)  # Unzips the sizes into two lists: widths and heights

def print_image_statistics(widths, heights):
    """Print statistics for a collection of image widths and heights."""
    num_images = len(widths)
    avg_width = round(np.mean(widths))
    avg_height = round(np.mean(heights))
    avg_aspect_ratio = np.mean(widths) / np.mean(heights)
    max_width = max(widths)
    max_height = max(heights)
    min_width = min(widths)
    min_height = min(heights)
    
    print(f"Number of images: {num_images}")
    print(f"Average width: {avg_width}")
    print(f"Average height: {avg_height}")
    print(f"Average aspect ratio: {avg_aspect_ratio}")
    print(f"Max width: {max_width}")
    print(f"Max height: {max_height}")
    print(f"Min width: {min_width}")
    print(f"Min height: {min_height}")
    return avg_aspect_ratio
