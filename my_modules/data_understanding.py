import os
import random
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec

def random_image(directory):
    """Select a random image from a specified directory and return the image object."""
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # Filter for common image extensions
    image_files = [f for f in files if f.lower().endswith(('jpeg'))]
    
    if not image_files:
        print("No images found in the directory.")
        return None
    
    # Select a random image file
    random_image_path = os.path.join(directory, random.choice(image_files))
    
    # Open the image
    img = Image.open(random_image_path)

    return img  # Return the PIL image object for further manipulation or information if needed

def show_images(normal_dir,pneumonia_dir):
    """Display random images from Normal directory and Pneumonia directory."""
    image_with_labels = []
    for i in range(6):
        img_dir = random.choice([normal_dir, pneumonia_dir])
        if img_dir == normal_dir:
            rand_image = (random_image(normal_dir),'Normal')
        else:
            rand_image = (random_image(pneumonia_dir),'Pneumonia')
        image_with_labels.append(rand_image)
    plt.figure(figsize=(10, 6))
    for i, (image, label) in enumerate(image_with_labels):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(image,cmap='gray')
        plt.title(label)
        plt.axis("off")

def count_images(directory):
    """
    Counts the number of normal and pneumonia images in a given directory.
    """
    normal_count = len(os.listdir(directory / 'NORMAL'))
    pneumonia_count = len(os.listdir(directory / 'PNEUMONIA'))
    return normal_count, pneumonia_count

def prepare_plot(base_dir, folders):
    """
    Prepares the dataset for analysis, counting normal and pneumonia images.
    """
    data = {'Dataset':folders, 'Normal': [], 'Pneumonia': [], 'Total': []}
    for dataset in data['Dataset']:
        normal, pneumonia = count_images(Path(base_dir) / dataset)
        data['Normal'].append(normal)
        data['Pneumonia'].append(pneumonia)
        data['Total'].append(normal + pneumonia)
    return pd.DataFrame(data)

def bars_data(df):
    """
    Plots the distribution of images using a table and a stacked bar chart.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display DataFrame as a table
    the_table = axs[0].table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 1.2)
    axs[0].axis('off')

    # Plot a bar chart
    axs[1].bar(df['Dataset'], df['Normal'], label='Normal')
    axs[1].bar(df['Dataset'], df['Pneumonia'], bottom=df['Normal'], label='Pneumonia')
    axs[1].set_ylabel('Number of Images')
    axs[1].set_title('Distribution of Images')
    axs[1].legend()

    plt.tight_layout()
    plt.show()