# Standard library imports
import os
import random
from pathlib import Path

# Third-party library imports
from PIL import Image
import matplotlib.pyplot as plt
from pandas import DataFrame

def random_image(directory):
    """
    Select a random image from a specified directory.
    
    Parameters:
        directory (str): The path to the directory from which to select an image.
        
    Returns:
        Image object if an image is found, otherwise None.
    """
    # Generate a list of files with common image extension 'jpeg' in the directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith('jpeg') 
                   and os.path.isfile(os.path.join(directory, f))]
    
    random_image_path = os.path.join(directory, random.choice(image_files))
    img = Image.open(random_image_path)
    return img

def show_images(normal_dir, pneumonia_dir):
    """
    Displays a set of random images from two directories.
    
    Parameters:
        normal_dir (str): The path to the directory containing normal images.
        pneumonia_dir (str): The path to the directory containing pneumonia images.
    """
    images_with_labels = []
    for _ in range(6):
        img_dir = random.choice([normal_dir, pneumonia_dir])
        label = 'Normal' if img_dir == normal_dir else 'Pneumonia'
        image = random_image(img_dir)
        images_with_labels.append((image, label))
    
    plt.figure(figsize=(10, 6))
    for i, (image, label) in enumerate(images_with_labels):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.axis("off")
    plt.show()

def count_images(directory):
    """
    Counts the number of normal and pneumonia images in a given directory.
    
    Parameters:
        directory (Path): The path object to the directory containing image subdirectories.
        
    Returns:
        Tuple[int, int]: A tuple containing counts of normal and pneumonia images.
    """
    normal_count = len(list((directory / 'NORMAL').glob('*')))
    pneumonia_count = len(list((directory / 'PNEUMONIA').glob('*')))
    return normal_count, pneumonia_count

def prepare_plot(base_dir, folders):
    """
    Prepares the data for analysis by counting the images in each category.
    
    Parameters:
        base_dir (str): The base directory containing dataset folders.
        folders (list): A list of folder names within the base directory to analyze.
        
    Returns:
        DataFrame: A pandas DataFrame containing the counts of images.
    """
    data = {'Dataset': [], 'Normal': [], 'Pneumonia': [], 'Total': []}
    for folder in folders:
        path = Path(base_dir) / folder
        normal, pneumonia = count_images(path)
        data['Dataset'].append(folder)
        data['Normal'].append(normal)
        data['Pneumonia'].append(pneumonia)
        data['Total'].append(normal + pneumonia)
    
    return DataFrame(data)

def plot_data_distribution(dataframe):
    """
    Plots the distribution of normal and pneumonia images in the dataset.
    
    Parameters:
        dataframe (DataFrame): The pandas DataFrame containing the image counts.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the DataFrame as a table
    the_table = axs[0].table(cellText=dataframe.values, colLabels=dataframe.columns, 
                             loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 1.2)
    axs[0].axis('off')
    
    # Plot a stacked bar chart
    axs[1].bar(dataframe['Dataset'], dataframe['Normal'], label='Normal')
    axs[1].bar(dataframe['Dataset'], dataframe['Pneumonia'], bottom=dataframe['Normal'], 
               label='Pneumonia')
    axs[1].set_ylabel('Number of Images')
    axs[1].set_title('Distribution of Images')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
