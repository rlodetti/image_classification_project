import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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