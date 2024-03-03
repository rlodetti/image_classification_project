import os
import shutil
import shutil
from pathlib import Path
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    return DataFrame(data)

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


def pies(df):
    df['norm_rate']=round((df['Normal']/df['Total'])*100,2)
    df['pneu_rate']=round((df['Pneumonia']/df['Total'])*100,2)
    fig, ax = plt.subplots(1,3, figsize=(9,3))
    ax[0].pie(list(df.iloc[0,4:6]), labels=['Normal','Pneumonia'], autopct='%1.f%%', textprops={'fontsize': 12, 'color': 'white', 'fontweight':'bold'})
    ax[0].set_title('Train')
    ax[1].pie(list(df.iloc[1,4:6]), labels=['Normal','Pneumonia'], autopct='%1.f%%', textprops={'fontsize': 12, 'color': 'white'})
    ax[1].set_title('Validation')
    ax[2].pie(list(df.iloc[2,4:6]), labels=['Normal','Pneumonia'], autopct='%1.f%%', textprops={'fontsize': 12, 'color': 'white'})
    ax[2].set_title('Test')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()