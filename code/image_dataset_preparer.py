import os
import shutil
from sklearn.model_selection import train_test_split

# Base directory
base_dir = "../data/chest_xray"

# Labels
labels = ["NORMAL", "PNEUMONIA"]

def create_directory_structure(base_path, dir_names):
    """Create directory structure for dataset processing."""
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

def collect_and_combine_data(source_dirs, dest_dir):
    """Collect and combine data from multiple directories into a single directory."""
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

def split_and_distribute_data(source_dir, target_dirs, split_ratio=(0.8, 0.1, 0.1)):
    """Split and distribute data into train, validation, and test directories."""
    for label in labels:
        files = [os.path.join(source_dir, label, f) for f in os.listdir(os.path.join(source_dir, label)) if os.path.isfile(os.path.join(source_dir, label, f))]
        
        train_files, test_files = train_test_split(files, test_size=split_ratio[1] + split_ratio[2], random_state=42)
        val_files, test_files = train_test_split(test_files, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=42)
        
        for files, dir_name in zip([train_files, val_files, test_files], ['new_train', 'new_val', 'new_test']):
            copy_files(files, target_dirs[dir_name], label)

def copy_files(files, dest_dir, label):
    """Copy files to the designated directory."""
    for file in files:
        try:
            shutil.copy(file, os.path.join(dest_dir, label))
        except IOError as e:
            print(f"Unable to copy file. {e}")
        except:
            print("Unexpected error:", sys.exc_info())

# Initialize directories
dir_names = ['new_train', 'new_val', 'new_test', 'temp_combined']
directories = create_directory_structure(base_dir, dir_names)

# Original directories
original_dirs = [os.path.join(base_dir, name) for name in ['train', 'test', 'val']]

# Collect and combine data into a temporary directory
collect_and_combine_data(original_dirs, directories['temp_combined'])

# Split and distribute data
split_and_distribute_data(directories['temp_combined'], directories)