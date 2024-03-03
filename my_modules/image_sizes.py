from PIL import Image
import os
from numpy import mean, concatenate

def get_image_sizes(directory):
    """Collect widths and heights of images with given extensions in directory."""
    sizes = [(img.size) for root, dirs, files in os.walk(directory)
             for file in files if file.lower().endswith('jpeg')
             for img in (Image.open(os.path.join(root, file)),)]
    return zip(*sizes)  # Unzips the sizes into two lists: widths and heights

def print_image_statistics(widths, heights):
    """Print statistics for a collection of image widths and heights."""
    num_images = len(widths)
    avg_width = round(mean(widths))
    avg_height = round(mean(heights))
    avg_aspect_ratio = mean(widths) / mean(heights)
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

train_dir = 'data/chest_xray/new_train/'
test_dir = 'data/chest_xray/new_test/'

train_widths, train_heights = get_image_sizes(train_dir)
test_widths, test_heights = get_image_sizes(test_dir)

widths = concatenate((train_widths, test_widths))
heights = concatenate((train_heights, test_heights))

avg_ratio = print_image_statistics(widths, heights)