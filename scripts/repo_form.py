import os
import shutil

"""
Script to combine images and labels from different years into unified 'img' and 'msk' directories.

This script traverses through the source directories containing images and labels from various years,
and copies all the files into a single repository structure:
- 'img/' for images
- 'msk/' for masks/labels

Directories are created if they do not exist.

Variables:
- images_source_dir: Source directory containing subfolders of images (e.g., '~/datasets/flair_aerial_train/flair_aerial_train').
- labels_source_dir: Source directory containing subfolders of labels (e.g., '~/datasets/flair_aerial_train').
- common_repository: Target directory where 'img' and 'msk' will be created (e.g., '~/datasets/flair_dataset').

Usage:
- No command-line arguments are required. Simply run the script, and it will copy all images and labels
  from the source directories to the target directories.

Notes:
- The script assumes that images are stored in subdirectories named 'img' and labels are stored in subdirectories named 'msk'.
"""

# Source directories
root = os.path.expanduser('~/datasets')
images_source_dir = os.path.join(root, 'flair_aerial_train/flair_aerial_train')
labels_source_dir = os.path.join(root, 'flair_aerial_train')

# Target directory
common_repository = os.path.join(root, 'flair_dataset')

# Common repository directory if it doesn't exist
if not os.path.exists(common_repository):
    os.makedirs(common_repository)

# Target subdirectories within the common repository
images_target_dir = os.path.join(common_repository, 'img')
labels_target_dir = os.path.join(common_repository, 'msk')

# Create target directories if they don't exist
os.makedirs(images_target_dir, exist_ok=True)
os.makedirs(labels_target_dir, exist_ok=True)

# Function to copy files from source to target
def copy_files(source_dir, target_dir, subfolder):
    for root, dirs, files in os.walk(source_dir):
        if subfolder in root:
            for file in files:
                file_path = os.path.join(root, file)
                shutil.copy(file_path, target_dir)

# Copy images
copy_files(images_source_dir, images_target_dir, 'img')

# Copy labels
copy_files(labels_source_dir, labels_target_dir, 'msk')

print("All files have been moved successfully.")