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
- images_source_dir: Source directory containing subfolders of images.
- labels_source_dir: Source directory containing subfolders of labels.
- common_repository: Target directory where 'img' and 'msk' will be created.

Usage:
- No command-line arguments are required. Run the script, and it will copy all images and labels
  from the source directories to the target directories.

Notes:
- The script assumes that images are stored in subdirectories named 'img' and labels are stored in subdirectories named 'msk'.
"""

# Source train directories
root = os.path.expanduser('~/datasets')
images_source_dir = os.path.join(root, 'flair_aerial_train/flair_aerial_train')
labels_source_dir = os.path.join(root, 'flair_aerial_train')

common_repository = os.path.join(root, 'flair_dataset_train_val')

if not os.path.exists(common_repository):
    os.makedirs(common_repository)

images_target_dir = os.path.join(common_repository, 'img')
labels_target_dir = os.path.join(common_repository, 'msk')

os.makedirs(images_target_dir, exist_ok=True)
os.makedirs(labels_target_dir, exist_ok=True)
def copy_files(source_dir, target_dir, subfolder):
    for root, dirs, files in os.walk(source_dir):
        if subfolder in root:
            for file in files:
                file_path = os.path.join(root, file)
                shutil.copy(file_path, target_dir)

copy_files(images_source_dir, images_target_dir, 'img')
copy_files(labels_source_dir, labels_target_dir, 'msk')

## Test Dataset
images_source_dir = os.path.join(root, 'flair_1_aerial_test')
labels_source_dir = os.path.join(root, 'flair_1_labels_test')

common_repository = os.path.join(root, 'flair_dataset_test')

if not os.path.exists(common_repository):
    os.makedirs(common_repository)

images_target_dir = os.path.join(common_repository, 'img')
labels_target_dir = os.path.join(common_repository, 'msk')

os.makedirs(images_target_dir, exist_ok=True)
os.makedirs(labels_target_dir, exist_ok=True)

copy_files(images_source_dir, images_target_dir, 'img')
copy_files(labels_source_dir, labels_target_dir, 'msk')

print("All files have been moved successfully.")