import os
import os.path as osp
import tifffile as tiff
import numpy as np

img_root = osp.expanduser('~/datasets/flair_dataset')
IMAGE_PATH = osp.join(img_root, 'img')
LABEL_PATH = osp.join(img_root, 'msk')

image_files = [f for f in os.listdir(IMAGE_PATH) if f.startswith('IMG') and f.endswith('.tif')]
mask_files = [f for f in os.listdir(LABEL_PATH) if f.startswith('MSK') and f.endswith('.tif')]

X = []
for image_file in image_files:
    corresponding_mask = image_file.replace('IMG', 'MSK')
    if corresponding_mask in mask_files:
        image_path = os.path.join(IMAGE_PATH, image_file)
        mask_path = os.path.join(LABEL_PATH, corresponding_mask)
        X.append((image_path, mask_path))
    else:
        raise FileNotFoundError

print('Total Images: ', len(X))

from  sklearn.model_selection import train_test_split
train, val = train_test_split(X, test_size=0.20, random_state=19)

# Train file path to save the list as a .txt file
output_train_path = os.path.join(img_root, 'train.txt')

with open(output_train_path, 'w') as f:
    for img_path, msk_path in train:
        # Write the image and mask path to the file, separated by a space or any other delimiter
        f.write(f"{img_path} {msk_path}\n")

# File path to save the list as a .txt file
output_val_path = os.path.join(img_root, 'val.txt')

# Open the file in write mode
with open(output_val_path, 'w') as f:
    for img_path, msk_path in val:
        # Write the image and mask path to the file, separated by a space or any other delimiter
        f.write(f"{img_path} {msk_path}\n")

print("Lists of image names saved to train.txt and val.txt successfully.")