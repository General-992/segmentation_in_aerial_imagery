import os
import os.path as osp
from  sklearn.model_selection import train_test_split
"""
This script processes a dataset of FLAIR images and corresponding masks for either training/validation or testing. It creates file lists for each split (train, val, or test) and saves them in text files for later use in model training or evaluation.

Usage:
    - For generating train and validation splits:
        main('train_val')
    - For generating test split:
        main('test')

Notes:
    - The script assumes the dataset is organized into two directories:
        'img': containing images with the prefix 'IMG' and suffix '.tif'.
        'msk': containing masks with the prefix 'MSK' and suffix '.tif'.
    - The paths for the images and masks are saved to 'train.txt', 'val.txt', or 'test.txt' in the respective dataset directory.
    - The dataset root is expected to be at '~/datasets/flair_dataset' or '~/datasets/flair_dataset_tests', depending on the split.
    - The train/val split is done with an 80/20 ratio.
"""
def main(split: str = 'train_val'):
    if split == 'train_val':
        img_root = osp.expanduser('~/flair_dataset_train_val')
    elif split == 'test':
        img_root = osp.expanduser('~/flair_dataset_test')
    else:
        raise Exception('Split must be either "train_val" or "test"')
    IMAGE_PATH = osp.join(img_root, 'img')
    LABEL_PATH = osp.join(img_root, 'msk')

    image_files = [f for f in os.listdir(IMAGE_PATH) if f.startswith('IMG') and f.endswith('.tif')]
    mask_files = [f for f in os.listdir(LABEL_PATH) if f.startswith('MSK') and f.endswith('.tif')]

    X = []
    for image_file in image_files:
        corresponding_mask = image_file.replace('IMG', 'MSK')
        if corresponding_mask in mask_files:
            image_path = os.path.join('img', image_file)
            mask_path = os.path.join('msk', corresponding_mask)
            X.append((image_path, mask_path))
        else:
            raise FileNotFoundError

    print('Total Images: ', len(X))

    if split == 'train_val':
        train, val = train_test_split(X, test_size=0.20, random_state=19)
        output_train_path = os.path.join(img_root, 'train.txt')

        with open(output_train_path, 'w') as f:
            for img_path, msk_path in train:
                f.write(f"{img_path} {msk_path}\n")

        output_val_path = os.path.join(img_root, 'val.txt')

        with open(output_val_path, 'w') as f:
            for img_path, msk_path in val:
                f.write(f"{img_path} {msk_path}\n")
    elif split == 'test':
        output_test_path = os.path.join(img_root, 'test.txt')
        with open(output_test_path, 'w') as f:
            for img_path, msk_path in X:
                f.write(f"{img_path} {msk_path}\n")

    print("Lists of image names saved to train.txt and val.txt successfully.")

if __name__ == '__main__':
    splits = ['train_val', 'test']
    for split in splits:
        main(split)