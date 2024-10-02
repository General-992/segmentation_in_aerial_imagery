#!/usr/bin/env python

import os.path as osp

import numpy as np
from PIL import Image
import torch

from torch.utils import data
import tifffile as tiff
import albumentations as A
import scripts

class FLAIRSegBase(data.Dataset):
    """
    Base FLAIR dataset class

    :param root: Dataset root directory.
    :param split: Dataset split (train/val/test).
    :param transform: Whether to apply data augmentation transformations.
    :param patch_size: Patch size for cropping.
    :param test: Whether this is a test dataset.
    """
    class_names = np.array([
        'Soil, Snow, clear - cuts, herbaceous vegetation, brushes, low-vegetation',
        'Pervious and transportation surfaces and sports fields',
        'Buildings, swimming pools, Green houses',
        'Trees',
        'Agricultural surfaces',
        'Water bodies'
    ])

    mean_rgb = np.array([113.775, 118.081, 109.273], dtype=np.float32)
    std_rgb = np.array([52.419, 46.028, 45.260], dtype=np.float32)
    transforms = A.Compose([A.HorizontalFlip(), A.VerticalFlip(),
                            A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                            A.GaussNoise()])
    def __init__(self, root: str, split: str, transform: bool=False, patch_size=256, tile: bool=False):
        self.root = root
        self._transform = transform
        self.patch_size = patch_size
        self.tile = tile
        self.files = []
        imgsets_file = osp.join(self.root, '%s.txt' % split)
        for did in open(imgsets_file):
            img_file, lbl_file = did.strip().split(' ')
            img_file = osp.join(self.root, img_file)
            lbl_file = osp.join(self.root, lbl_file)
            self.files.append({
                'img': img_file,
                'msk': lbl_file,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img = tiff.imread(self.files[idx]['img'])
        img = img[..., :3]
        mask = Image.open(self.files[idx]['msk'])
        mask = np.asarray(mask)
        mask = self.mask_encode(mask)

        if self._transform:
            img, mask = self.transform(img, mask)
        else:
            img = img.astype(np.float32)
            img = self._normalize(img)
            img = np.transpose(img, (2, 0, 1))

        if self.patch_size:
            if not self.tile:
                img, mask = scripts.utils.patch_sample(img=img, mask=mask, patch_size=self.patch_size)
            else:
                img, mask = scripts.utils.patch_divide(img=img, mask=mask, patch_size=self.patch_size)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask
    def transform(self, img, mask):
        aug = self.transforms(image=img, mask=mask)
        img = aug['image'].astype(np.float32)
        img = self._normalize(img)
        img = np.transpose(img, (2, 0, 1))
        mask = aug['mask']
        return img, mask
    def untransform(self, img, mask):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img *= self.std_rgb
        img += self.mean_rgb
        img = img.astype(np.uint8)
        mask = mask.numpy()
        return img, mask

    def _normalize(self, img):
        img -= self.mean_rgb
        img /= self.std_rgb
        return img
    def mask_encode(self, mask):
        """
        Encode mask pixel distinct values to discrete class numbers
        :param mask:
        mask with ([0, 1, 2, .., num_classes]), original pixels values
        :return:
        new_mask with ([0, 1, 2, .., num_classes]), new pixels values
        """

        class_mapping = {
            4: 0, 10: 0, 14: 0, 15: 0, 8: 0,   # Soil, Snow, clear-cuts, herbaceous vegetation, bushes
            2: 1, 3: 1,                        # Pervious, Impervious and transportation surfaces and sports fields
            1: 2, 13: 2, 18: 2,                # Buildings, swimming pools, Green houses
            6: 3, 7: 3, 16: 3, 17: 3,          # Trees
            9: 4, 11: 5, 12: 4,                # Agricultural surfaces
            5: 5,                              # Water bodies
        }
        # Initialize a new mask with the same shape
        new_mask = np.zeros_like(mask)

        # Reassign classes according to the mapping
        for old_class, new_class in class_mapping.items():
            new_mask[mask == old_class] = new_class
        return new_mask

class FLAIRSegMeta(FLAIRSegBase):
    """
    Inherits from FLAIRSegBase and adds functionality to track image metadata such as
    camera type and capture month.

    :param root: Dataset root directory.
    :param split: Dataset split (train/val/test).
    :param metadata: A dictionary containing metadata for each image (camera type, date, etc.).
    :param transform: Whether to apply data augmentation transformations.
    :param patch_size: Patch size for cropping.
    :param test: Whether this is a test dataset.
    """
    def __init__(self, root, split, metadata, transform=False, patch_size=256, test=False):
        super().__init__(root, split, transform, patch_size, test)
        self.metadata = metadata

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        img_file = self.files[idx]['img']
        img_name = osp.basename(img_file)  # extract the base name of the image
        img_key = osp.splitext(img_name)[0]

        if img_key in self.metadata:
            camera = self.metadata[img_key].get('camera', 'Unknown')
            date = self.metadata[img_key].get('date', 'Unknown')
            month = date.split('-')[1] if date != 'Unknown' else 'Unknown'
        else:
            raise Exception(f'{img_key} not found in metadata')
        return img, mask, int(month), camera

if __name__ == '__main__':
    root = osp.expanduser('~/datasets/flair_dataset')

    file_path = osp.join(root, 'flair-1_metadata_aerial.json')
    import json
    with open(file_path, 'r') as file:
        metadata = json.load(file)

    test_meta = FLAIRSegMeta(root=root,split='val', metadata=metadata, transform=False)
    print(test_meta[0])

