#!/usr/bin/env python

import collections
import os.path as osp
import os

import numpy as np
from PIL import Image
import scipy.io
import torch
from torchvision.transforms import transforms as T
from torch.utils import data
from random import randint
import tifffile as tiff
import albumentations as A
import scripts

class FLAIRSegBase(data.Dataset):
    class_names = np.array([
        'Soil, Snow, clear - cuts, herbaceous vegetation',
        'Pervious and transportation surfaces and sports fields',
        'Buildings, swimming pools, Green houses',
        'Trees',
        'Bushes',
        'Agricultural surfaces',
        'Water bodies'
    ])

    mean_rgb = np.array([113.775, 118.081, 109.273], dtype=np.float32)
    std_rgb = np.array([52.419, 46.028, 45.260], dtype=np.float32)
    transforms = A.Compose([A.HorizontalFlip(), A.VerticalFlip(),
                            A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                            A.GaussNoise()])
    def __init__(self, root, split, transform=False, patch_size=256, test=False):
        self.root = root
        self._transform = transform
        self.patch_size = patch_size
        self.test = test

        # defaultdict creates dict that automatically expands creating a key and a respectitve list
        self.files = []
        imgsets_file = osp.join(self.root, '%s.txt' % split)
        for did in open(imgsets_file):
            img_file, lbl_file = did.strip().split(' ')
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

            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()

            if self.patch_size:
                if not self.test:
                    img, mask = scripts.utils.patch_sample(img=img, mask=mask, patch_size=self.patch_size)
                else:
                    img, mask = scripts.utils.patch_divide(img=img, mask=mask, patch_size=self.patch_size)
        return img, mask
    def transform(self, img, mask):
        aug = self.transforms(image=img, mask=mask)
        img = aug['image'].astype(np.float32)
        img -= self.mean_rgb
        img /= self.std_rgb
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


    def mask_encode(self, mask):
        """
        Encode mask pixel distinct values to discrete class numbers
        :param mask:
        mask with ([0, 1, 2, .., num_classes]), original pixels values
        :return:
        new_mask with ([0, 1, 2, .., num_classes]), new pixels values
        """

        class_mapping = {
            4: 0, 10: 0, 14: 0, 15: 0,   # Soil, Snow, clear-cuts, herbaceous vegetation
            2: 1, 3: 1,                  # Pervious and transportation surfaces and sports fields
            1: 2, 13: 2, 18: 2,          # Buildings, swimming pools, Green houses
            6: 3, 7: 3, 16: 3, 17: 3,    # Trees
            8: 4,                        # Bushes
            9: 5, 11: 5, 12: 5,          # Agricultural surfaces
            5: 6,                        # Water bodies
        }
        # Initialize a new mask with the same shape
        new_mask = np.zeros_like(mask)

        # Reassign classes according to the mapping
        for old_class, new_class in class_mapping.items():
            new_mask[mask == old_class] = new_class
        return new_mask

class OriginalSize(FLAIRSegBase):
    def __init__(self, root, split, transform=False, patch_size=None, test=False):
        super().__init__(root, split, transform, patch_size, test)

    def __getitem__(self, idx):
        img = tiff.imread(self.files[idx]['img'])
        img = img[..., :3]
        mask = Image.open(self.files[idx]['msk'])
        mask = np.asarray(mask)
        mask = self.mask_encode(mask)

        if self._transform:
            img, mask = self.transform(img, mask)
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()

        return img, mask