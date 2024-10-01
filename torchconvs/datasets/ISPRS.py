import os.path
import os.path as osp

import numpy as np
from PIL import Image
import torch

from torch.utils import data
import tifffile as tiff
import albumentations as A
import scripts

class ISPRSBase(data.Dataset):
    """
    Base ISPRS dataset class

    :param root: Dataset root directory.
    :param transform: Whether to apply data augmentation transformations.
    :param patch_size: Patch size for cropping.
    :param test: Whether this is a test dataset.
    """
    class_names = np.array([
        'Soil, Snow, clear - cuts, herbaceous vegetation',
        'Pervious and transportation surfaces and sports fields',
        'Buildings, swimming pools, Green houses',
        'Trees',
        'Agricultural surfaces',
        'Water bodies'
    ])
    ISPRS_labels = ['Impervious surfaces', 'Buildings',
                    'Low vegetation', 'Tree', 'Car']
    isprs_to_flair_mapping = {
        (255, 255, 255): 1, (0, 0, 255): 2,
        (0, 255, 255): 0, (0, 255, 0): 3,
        (255, 255, 0): 1
    }


    mean_rgb = np.array([113.775, 118.081, 109.273], dtype=np.float32)
    std_rgb = np.array([52.419, 46.028, 45.260], dtype=np.float32)
    transforms = A.Compose([A.HorizontalFlip(), A.VerticalFlip(),
                            A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                            A.GaussNoise()])
    def __init__(self, root: str, transform: bool=False, patch_size = None, test: bool=False):
        self.root = root
        self._transform = transform
        self.patch_size = patch_size
        self.test = test
        self.files = []
        imgsets_file = osp.join(self.root, 'test.txt')
        for did in open(imgsets_file):
            img_file, lbl_file = did.strip().split(' ')
            img_file = osp.join(self.root, 'img/patches',img_file)
            lbl_file = osp.join(self.root, 'msk/patches',lbl_file)
            self.files.append({
                'img': img_file,
                'msk': lbl_file,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img = tiff.imread(self.files[idx]['img'])
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
            if not self.test:
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
    def mask_encode(self, isprs_mask):

        """
        Encode mask pixel distinct values to discrete class numbers
        :param mask:
        mask with ([0, 1, 2, .., num_classes]), original pixels values
        :return:
        new_mask with ([0, 1, 2, .., num_classes]), new pixels values
        """
        flair_mask = np.zeros((isprs_mask.shape[0], isprs_mask.shape[1]), dtype=np.uint8)

        for rgb, class_label in self.isprs_to_flair_mapping.items():
            mask = np.all(isprs_mask == rgb, axis=-1)
            flair_mask[mask] = class_label
        return flair_mask



if __name__ == '__main__':
    root = os.path.expanduser('~/datasets/ISPRS/Potsdam')
    isprs = ISPRSBase(root=root)
    isprs[0]