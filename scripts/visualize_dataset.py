import argparse

import torchconvs
import os.path as osp
from utils import plot_image_label_classes
import random


def main():
    random.seed(19)

    root = osp.expanduser('~/datasets/ISPRS/Potsdam')
    dataset = torchconvs.datasets.ISPRSBase(root, transform=False)
    random_image = random.randint(0, len(dataset))
    image, label = dataset[random_image]
    image, label = dataset.untransform(image, label)
    plot_image_label_classes(image, label, dataset.class_names)


if __name__ == '__main__':
    main()