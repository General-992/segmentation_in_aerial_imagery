import torchconvs
import os.path as osp
from utils import plot_image_label_classes
import random


def main():
    random.seed(19)

    root = osp.expanduser('~/datasets/flair_dataset')
    dataset = torchconvs.datasets.FLAIRSegBase(root, split='train', transform=False)
    random_image = random.randint(0, len(dataset))
    plot_image_label_classes(dataset[random_image][0], dataset[random_image][1], dataset.class_names)
if __name__ == '__main__':
    main()