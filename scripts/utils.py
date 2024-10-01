from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torchconvs
def patch_divide(img, mask, patch_size):
    """
    If the image is larger than 256x256 and it is a testing stage
    it divides the image into patches of 256x256 pixels.
    input shape: ([3, l, w])
    output shape: ([3, num_patches^2, l/num_patches, w/num_patches])
    """
    if img.shape[-1] == img.shape[-2]:
        length = img.shape[-1] / patch_size
        number_of_patches = int((length) ** 2)
        img = img.view(number_of_patches, 3, int(img.shape[-1] / length), int(img.shape[-1] / length))
        mask = mask.view(number_of_patches, int(mask.shape[-1] / length), int(mask.shape[-1] / length))
    else:
        raise NotImplementedError(
            'Not implemented the division of the test images of non-rectangular shape')
    return img, mask

def patch_sample(img, mask, patch_size):
    """
    Randomly samples patches from images, useful
    when dealing with large resolution map images.
    Each patch has a fixed size of patch_size x patch_size pixels

    Input:
    :img: ndarray[num_channels, height, width]
    :mask: ndarray[height, width]
    :patch_size: int

    Output:
    :image_patch: ndarray[num_channels, patch_size, patch_size]
    :mask_patch: ndarray[patch_size, patch_size]
    """
    _, h, w = img.shape
    top = randint(0, h - patch_size)
    left = randint(0, w - patch_size)

    image_patch = img[:, top:top + patch_size, left:left + patch_size]
    mask_patch = mask[top:top + patch_size, left:left + patch_size]
    return image_patch, mask_patch


def plot_image_label_classes(img, mask, class_names):
    """
    Plots the image, the mask, and the semantic classes on the same level.

    :param img: numpy array of shape [512, 512, 3], the image.
    :param mask: numpy array of shape [512, 512], the label mask.
    :param class_names: list or array of class names.
    """

    # Define the colormap for the mask (label) to make it more visually distinct
    cmap = plt.get_cmap('tab10')
    unique_classes = np.unique(mask)
    colors = cmap(np.linspace(0, 1, len(class_names)))

    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the image
    axes[0].imshow(img.astype(np.uint8))
    axes[0].axis('off')
    axes[0].set_title("Image")

    # Plot the mask (label)
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, cls in enumerate(unique_classes):
        colored_mask[mask == cls] = (colors[cls][:3] * 255).astype(np.uint8)

    axes[1].imshow(colored_mask)
    axes[1].axis('off')
    axes[1].set_title("Label")

    # Plot the semantic classes
    legend_patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(len(class_names))]
    axes[2].legend(handles=legend_patches, loc='center', fontsize='small')
    axes[2].axis('off')
    axes[2].set_title("Semantic Classes")

    # Adjust layout to avoid overlapping
    plt.tight_layout()
    plt.show()


def plot_metrics_per_month(month_metrics, model_name):
    """
    Plots bar charts showing the segmentation metrics (accuracy, class accuracy, mean IU, FWAV accuracy)
    for each month.

    Parameters:
    month_metrics (dict): A dictionary where keys are months ('1', '2', ..., '12')
                          and values are lists of metric tuples (accuracy, accuracy_class, mean_iu, fwavacc).
    """
    months = []
    accuracies = []
    accuracy_classes = []
    mean_ius = []
    fwav_accs = []

    for month, metrics in month_metrics.items():
        if metrics:  # Ensure the month has metrics
            avg_metrics = np.mean(metrics, axis=0)  # Calculate mean metrics for the month
            accuracy, acc_cls, mean_iu, fwavacc = avg_metrics * 100  # Convert to percentages
            months.append(int(month))  # Store the month (convert string month to integer)
            accuracies.append(accuracy)  # Store accuracy
            accuracy_classes.append(acc_cls)  # Store class accuracy
            mean_ius.append(mean_iu)  # Store mean IU
            fwav_accs.append(fwavacc)  # Store FWAV accuracy

    months, accuracies, accuracy_classes, mean_ius, fwav_accs = zip(
        *sorted(zip(months, accuracies, accuracy_classes, mean_ius, fwav_accs)))

    metrics_names = ['Accuracy (%)', 'Class Accuracy (%)', 'Mean IU (%)', 'FWAV Accuracy (%)']
    metrics_data = [accuracies, accuracy_classes, mean_ius, fwav_accs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        ax.bar(months, metrics_data[idx], color='skyblue')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel(metrics_names[idx], fontsize=12)
        ax.set_title(f'{metrics_names[idx]} per Month', fontsize=14)
        ax.set_xticks(months)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle(f'Metrics per Month for {model_name}', fontsize=16)
    plt.tight_layout()
    plt.show()


def model_select(model_name: str, n_class: int = 6) -> torch.nn.Module:
    """
    Sets up the model
    """
    if model_name.lower().startswith('unet'):
        # num of trainable params = 26.079.479
        print('Start training Unet')
        model = torchconvs.models.UnetPlusPlus(n_class=n_class)
    elif model_name.lower().startswith('deep'):
        #  num of trainable params = 39.758.247
        print('Start training Deeplab')
        model = torchconvs.models.Deeplabv3plus_resnet(n_class=n_class)
    elif model_name.lower().startswith('segnet'):
        print('Start training Segnet')
        # num of trainable params = 12.932.295
        model = torchconvs.models.SegNet(n_class=n_class)
    elif model_name.lower().startswith('hrnet'):
        # num of trainable params = 31.990.087
        print('Start training HRNet')
        model = torchconvs.models.HRNet(n_class=n_class)
    else:
        raise Exception('Unknown model')
    sum = 0
    for param in model.parameters():
        if param.requires_grad:
            sum += param.numel()
    print(f'Total trainable params: {sum}, model: {model_name}')
    return model

import imgviz
import copy
def visualize_segmentation(**kwargs):
    """Visualize segmentation.

    Parameters
    ----------
    img: ndarray
        Input image to predict label.
    lbl_true: ndarray
        Ground truth of the label.
    lbl_pred: ndarray
        Label predicted.
    n_class: int
        Number of classes.
    label_names: dict or list
        Names of each label value.
        Key or index is label_value and value is its name.

    Returns
    -------
    img_array: ndarray
        Visualized image.
    """
    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred = kwargs.pop('lbl_pred', None)
    n_class = kwargs.pop('n_class', None)
    label_names = kwargs.pop('label_names', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    if lbl_true is None and lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')

    lbl_true = copy.deepcopy(lbl_true)
    lbl_pred = copy.deepcopy(lbl_pred)

    mask_unlabeled = None
    viz_unlabeled = None
    if lbl_true is not None:
        mask_unlabeled = lbl_true == -1
        lbl_true[mask_unlabeled] = 0
        viz_unlabeled = (
            np.random.random((lbl_true.shape[0], lbl_true.shape[1], 3)) * 255
        ).astype(np.uint8)
        if lbl_pred is not None:
            lbl_pred[mask_unlabeled] = 0

    vizs = []

    if lbl_true is not None:
        viz_trues = [
            img,
            imgviz.label2rgb(label=lbl_true, label_names=label_names),
            imgviz.label2rgb(lbl_true, img, label_names=label_names),
        ]
        viz_trues[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        viz_trues[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(imgviz.tile(viz_trues, (1, 3)))

    if lbl_pred is not None:
        viz_preds = [
            img,
            imgviz.label2rgb(lbl_pred, label_names=label_names),
            imgviz.label2rgb(lbl_pred, img, label_names=label_names),
        ]
        if mask_unlabeled is not None and viz_unlabeled is not None:
            viz_preds[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
            viz_preds[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(imgviz.tile(viz_preds, (1, 3)))

    if len(vizs) == 1:
        return vizs[0]
    elif len(vizs) == 2:
        return imgviz.tile(vizs, (2, 1))
    else:
        raise RuntimeError