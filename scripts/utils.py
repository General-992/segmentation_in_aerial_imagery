from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    """
    _, h, w = img.size()
    top = randint(0, h - patch_size)
    left = randint(0, w - patch_size)

    image_patch = img[:, top:top + patch_size, left:left + patch_size]
    mask_patch = mask[top:top + patch_size, left:left + patch_size]
    return image_patch, mask_patch

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - Overall accuracy
      - Mean accuracy
      - Mean IU
      - Frequency-Weighted Average Accuracy (fwavacc)
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


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