from random import randint
import numpy as np
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