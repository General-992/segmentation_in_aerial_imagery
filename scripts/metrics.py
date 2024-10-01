import numpy as np
import cv2


def _fast_hist(label_true, label_pred, n_class):
    """
    computes the confusion matrix between the true and
    predicted labels for a multi-class classification task
    """
    # identify only valid pixels
    mask = (label_true >= 0) & (label_true < n_class)
    # compute combined index for each valid pixel and count each True-Predcited Pair occurence
    # then reshape into a confusion matrix shape n_classes * n_classes
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

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert a binary mask to a boundary mask.
    :param mask (numpy array): Binary mask of shape (H, W)
    :param dilation_ratio (float): Ratio to calculate dilation based on image diagonal.
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))

    if dilation < 1:
        dilation = 1

    # Pad image to ensure boundary pixels near the image border are handled correctly
    padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Erode to find boundaries
    eroded_mask = cv2.erode(padded_mask, kernel, iterations=dilation)
    boundary_mask = padded_mask[1:h + 1, 1:w + 1] - eroded_mask[1:h + 1, 1:w + 1]

    return boundary_mask


def boundary_iou_per_class(gt_mask, pred_mask, class_id, dilation_ratio=0.02):
    """
    Compute Boundary IoU for a specific class between two multi-class masks (One vs. All).
    :param gt_mask (numpy array): Ground truth multi-class mask of shape (H, W)
    :param pred_mask (numpy array): Predicted multi-class mask of shape (H, W)
    :param class_id (int): Class ID to compute Boundary IoU for.
    :param dilation_ratio (float): Ratio for dilation to calculate the boundary.
    :return: Boundary IoU score (float)
    """

    gt_class_mask = (gt_mask == class_id).astype(np.uint8)
    pred_class_mask = (pred_mask == class_id).astype(np.uint8)

    gt_boundary = mask_to_boundary(gt_class_mask, dilation_ratio)
    pred_boundary = mask_to_boundary(pred_class_mask, dilation_ratio)

    intersection = np.sum((gt_boundary * pred_boundary) > 0)
    union = np.sum((gt_boundary + pred_boundary) > 0)

    if union == 0:
        return 0.0

    boundary_iou_score = intersection / union
    return boundary_iou_score


def boundary_iou_multiclass(gt_mask, pred_mask, num_classes, dilation_ratio=0.02):
    """
    Compute the average Boundary IoU for all classes in multi-class segmentation.
    :param gt_mask (numpy array): Ground truth multi-class mask of shape (H, W)
    :param pred_mask (numpy array): Predicted multi-class mask of shape (H, W)
    :param num_classes (int): Number of classes in the segmentation mask.
    :param dilation_ratio (float): Ratio for dilation to calculate the boundary.
    :return: Average Boundary IoU score (float)
    """
    boundary_iou_scores = []

    for class_id in range(num_classes):
        iou_score = boundary_iou_per_class(gt_mask, pred_mask, class_id, dilation_ratio)
        if iou_score != 0:
            boundary_iou_scores.append(iou_score)
    if np.mean(boundary_iou_scores) == 1.0:
        return np.nan  # Return NaN if the image is single class
    else:
        return np.mean(boundary_iou_scores)


