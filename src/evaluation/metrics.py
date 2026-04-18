"""
Evaluation metrics for binary segmentation masks.
"""

import numpy as np
from skimage.measure import label


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice coefficient between two binary masks.
    Returns 1.0 if both masks are empty (perfect match on background).
    """
    pred_b = pred > 0
    gt_b   = gt > 0
    intersection = (pred_b & gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0
    return round(float(2 * intersection / denom), 4)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Intersection over Union between two binary masks.
    Returns 1.0 if both masks are empty.
    """
    pred_b = pred > 0
    gt_b   = gt > 0
    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    if union == 0:
        return 1.0
    return round(float(intersection / union), 4)


def basic_stats(mask: np.ndarray) -> dict:
    """
    Compute mask statistics that don't require a ground truth.
    Always available regardless of whether GT exists.
    """
    binary = mask > 0
    labeled = label(binary)
    num_objects = int(labeled.max())
    total_area  = int(binary.sum())
    h, w = mask.shape
    density = round(total_area / (h * w), 4)
    return {
        "num_objects": num_objects,
        "total_area":  total_area,
        "density":     density,
    }
