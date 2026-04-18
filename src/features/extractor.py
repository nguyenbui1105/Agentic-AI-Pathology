"""
Feature extraction from binary segmentation masks.

All features are scalar, numeric, and interpretable by a downstream agent.
Input mask: uint8 (H, W), foreground = 255 or 1, background = 0.
"""

import math
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.measure import label, regionprops


# Threshold below which an object is considered "small" (noise/debris).
# At 20x magnification a real gland is typically > 500 px.
_SMALL_OBJECT_PX = 500

# Max pixel gap for a small object to be called a "nearby fragment" of a
# large component (vs. isolated noise).
_FRAGMENT_PROXIMITY_PX = 50


def extract_features(mask: np.ndarray) -> dict:
    """
    Extract 10 quantitative features from a binary segmentation mask.

    Features guide an agent to decide which post-processing tool to apply:
    - Too many objects / high small_object_ratio  → remove_small_objects
    - High holes_count / avg_hole_size            → morph_close / fill_holes
    - Low compactness / high boundary_roughness   → morph_close / erosion
    - Few large merged objects                    → watershed_split

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), dtype uint8. Foreground pixels are 255 or 1.

    Returns
    -------
    dict
        Feature dictionary with float values. Keys:
        num_objects, avg_object_size, object_size_std, small_object_ratio,
        boundary_roughness, holes_count, avg_hole_size,
        total_mask_area, mask_density, compactness
    """
    binary = mask > 0
    h, w = binary.shape
    total_pixels = h * w

    # ── Object-level features ────────────────────────────────────────────────
    labeled = label(binary)
    props = regionprops(labeled)

    num_objects = len(props)

    if num_objects == 0:
        return _empty_features()

    areas = np.array([p.area for p in props], dtype=float)

    avg_object_size = float(areas.mean())
    max_object_size = float(areas.max())
    object_size_std = float(areas.std())
    # Coefficient of variation: std / mean.
    # Low (~0) = all objects similar size (likely all real glands).
    # High (>1) = wildly unequal sizes (large glands mixed with tiny debris).
    size_cv = float(object_size_std / avg_object_size) if avg_object_size > 0 else 0.0
    small_object_ratio = float((areas < _SMALL_OBJECT_PX).sum() / num_objects)
    total_mask_area = float(areas.sum())
    mask_density = total_mask_area / total_pixels

    # ── Shape features (per object, then averaged) ───────────────────────────
    # boundary_roughness: perimeter / (2 * sqrt(pi * area))
    #   Perfect circle = 1.0; jagged/elongated shapes > 1.0
    roughness_vals = []
    compactness_vals = []
    for p in props:
        perim = p.perimeter
        area = p.area
        if perim > 0 and area > 0:
            roughness_vals.append(perim / (2.0 * math.sqrt(math.pi * area)))
            compactness_vals.append((4.0 * math.pi * area) / (perim ** 2))

    boundary_roughness = float(np.mean(roughness_vals)) if roughness_vals else 0.0
    compactness = float(np.mean(compactness_vals)) if compactness_vals else 0.0

    # ── Hole features ────────────────────────────────────────────────────────
    filled = binary_fill_holes(binary)
    hole_mask = filled & ~binary

    hole_labeled = label(hole_mask)
    hole_props = regionprops(hole_labeled)

    holes_count = len(hole_props)
    hole_areas = np.array([p.area for p in hole_props], dtype=float)
    avg_hole_size = float(hole_areas.mean()) if holes_count > 0 else 0.0

    # ── Fragment proximity feature ────────────────────────────────────────────
    # Fraction of small objects whose nearest large component is within
    # _FRAGMENT_PROXIMITY_PX pixels.  High value → fragmented gland (use
    # connect_fragments); low value → scattered noise (use remove_small_objects).
    small_props = [p for p in props if p.area < _SMALL_OBJECT_PX]
    large_props = [p for p in props if p.area >= _SMALL_OBJECT_PX]

    if small_props and large_props:
        large_mask = np.zeros_like(binary)
        for lp in large_props:
            large_mask[labeled == lp.label] = True
        dist_from_large = distance_transform_edt(~large_mask)
        nearby = sum(
            1 for sp in small_props
            if dist_from_large[labeled == sp.label].min() <= _FRAGMENT_PROXIMITY_PX
        )
        nearby_fragment_ratio = round(nearby / len(small_props), 4)
    else:
        nearby_fragment_ratio = 0.0

    return {
        "num_objects":           num_objects,
        "avg_object_size":       round(avg_object_size, 2),
        "max_object_size":       round(max_object_size, 2),
        "object_size_std":       round(object_size_std, 2),
        "size_cv":               round(size_cv, 4),
        "small_object_ratio":    round(small_object_ratio, 4),
        "boundary_roughness":    round(boundary_roughness, 4),
        "holes_count":           holes_count,
        "avg_hole_size":         round(avg_hole_size, 2),
        "total_mask_area":       round(total_mask_area, 2),
        "mask_density":          round(mask_density, 4),
        "compactness":           round(compactness, 4),
        "nearby_fragment_ratio": nearby_fragment_ratio,
    }


def _empty_features() -> dict:
    """Return a zero-filled feature dict when the mask is empty."""
    return {
        "num_objects":           0,
        "avg_object_size":       0.0,
        "max_object_size":       0.0,
        "object_size_std":       0.0,
        "size_cv":               0.0,
        "small_object_ratio":    0.0,
        "boundary_roughness":    0.0,
        "holes_count":           0,
        "avg_hole_size":         0.0,
        "total_mask_area":       0.0,
        "mask_density":          0.0,
        "compactness":           0.0,
        "nearby_fragment_ratio": 0.0,
    }
