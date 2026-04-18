"""
Post-processing tools for pathology gland segmentation masks.

Each tool accepts a binary mask (uint8, values 0/255 or 0/1) and returns
a binary mask of the same shape and dtype.
"""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage import morphology, measure, segmentation, feature
from skimage.measure import label as sk_label, regionprops as sk_rprops


def remove_small_objects(mask: np.ndarray, min_size: int = 500) -> np.ndarray:
    """
    Remove small spurious objects (noise/debris) from a segmentation mask.

    In pathology, small disconnected regions are typically staining artifacts,
    tissue debris, or model false positives rather than real glands.
    Removing them reduces over-segmentation noise without affecting true glands.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), dtype uint8. Foreground pixels are 255 or 1.
    min_size : int
        Minimum connected-component area in pixels. Components smaller than
        this threshold are removed. Default 500 (~25x25 px at 20x magnification).

    Returns
    -------
    np.ndarray
        Cleaned binary mask, same shape and dtype as input.

    Example
    -------
    >>> cleaned = remove_small_objects(mask, min_size=300)
    """
    binary = mask > 0
    # max_size removes objects with area <= max_size; use min_size-1 to match
    # the original semantics (remove objects strictly smaller than min_size).
    cleaned = morphology.remove_small_objects(binary, max_size=min_size - 1)
    return (cleaned.astype(np.uint8)) * (255 if mask.max() == 255 else 1)


def morph_close(
    mask: np.ndarray,
    kernel_size: int = 7,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply morphological closing to fill gaps and smooth gland boundaries.

    Gland lumens sometimes appear as small holes in the segmentation mask,
    and boundaries can be jagged due to model uncertainty. Closing (dilation
    followed by erosion) fills these internal holes and smooths rough edges
    without significantly changing gland size.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), dtype uint8. Foreground pixels are 255 or 1.
    kernel_size : int
        Side length of the square structuring element in pixels. Larger values
        fill bigger gaps. Default 7 (suitable for small lumen artifacts).
    iterations : int
        Number of times the closing operation is applied. Default 1.

    Returns
    -------
    np.ndarray
        Closed binary mask, same shape and dtype as input.

    Example
    -------
    >>> smoothed = morph_close(mask, kernel_size=9, iterations=2)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=iterations
    )
    return closed


def watershed_split(mask: np.ndarray, min_distance: int = 15) -> np.ndarray:
    """
    Separate merged/touching glands using the watershed algorithm.

    When two adjacent glands touch or overlap, the model may predict them as
    a single large object. Watershed uses the distance transform to find local
    maxima (gland centers) and then grows regions outward, placing boundaries
    between touching glands.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), dtype uint8. Foreground pixels are 255 or 1.
    min_distance : int
        Minimum number of pixels between detected gland centers (peaks in the
        distance transform). Smaller values split more aggressively; larger
        values only separate clearly distinct glands. Default 15.

    Returns
    -------
    np.ndarray
        Binary mask after splitting, same shape and dtype as input.
        Shape of individual glands may change but overall foreground area
        is preserved.

    Example
    -------
    >>> separated = watershed_split(mask, min_distance=20)
    """
    binary = (mask > 0).astype(np.uint8)

    # Distance transform: each pixel gets distance to nearest background
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Find local maxima in distance map → gland centers (seeds for watershed)
    local_max = feature.peak_local_max(
        dist,
        min_distance=min_distance,
        labels=binary,
        exclude_border=False,
    )

    # Build marker image from detected centers
    markers = np.zeros(dist.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(local_max, start=1):
        markers[r, c] = idx

    # If no peaks found, return the original mask unchanged
    if markers.max() == 0:
        return mask

    # Watershed on inverted distance map
    labels = segmentation.watershed(-dist, markers, mask=binary)

    # Build a boundary mask: pixels adjacent to a different label are boundaries.
    # Dilating each label and finding overlaps gives a clean, controllable gap.
    n_labels = int(labels.max())
    # Dilate each label slightly, then mark overlapping pixels as background.
    dilated = np.zeros((n_labels + 1, *binary.shape), dtype=np.uint8)
    expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for label_id in range(1, n_labels + 1):
        region = (labels == label_id).astype(np.uint8)
        dilated[label_id] = cv2.dilate(region, expand_kernel, iterations=1)

    # Pixels claimed by more than one label become background (the gap)
    overlap = (dilated[1:].sum(axis=0) > 1).astype(np.uint8)

    result = np.zeros_like(binary)
    for label_id in range(1, n_labels + 1):
        region = (labels == label_id).astype(np.uint8)
        # Remove overlap zone from each region
        result = np.maximum(result, region * (1 - overlap))

    return result * (255 if mask.max() == 255 else 1)


def morph_open(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply morphological opening to remove thin protrusions and separate
    weakly connected regions.

    In pathology, gland boundaries predicted by the model sometimes include
    thin, spurious bridges connecting adjacent glands, or sharp spikes along
    the gland edge. Opening (erosion followed by dilation) removes structures
    thinner than the kernel without significantly shrinking the gland body.
    It can also break narrow necks between touching glands when watershed
    is too aggressive.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), dtype uint8. Foreground pixels are 255 or 1.
    kernel_size : int
        Side length of the elliptical structuring element in pixels.
        Larger values remove thicker protrusions. Default 5.
    iterations : int
        Number of times the opening operation is applied. Default 1.

    Returns
    -------
    np.ndarray
        Opened binary mask, same shape and dtype as input.

    Example
    -------
    >>> cleaned = morph_open(mask, kernel_size=7, iterations=1)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    opened = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=iterations
    )
    return opened


def erosion(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Shrink gland boundaries inward by eroding the foreground mask.

    Useful when the segmentation model over-predicts gland boundaries,
    producing masks that bleed into the surrounding stroma. Erosion pulls
    the boundary inward by kernel_size/2 pixels. Applying it before
    watershed can also improve separation of touching glands by widening
    the background gap between them.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), dtype uint8. Foreground pixels are 255 or 1.
    kernel_size : int
        Side length of the elliptical structuring element. Larger values
        erode more aggressively. Default 3 (conservative, ~1 px per side).
    iterations : int
        Number of erosion passes. Default 1.

    Returns
    -------
    np.ndarray
        Eroded binary mask, same shape and dtype as input.

    Example
    -------
    >>> tightened = erosion(mask, kernel_size=5, iterations=1)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    return eroded


def dilation(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Expand gland boundaries outward by dilating the foreground mask.

    Useful when the segmentation model under-predicts gland boundaries,
    leaving a thin background ring around glands that should be foreground.
    Common in cases where the gland boundary staining is faint or the
    model was trained on a different staining protocol. Dilation grows
    each foreground region outward by kernel_size/2 pixels.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), dtype uint8. Foreground pixels are 255 or 1.
    kernel_size : int
        Side length of the elliptical structuring element. Default 3.
    iterations : int
        Number of dilation passes. Default 1.

    Returns
    -------
    np.ndarray
        Dilated binary mask, same shape and dtype as input.

    Example
    -------
    >>> expanded = dilation(mask, kernel_size=5, iterations=1)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated


def connect_fragments(
    mask: np.ndarray,
    max_gap_px: int = 25,
    size_ratio_max: float = 0.10,
    min_main_size: int = 500,
    bridge_width: int = 4,
) -> np.ndarray:
    """
    Reconnect small satellite fragments to their nearest large gland body.

    Fragmented glands occur when the model under-predicts boundary continuity,
    producing a large main body surrounded by small disconnected pieces that
    were part of the same gland.  This tool identifies such fragments, measures
    the pixel gap between fragment and main, and fills the gap with a thin
    bridge when the gap is small enough.

    Bridge geometry
    ---------------
    For fragment F and main component M, the bridge is defined as:

        dist_from_F(p) + dist_from_M(p) <= gap + bridge_width

    This is the set of pixels within a corridor connecting the two closest
    points on each component's boundary.  The corridor has an approximately
    elliptical cross-section of half-width ~ sqrt(gap * bridge_width / 2).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), uint8.  Foreground = 255 or 1.
    max_gap_px : int
        Maximum pixel distance between fragment and main to attempt bridging.
        Fragments farther away are left unchanged.  Default 25.
    size_ratio_max : float
        Maximum fragment_area / main_area ratio.  If a "fragment" is more than
        this fraction of the main gland's area, it may be an independent gland
        and is left unchanged.  Default 0.10 (fragment < 10 % of main).
    min_main_size : int
        Minimum area (px) for a component to be treated as a main gland.
        Components below this threshold are always considered fragments.
        Default 500.
    bridge_width : int
        Controls the width of the bridge corridor in pixels (added to the
        measured gap in the corridor condition above).  Default 4.

    Returns
    -------
    np.ndarray
        Mask with fragment bridges added, same shape and dtype as input.
        Background pixels only — existing foreground is never modified.

    Example
    -------
    >>> reconnected = connect_fragments(mask, max_gap_px=20, size_ratio_max=0.08)
    """
    binary = mask > 0
    val    = 255 if mask.max() == 255 else 1

    labeled = sk_label(binary)
    n_comp  = int(labeled.max())

    if n_comp < 2:
        return mask.copy()

    props = {p.label: p for p in sk_rprops(labeled)}

    mains     = {lbl: p for lbl, p in props.items() if p.area >= min_main_size}
    fragments = {lbl: p for lbl, p in props.items() if p.area < min_main_size}

    if not mains or not fragments:
        return mask.copy()

    # Distance from every pixel to the nearest main-gland pixel (fast: one DT).
    main_combined = np.zeros(binary.shape, dtype=bool)
    for lbl in mains:
        main_combined |= (labeled == lbl)
    dist_from_mains = distance_transform_edt(~main_combined)

    result = (binary.astype(np.uint8)) * val

    for frag_lbl, frag_prop in fragments.items():
        frag_bool = labeled == frag_lbl

        # Quick pre-filter: if nearest main pixel is too far away, skip.
        min_gap = float(dist_from_mains[frag_bool].min())
        if min_gap > max_gap_px:
            continue

        # Compute per-fragment distance transform (needed for bridge formula).
        dist_from_frag = distance_transform_edt(~frag_bool)

        # Find the specific main component that is closest.
        best_lbl  = None
        best_gap  = float("inf")
        for main_lbl in mains:
            main_bool = labeled == main_lbl
            gap = float(dist_from_frag[main_bool].min())
            if gap < best_gap:
                best_gap  = gap
                best_lbl  = main_lbl

        if best_lbl is None or best_gap > max_gap_px:
            continue

        # Enforce size-ratio safety: don't bridge similarly-sized components.
        if frag_prop.area / mains[best_lbl].area > size_ratio_max:
            continue

        # Build bridge: corridor connecting fragment and nearest main.
        dist_from_main = distance_transform_edt(~(labeled == best_lbl))
        bridge = (dist_from_frag + dist_from_main) <= (best_gap + bridge_width)

        # Only fill background pixels.
        result[bridge & ~binary] = val

    return result


def fill_holes(mask: np.ndarray, max_hole_size: int = 500) -> np.ndarray:
    """
    Fill internal holes in gland masks up to a maximum hole area.

    Gland lumens are sometimes predicted as background, creating holes
    inside the gland mask. Unlike morph_close (which also smooths the
    boundary), fill_holes targets only enclosed holes without changing
    the outer boundary shape at all. Useful when the gland boundary is
    already well-localised but lumen regions are incorrectly excluded.

    max_hole_size limits filling to small lumen artifacts; very large
    holes (e.g. the entire background) are left untouched.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), dtype uint8. Foreground pixels are 255 or 1.
    max_hole_size : int
        Maximum hole area in pixels to fill. Holes larger than this are
        kept as-is. Default 500. Set to a large value to fill all holes.

    Returns
    -------
    np.ndarray
        Mask with small internal holes filled, same shape and dtype as input.

    Example
    -------
    >>> filled = fill_holes(mask, max_hole_size=1000)
    """
    binary = mask > 0
    # remove_small_holes fills holes strictly smaller than area_threshold
    filled = morphology.remove_small_holes(binary, max_size=max_hole_size)
    return (filled.astype(np.uint8)) * (255 if mask.max() == 255 else 1)
