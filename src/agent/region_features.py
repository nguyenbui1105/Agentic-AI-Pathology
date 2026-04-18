"""
Per-region (connected component) feature extraction.

Returns one feature dict per significant connected component in a binary mask.
Each dict is compatible with PathologyReasoningAgent.decide() — same keys as
the global extractor — plus metadata keys prefixed with '_'.

Design constraints
------------------
- Regions below _SMALL_REGION_PX are excluded; they are debris handled globally.
- Boundary roughness formula matches the global extractor exactly:
      perimeter / (2 * sqrt(pi * area))   — circle = 1.0
- Compactness formula matches:
      4 * pi * area / perimeter²           — circle = 1.0
- Local mask_density is computed as area / bounding-box area (proxy for how
  sparse the region's boundary prediction is within its own footprint).
"""

from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops

# Objects below this threshold are considered debris; skip per-region analysis.
_SMALL_REGION_PX = 500


def extract_region_features(mask: np.ndarray) -> list[dict]:
    """
    Extract per-region feature dicts for all significant connected components.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), uint8.  Foreground = 255 or 1.

    Returns
    -------
    list[dict]
        One dict per significant region (area >= _SMALL_REGION_PX), sorted by
        area descending (largest / most important regions first).

        Standard feature keys (agent-compatible):
            num_objects, avg_object_size, max_object_size, object_size_std,
            size_cv, small_object_ratio, boundary_roughness, holes_count,
            avg_hole_size, total_mask_area, mask_density, compactness

        Metadata keys (prefixed '_', not used by agent scoring):
            _label     : int   — connected-component label in the labeled mask
            _bbox      : tuple — (min_row, min_col, max_row, max_col)
            _centroid  : tuple — (row, col) centroid of the region
            _area      : int   — region area in pixels
    """
    binary = mask > 0
    labeled = label(binary)
    props   = regionprops(labeled)

    if not props:
        return []

    results: list[dict] = []

    for prop in props:
        if prop.area < _SMALL_REGION_PX:
            continue

        area  = float(prop.area)
        perim = prop.perimeter

        if perim > 0 and area > 0:
            roughness   = perim / (2.0 * math.sqrt(math.pi * area))
            compactness = (4.0 * math.pi * area) / (perim ** 2)
        else:
            roughness   = 0.0
            compactness = 1.0

        # Internal holes: fill the isolated region, subtract original.
        region_bool = labeled == prop.label
        filled      = binary_fill_holes(region_bool)
        hole_mask   = filled & ~region_bool

        h_labeled = label(hole_mask)
        h_props   = regionprops(h_labeled)
        holes_count   = len(h_props)
        h_areas       = [hp.area for hp in h_props]
        avg_hole_size = float(np.mean(h_areas)) if h_areas else 0.0

        # Local density: area vs bounding-box area.
        # prop.area_bbox is the canonical name since skimage 0.26.
        bbox_area    = max(getattr(prop, "area_bbox", None) or getattr(prop, "bbox_area", None) or 1, 1)
        mask_density = area / bbox_area

        results.append({
            # ── Agent-compatible feature keys ─────────────────────────────
            "num_objects":           1,
            "avg_object_size":       round(area, 2),
            "max_object_size":       round(area, 2),
            "object_size_std":       0.0,
            "size_cv":               0.0,
            "small_object_ratio":    0.0,
            "boundary_roughness":    round(roughness, 4),
            "holes_count":           holes_count,
            "avg_hole_size":         round(avg_hole_size, 2),
            "total_mask_area":       round(area, 2),
            "mask_density":          round(mask_density, 4),
            "compactness":           round(compactness, 4),
            "nearby_fragment_ratio": 0.0,  # not meaningful for a single isolated region
            # ── Region metadata ───────────────────────────────────────────
            "_label":    prop.label,
            "_bbox":     prop.bbox,
            "_centroid": (round(prop.centroid[0], 1), round(prop.centroid[1], 1)),
            "_area":     int(area),
        })

    results.sort(key=lambda r: -r["_area"])
    return results
