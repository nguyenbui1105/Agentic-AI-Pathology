"""
Test feature extraction on three synthetic masks that represent
common pathology segmentation failure modes.

Run with:
    python tests/test_features.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.features.extractor import extract_features


# ── Synthetic masks ──────────────────────────────────────────────────────────

def make_clean_mask():
    """Single clean circular gland — ideal output."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    cy, cx = np.ogrid[:256, :256]
    mask[(cy - 128) ** 2 + (cx - 128) ** 2 <= 60 ** 2] = 255
    return mask


def make_noisy_mask():
    """One real gland + many small debris objects."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[80:160, 80:160] = 255
    rng = np.random.default_rng(42)
    for _ in range(25):
        r, c = rng.integers(5, 245), rng.integers(5, 245)
        s = rng.integers(2, 8)
        mask[r:r+s, c:c+s] = 255
    return mask


def make_holey_mask():
    """One gland with internal holes (unfilled lumen)."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:200, 50:200] = 255
    mask[90:115, 90:115] = 0   # hole 1
    mask[130:155, 130:155] = 0  # hole 2
    return mask


def make_merged_mask():
    """Two glands merged into one blob."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    cy, cx = np.ogrid[:256, :256]
    mask[(cy - 128) ** 2 + (cx - 80) ** 2 <= 50 ** 2] = 255
    mask[(cy - 128) ** 2 + (cx - 175) ** 2 <= 50 ** 2] = 255
    mask[115:141, 80:175] = 255  # bridge
    return mask


# ── Tests ────────────────────────────────────────────────────────────────────

def test_empty_mask():
    mask = np.zeros((256, 256), dtype=np.uint8)
    f = extract_features(mask)
    assert f["num_objects"] == 0
    assert f["total_mask_area"] == 0.0
    print("[empty_mask]   PASS")


def test_clean_mask():
    f = extract_features(make_clean_mask())
    assert f["num_objects"] == 1
    assert f["holes_count"] == 0
    assert f["small_object_ratio"] == 0.0
    assert 0.8 <= f["compactness"] <= 1.0,  f"Expected compactness ~1, got {f['compactness']}"
    assert f["boundary_roughness"] >= 1.0,  f"Expected roughness >= 1, got {f['boundary_roughness']}"
    print(f"[clean_mask]   PASS  compactness={f['compactness']}  roughness={f['boundary_roughness']}")


def test_noisy_mask():
    f = extract_features(make_noisy_mask())
    assert f["num_objects"] > 1,            f"Expected >1 objects, got {f['num_objects']}"
    assert f["small_object_ratio"] > 0.5,   f"Expected high small_object_ratio, got {f['small_object_ratio']}"
    print(f"[noisy_mask]   PASS  num_objects={f['num_objects']}  small_ratio={f['small_object_ratio']}")


def test_holey_mask():
    f = extract_features(make_holey_mask())
    assert f["holes_count"] >= 2,           f"Expected >=2 holes, got {f['holes_count']}"
    assert f["avg_hole_size"] > 0
    print(f"[holey_mask]   PASS  holes={f['holes_count']}  avg_hole_size={f['avg_hole_size']}")


def test_merged_mask():
    f = extract_features(make_merged_mask())
    assert f["num_objects"] == 1,           f"Expected 1 merged object, got {f['num_objects']}"
    assert f["avg_object_size"] > 5000,     f"Expected large object, got {f['avg_object_size']}"
    assert f["compactness"] < 0.8,          f"Expected low compactness for merged blob, got {f['compactness']}"
    print(f"[merged_mask]  PASS  num_objects={f['num_objects']}  size={f['avg_object_size']}  compactness={f['compactness']}")


# ── Pretty print ─────────────────────────────────────────────────────────────

def print_all():
    masks = {
        "clean":  make_clean_mask(),
        "noisy":  make_noisy_mask(),
        "holey":  make_holey_mask(),
        "merged": make_merged_mask(),
    }
    col_w = 22
    header = f"{'Feature':<{col_w}}" + "".join(f"{k:>12}" for k in masks)
    print("\n" + "=" * (col_w + 12 * len(masks)))
    print(header)
    print("=" * (col_w + 12 * len(masks)))

    features = {name: extract_features(m) for name, m in masks.items()}
    for key in features["clean"]:
        row = f"{key:<{col_w}}"
        for name in masks:
            row += f"{str(features[name][key]):>12}"
        print(row)
    print("=" * (col_w + 12 * len(masks)))


if __name__ == "__main__":
    print("Running feature extraction tests...\n")
    test_empty_mask()
    test_clean_mask()
    test_noisy_mask()
    test_holey_mask()
    test_merged_mask()
    print("\nAll tests passed.")
    print_all()
