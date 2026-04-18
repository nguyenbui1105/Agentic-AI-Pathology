"""
Test and visualize all 7 post-processing tools on synthetic masks.

Run with:
    python tests/test_tools.py

Saves: outputs/test_tools.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

from src.tools.postprocessing import (
    remove_small_objects,
    morph_close,
    morph_open,
    erosion,
    dilation,
    fill_holes,
    watershed_split,
    connect_fragments,
)


# ── Synthetic mask builders ──────────────────────────────────────────────────

def make_noisy_mask(h=256, w=256):
    """Large gland + scattered small debris blobs."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[80:160, 80:160] = 255
    rng = np.random.default_rng(42)
    for _ in range(30):
        r, c = rng.integers(5, h-5), rng.integers(5, w-5)
        mask[r:r+rng.integers(2, 9), c:c+rng.integers(2, 9)] = 255
    return mask


def make_holey_mask(h=256, w=256):
    """Gland with internal holes (unfilled lumen)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[60:196, 60:196] = 255
    mask[90:110, 90:110] = 0
    mask[130:150, 130:150] = 0
    mask[90:105, 140:155] = 0
    return mask


def make_merged_mask(h=256, w=256):
    """Two circular glands connected by a thin bridge."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = np.ogrid[:h, :w]
    mask[(cy-128)**2 + (cx-85)**2  <= 38**2] = 255
    mask[(cy-128)**2 + (cx-170)**2 <= 38**2] = 255
    mask[125:131, 85:170] = 255
    return mask


def make_spiky_mask(h=256, w=256):
    """Round gland with thin spiky protrusions on the boundary."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = np.ogrid[:h, :w]
    mask[(cy-128)**2 + (cx-128)**2 <= 55**2] = 255
    # Add thin spikes
    mask[50:128, 125:131] = 255   # top spike
    mask[128:140, 183:187] = 255  # right spike
    mask[175:200, 120:126] = 255  # bottom spike
    return mask


def make_shrunk_mask(h=256, w=256):
    """Two small glands that are slightly under-segmented (too tight)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = np.ogrid[:h, :w]
    mask[(cy-90)**2  + (cx-90)**2  <= 28**2] = 255
    mask[(cy-160)**2 + (cx-160)**2 <= 28**2] = 255
    return mask


def make_fragmented_mask(h=256, w=256):
    """
    Large main gland (~14 300px) with two small satellite fragments.
    Fragments are < 500px so connect_fragments classifies them correctly.
    Gaps: ~12px right, ~10px below  — both within max_gap_px=25.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    # Main gland body (~14 300px)
    mask[60:170, 50:180] = 255
    # Fragment 1: 18×18 = 324px, 12px gap to the right
    mask[85:103, 192:210] = 255
    # Fragment 2: 16×20 = 320px, 10px gap below
    mask[180:196, 78:98]  = 255
    return mask


def make_lumen_mask(h=256, w=256):
    """Gland ring with a large lumen hole in the center."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = np.ogrid[:h, :w]
    mask[(cy-128)**2 + (cx-128)**2 <= 70**2] = 255
    mask[(cy-128)**2 + (cx-128)**2 <= 35**2] = 0   # lumen
    return mask


# ── Test cases: (name, mask_fn, tool_fn, params, assertion_fn) ───────────────

def run_tests():
    cases = [
        {
            "name":   "remove_small_objects\n(min_size=100)",
            "mask":   make_noisy_mask(),
            "fn":     lambda m: remove_small_objects(m, min_size=100),
            "check":  lambda b, a: np.count_nonzero(a) < np.count_nonzero(b),
            "metric": lambda b, a: f"{np.count_nonzero(b)}px -> {np.count_nonzero(a)}px",
        },
        {
            "name":   "morph_close\n(kernel=11, iter=2)",
            "mask":   make_holey_mask(),
            "fn":     lambda m: morph_close(m, kernel_size=11, iterations=2),
            "check":  lambda b, a: np.count_nonzero(a) >= np.count_nonzero(b),
            "metric": lambda b, a: f"{np.count_nonzero(b)}px -> {np.count_nonzero(a)}px",
        },
        {
            "name":   "morph_open\n(kernel=7, iter=1)",
            "mask":   make_spiky_mask(),
            "fn":     lambda m: morph_open(m, kernel_size=7, iterations=1),
            "check":  lambda b, a: np.count_nonzero(a) <= np.count_nonzero(b),
            "metric": lambda b, a: f"{np.count_nonzero(b)}px -> {np.count_nonzero(a)}px",
        },
        {
            "name":   "erosion\n(kernel=7, iter=1)",
            "mask":   make_merged_mask(),
            "fn":     lambda m: erosion(m, kernel_size=7, iterations=1),
            "check":  lambda b, a: np.count_nonzero(a) < np.count_nonzero(b),
            "metric": lambda b, a: f"{np.count_nonzero(b)}px -> {np.count_nonzero(a)}px",
        },
        {
            "name":   "dilation\n(kernel=7, iter=1)",
            "mask":   make_shrunk_mask(),
            "fn":     lambda m: dilation(m, kernel_size=7, iterations=1),
            "check":  lambda b, a: np.count_nonzero(a) > np.count_nonzero(b),
            "metric": lambda b, a: f"{np.count_nonzero(b)}px -> {np.count_nonzero(a)}px",
        },
        {
            # Positive case: gland with small internal staining voids (< 2000px each).
            # fill_holes should fill them; the outer boundary must not change.
            # The large lumen ring (make_lumen_mask) is deliberately excluded here —
            # the agent's score_tools() already suppresses fill_holes for large lumen
            # via the lumen_ratio gate (hole > 20% of object area).
            "name":   "fill_holes\n(max_hole=2000)",
            "mask":   make_holey_mask(),
            "fn":     lambda m: fill_holes(m, max_hole_size=2000),
            "check":  lambda b, a: np.count_nonzero(a) > np.count_nonzero(b),
            "metric": lambda b, a: f"{np.count_nonzero(b)}px -> {np.count_nonzero(a)}px",
        },
        {
            # Negative (lumen-safety) case: ring gland with a large central lumen
            # (~3848px). max_hole_size=2000 keeps lumen intact (3848 > 2000).
            # The agent's score_tools() adds a second layer: hole_ratio > 20% of
            # object area suppresses fill_holes before it is even called.
            "name":   "fill_holes\n(lumen_safe)",
            "mask":   make_lumen_mask(),
            "fn":     lambda m: fill_holes(m, max_hole_size=2000),
            "check":  lambda b, a: np.count_nonzero(a) == np.count_nonzero(b),
            "metric": lambda b, a: (
                f"{np.count_nonzero(b)}px -> {np.count_nonzero(a)}px "
                f"({'lumen preserved' if np.count_nonzero(a) == np.count_nonzero(b) else 'LUMEN FILLED - BUG'})"
            ),
        },
        {
            # Positive case: main gland + two nearby fragments (gap < 25px).
            # connect_fragments should bridge both fragments to the main body,
            # increasing area and reducing object count from 3 to 1.
            "name":   "connect_fragments\n(gap=25, ratio=0.10)",
            "mask":   make_fragmented_mask(),
            "fn":     lambda m: connect_fragments(m, max_gap_px=25, size_ratio_max=0.10, min_main_size=500),
            "check":  lambda b, a: (
                measure.label(a > 0).max() < measure.label(b > 0).max()
                and np.count_nonzero(a) > np.count_nonzero(b)
            ),
            "metric": lambda b, a: (
                f"{measure.label(b>0).max()} obj -> {measure.label(a>0).max()} obj  "
                f"{np.count_nonzero(b)}px -> {np.count_nonzero(a)}px"
            ),
        },
        {
            "name":   "watershed_split\n(min_dist=30)",
            "mask":   make_merged_mask(),
            "fn":     lambda m: watershed_split(m, min_distance=30),
            "check":  lambda b, a: measure.label(a>0).max() >= measure.label(b>0).max(),
            "metric": lambda b, a: (
                f"{measure.label(b>0).max()} obj -> {measure.label(a>0).max()} obj"
            ),
        },
    ]

    print("Running all 8 tool tests...\n")
    results = []
    all_pass = True
    for c in cases:
        before = c["mask"]
        after  = c["fn"](before)
        ok     = c["check"](before, after)
        label  = c["name"].replace("\n", " ")
        met    = c["metric"](before, after)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}]  {label:<35}  {met}")
        results.append((c["name"], before, after, ok))

    print()
    print("All tests passed." if all_pass else "SOME TESTS FAILED.")
    return results


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(results, out_path):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.2 * n))
    fig.suptitle("Post-processing Tools (8) — Before / After", fontsize=13, fontweight="bold")

    for i, (name, before, after, ok) in enumerate(results):
        for j, (img, lbl) in enumerate([(before, "Before"), (after, "After")]):
            ax = axes[i][j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{name}\n{lbl}", fontsize=8)
            ax.axis("off")
            color = "#2ecc71" if ok else "#e74c3c"
            ax.text(0.02, 0.02, f"{np.count_nonzero(img)}px",
                    transform=ax.transAxes, color=color,
                    fontsize=7, va="bottom")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    results = run_tests()
    visualize(
        results,
        out_path=os.path.join(os.path.dirname(__file__), "..", "outputs", "visualizations", "test_tools.png"),
    )
