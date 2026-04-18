"""
End-to-end demo of the agent-guided post-processing pipeline.

Four synthetic test cases — each with a noisy input mask and a clean GT:
  clean   → no_action      (GT = single circle)
  noisy   → remove_small_objects
  holey   → morph_close
  merged  → watershed_split

Run with:
    python tests/demo_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.pipeline import run_postprocessing_pipeline


# ── Synthetic masks ──────────────────────────────────────────────────────────

def circle(h, w, cy, cx, r):
    arr = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    arr[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 255
    return arr


def make_clean(h=256, w=256):
    mask = circle(h, w, 128, 128, 60)
    gt   = circle(h, w, 128, 128, 60)
    return mask, gt


def make_noisy(h=256, w=256):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[80:160, 80:160] = 255
    rng = np.random.default_rng(7)
    for _ in range(30):
        r, c = rng.integers(5, 245), rng.integers(5, 245)
        s = rng.integers(2, 9)
        mask[r:r+s, c:c+s] = 255
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[80:160, 80:160] = 255
    return mask, gt


def make_holey(h=256, w=256):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:200, 50:200] = 255
    mask[90:115, 90:115] = 0
    mask[130:155, 130:155] = 0
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[50:200, 50:200] = 255
    return mask, gt


def make_merged(h=256, w=256):
    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    mask[(yy - 128) ** 2 + (xx - 80) ** 2 <= 50 ** 2] = 255
    mask[(yy - 128) ** 2 + (xx - 175) ** 2 <= 50 ** 2] = 255
    mask[115:141, 80:175] = 255
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[(yy - 128) ** 2 + (xx - 80) ** 2 <= 50 ** 2] = 255
    gt[(yy - 128) ** 2 + (xx - 175) ** 2 <= 50 ** 2] = 255
    return mask, gt


# ── Reporting ────────────────────────────────────────────────────────────────

def print_report(name: str, r: dict) -> None:
    imp = r["improvement_summary"]
    dec = r["decision"]
    print(f"\n{'='*60}")
    print(f"  CASE: {name.upper()}")
    print(f"{'='*60}")
    print(f"  Issue detected     : {dec['issue']}")
    print(f"  Action decided     : {dec['selected_action']}")
    print(f"  Action applied     : {r['action_applied']}")
    print(f"  Confidence         : {r['confidence']}  (threshold={r['confidence_threshold']})")
    print(f"  Reason             : {dec['reason']}")
    if "safety_note" in r:
        print(f"  [SAFETY]           : {r['safety_note']}")
    print(f"  --- Metrics ---")
    print(f"  Objects  before/after : {imp['objects_before']} -> {imp['objects_after']}")
    print(f"  Area     before/after : {imp['area_before']} -> {imp['area_after']}")
    if "dice_before" in imp:
        arrow = "^" if imp["dice_delta"] >= 0 else "v"
        print(f"  Dice     before/after : {imp['dice_before']} -> {imp['dice_after']}  ({arrow}{abs(imp['dice_delta']):.4f})")
        print(f"  IoU      before/after : {imp['iou_before']}  -> {imp['iou_after']}   ({arrow}{abs(imp['iou_delta']):.4f})")
        print(f"  Improved              : {'YES' if imp['improved'] else 'NO (already optimal)'}")


# ── Visualization ────────────────────────────────────────────────────────────

def visualize(cases: list[tuple[str, dict]], out_path: str) -> None:
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.8 * n))
    fig.suptitle("Agent-Guided Post-processing Pipeline", fontsize=14, fontweight="bold")

    col_titles = ["Input Mask", "After Agent Action", "Ground Truth"]
    for j, ct in enumerate(col_titles):
        axes[0][j].set_title(ct, fontsize=11, fontweight="bold", pad=8)

    for i, (name, r) in enumerate(cases):
        imp = r["improvement_summary"]
        action = r["action_applied"]
        conf   = r["confidence"]

        imgs = [r["mask_before"], r["mask_after"], r.get("gt_mask")]

        for j, img in enumerate(imgs):
            ax = axes[i][j]
            if img is None:
                ax.set_visible(False)
                continue
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")

            # Row label on first column
            if j == 0:
                ax.set_ylabel(name.upper(), fontsize=10, fontweight="bold", rotation=0,
                              labelpad=55, va="center")

        # Annotation below the "after" panel
        ax_after = axes[i][1]
        has_gt = "dice_after" in imp
        if has_gt:
            delta = imp["dice_delta"]
            sign  = "+" if delta >= 0 else ""
            label_txt = (
                f"action: {action}  conf={conf}\n"
                f"Dice: {imp['dice_before']} -> {imp['dice_after']}  ({sign}{delta:.4f})"
            )
            color = "#2ecc71" if delta >= 0 else "#e74c3c"
        else:
            label_txt = f"action: {action}  conf={conf}"
            color = "white"

        ax_after.text(
            0.5, -0.04, label_txt,
            transform=ax_after.transAxes,
            ha="center", va="top", fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#111")
    print(f"\nVisualization saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    test_cases = {
        "clean":  make_clean(),
        "noisy":  make_noisy(),
        "holey":  make_holey(),
        "merged": make_merged(),
    }

    results = []
    for name, (mask, gt) in test_cases.items():
        r = run_postprocessing_pipeline(mask, gt_mask=gt, confidence_threshold=0.30)
        r["gt_mask"] = gt
        results.append((name, r))
        print_report(name, r)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Case':<10} {'Action Applied':<26} {'Dice Before':>11} {'Dice After':>10} {'Delta':>8}")
    print(f"  {'-'*10} {'-'*26} {'-'*11} {'-'*10} {'-'*8}")
    for name, r in results:
        imp = r["improvement_summary"]
        db  = imp.get("dice_before", "-")
        da  = imp.get("dice_after",  "-")
        dd  = imp.get("dice_delta",  "-")
        sign = "+" if isinstance(dd, float) and dd >= 0 else ""
        dd_str = f"{sign}{dd:.4f}" if isinstance(dd, float) else str(dd)
        print(f"  {name:<10} {r['action_applied']:<26} {str(db):>11} {str(da):>10} {dd_str:>8}")

    visualize(
        results,
        out_path=os.path.join(os.path.dirname(__file__), "..", "outputs", "visualizations", "pipeline_demo.png"),
    )
