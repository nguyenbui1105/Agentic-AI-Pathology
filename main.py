"""
Agentic AI Pathology — quick demo.

Runs the hybrid post-processing pipeline on four synthetic gland masks
(clean, noisy, holey, merged) and prints a summary table.

Usage:
    python main.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.hybrid_pipeline import run_hybrid_pipeline


# ── Synthetic mask generators ─────────────────────────────────────────────────

def _circle(h, w, cy, cx, r):
    arr = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    arr[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 255
    return arr


def make_clean(h=256, w=256):
    mask = _circle(h, w, 128, 128, 60)
    return mask, mask.copy()


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
    mask[(yy - 128) ** 2 + (xx - 80)  ** 2 <= 50 ** 2] = 255
    mask[(yy - 128) ** 2 + (xx - 175) ** 2 <= 50 ** 2] = 255
    mask[115:141, 80:175] = 255
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[(yy - 128) ** 2 + (xx - 80)  ** 2 <= 50 ** 2] = 255
    gt[(yy - 128) ** 2 + (xx - 175) ** 2 <= 50 ** 2] = 255
    return mask, gt


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(cases, out_path):
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.8 * n))
    fig.suptitle("Agentic Pathology Post-processing — Hybrid Pipeline",
                 fontsize=14, fontweight="bold")

    col_titles = ["Input Mask", "After Agent Action", "Ground Truth"]
    for j, ct in enumerate(col_titles):
        axes[0][j].set_title(ct, fontsize=11, fontweight="bold", pad=8)

    for i, (name, r) in enumerate(cases):
        imp    = r["improvement_summary"]
        action = r["action_applied"]
        conf   = r["confidence"]
        imgs   = [r["mask_before"], r["mask_after"], r.get("gt_mask")]

        for j, img in enumerate(imgs):
            ax = axes[i][j]
            if img is None:
                ax.set_visible(False)
                continue
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(name.upper(), fontsize=10, fontweight="bold",
                              rotation=0, labelpad=55, va="center")

        ax_after = axes[i][1]
        has_gt   = "dice_after" in imp
        if has_gt:
            delta    = imp["dice_delta"]
            sign     = "+" if delta >= 0 else ""
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
            ha="center", va="top", fontsize=8, color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#111")
    print(f"Visualization saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    test_cases = {
        "clean":  make_clean(),
        "noisy":  make_noisy(),
        "holey":  make_holey(),
        "merged": make_merged(),
    }

    results = []
    for name, (mask, gt) in test_cases.items():
        r = run_hybrid_pipeline(mask, gt_mask=gt, confidence_threshold=0.30)
        r["gt_mask"] = gt
        results.append((name, r))

    # Summary table
    print(f"\n{'='*72}")
    print(f"  {'Case':<10} {'Strategy':<10} {'Action Applied':<28} {'Dice Delta':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*28} {'-'*10}")
    for name, r in results:
        imp      = r["improvement_summary"]
        strategy = r.get("selected_strategy", "—")
        dd       = imp.get("dice_delta", None)
        sign     = "+" if dd is not None and dd >= 0 else ""
        dd_str   = f"{sign}{dd:.4f}" if dd is not None else "—"
        print(f"  {name:<10} {strategy:<10} {r['action_applied']:<28} {dd_str:>10}")
    print(f"{'='*72}\n")

    visualize(results, out_path="outputs/visualizations/pipeline_demo.png")


if __name__ == "__main__":
    main()
