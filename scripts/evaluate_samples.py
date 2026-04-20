"""
Full evaluation of the hybrid post-processing pipeline on data/samples/.

Folder structure:
    data/samples/
    |-- image/    *.png   (original H&E images)
    |-- mask/     *.png   (ground-truth masks: binary or instance label maps)
    `-- pred/     *.png   (binary predictions: 0 or 255)

Run:
    cd "d:/SidePrj/Agentic AI"
    python scripts/evaluate_samples.py

Outputs:
    outputs/data_samples_eval/full_report.csv
    outputs/visualizations/<sample>_<label>_eval.png
    outputs/visualizations/combined.png
"""

from __future__ import annotations

import os
import sys
import csv
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.hybrid_pipeline import run_hybrid_pipeline
from src.pipeline import reset_agent_session

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "samples"
EVAL_DIR  = ROOT / "outputs" / "data_samples_eval"
VIZ_DIR   = ROOT / "outputs" / "visualizations"

# ── Constants ─────────────────────────────────────────────────────────────────

CONF_THRESH      = 0.30
IMPROVED_THRESH  = 0.002
UNCHANGED_THRESH = 0.002

CSV_FIELDS = [
    "sample_id",
    "issue_detected", "action_applied", "confidence",
    "selected_strategy", "rollback_happened",
    "dice_before", "dice_after", "dice_delta",
    "iou_before",  "iou_after",  "iou_delta",
    "objects_before", "objects_after",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def discover_samples(data_root: Path) -> list[dict]:
    """
    Build sample list by iterating pred/ and matching image/ and mask/.
    Skips samples missing GT; image is optional.
    """
    pred_dir  = data_root / "pred"
    mask_dir  = data_root / "mask"
    image_dir = data_root / "image"

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"pred/ not found in {data_root}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask/ not found in {data_root}")

    def stem_index(folder: Path) -> dict[str, Path]:
        return {p.stem: p for p in sorted(folder.iterdir()) if p.is_file()}

    def pred_index(folder: Path) -> dict[str, Path]:
        """Index pred files, stripping a trailing '_pred' suffix from stems."""
        result = {}
        for p in sorted(folder.iterdir()):
            if not p.is_file():
                continue
            sid = p.stem[:-5] if p.stem.endswith("_pred") else p.stem
            result[sid] = p
        return result

    preds  = pred_index(pred_dir)
    masks  = stem_index(mask_dir)
    images = stem_index(image_dir) if image_dir.is_dir() else {}

    samples = []
    missing_gt = []
    for sid, pred_path in sorted(preds.items()):
        if sid not in masks:
            missing_gt.append(sid)
            continue
        samples.append({
            "sample_id": sid,
            "pred_path": pred_path,
            "gt_path":   masks[sid],
            "img_path":  images.get(sid),
        })

    if missing_gt:
        print(f"  [WARN] {len(missing_gt)} pred file(s) have no GT mask — skipped: "
              f"{', '.join(missing_gt[:5])}{'...' if len(missing_gt) > 5 else ''}")
    return samples


def _load_gray_binary(path: Path, threshold: int) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise IOError(f"Cannot read: {path}")
    return (raw > threshold).astype(np.uint8) * 255


def load_gt(path: Path)   -> np.ndarray: return _load_gray_binary(path, threshold=0)
def load_pred(path: Path) -> np.ndarray: return _load_gray_binary(path, threshold=127)


def load_image_rgb(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_sample(sample: dict) -> dict:
    """Run the hybrid pipeline on one sample. Returns a flat result dict."""
    sid = sample["sample_id"]
    try:
        pred = load_pred(sample["pred_path"])
        gt   = load_gt(sample["gt_path"])
        img  = load_image_rgb(sample["img_path"])

        if pred.shape != gt.shape:
            h, w = gt.shape
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        result = run_hybrid_pipeline(pred, gt_mask=gt, confidence_threshold=CONF_THRESH)

        imp  = result["improvement_summary"]
        dec  = result["decision"]

        return {
            "sample_id":         sid,
            "issue_detected":    dec["issue"],
            "action_applied":    result["action_applied"],
            "confidence":        result["confidence"],
            "selected_strategy": result.get("selected_strategy", "original"),
            "rollback_happened": int(result.get("selected_strategy", "original") == "original"),
            "dice_before":       imp["dice_before"],
            "dice_after":        imp["dice_after"],
            "dice_delta":        imp["dice_delta"],
            "iou_before":        imp["iou_before"],
            "iou_after":         imp["iou_after"],
            "iou_delta":         imp["iou_delta"],
            "objects_before":    imp["objects_before"],
            "objects_after":     imp["objects_after"],
            # visualization fields (not written to CSV)
            "_img":          img,
            "_pred":         pred,
            "_mask_before":  result["mask_before"],
            "_mask_after":   result["mask_after"],
            "_gt":           gt,
            "_spread":       result.get("spread", {}),
            "_region_log":   result.get("region_log", []),
            "_strat_reason": result.get("strategy_reason", ""),
            "_error":        None,
        }

    except Exception as exc:
        return {"sample_id": sid, "_error": str(exc)}


# ── Case selection ────────────────────────────────────────────────────────────

def _is_interesting(row: dict) -> bool:
    """Multi-step, fragmented/merged issues, mixed spread, or multi-region."""
    action  = row.get("action_applied", "")
    issue   = row.get("issue_detected", "")
    spread  = row.get("_spread", {})
    n_mod   = sum(1 for e in row.get("_region_log", []) if e.get("applied"))
    return (
        "connect_fragments" in action
        or "watershed_split" in action
        or issue in ("fragmented", "merged", "thin_bridge")
        or spread.get("is_mixed", False)
        or n_mod >= 2
    )


def select_cases(rows: list[dict]) -> list[tuple[str, dict]]:
    """
    Returns a labelled list (label, row) for visualization:
      - top_improvement x3
      - interesting      x2
      - safety_rollback  x1
    """
    valid    = [r for r in rows if not r.get("_error")]
    by_delta = sorted(valid, key=lambda r: -r["dice_delta"])

    selected: dict[str, tuple[str, dict]] = {}

    for r in by_delta:
        if len(selected) >= 3:
            break
        selected[r["sample_id"]] = ("top_improvement", r)

    for r in by_delta:
        if sum(1 for lbl, _ in selected.values() if lbl == "interesting") >= 2:
            break
        if r["sample_id"] not in selected and _is_interesting(r):
            selected[r["sample_id"]] = ("interesting", r)

    safety = sorted(
        [r for r in valid if r["rollback_happened"] or r["action_applied"] == "no_action"],
        key=lambda r: abs(r["dice_delta"]),
    )
    for r in safety:
        if r["sample_id"] not in selected:
            selected[r["sample_id"]] = ("safety_rollback", r)
            break

    return list(selected.values())


# ── Visualization: single case ─────────────────────────────────────────────────

_BG = "#111111"


def _make_panel_fig(row: dict, title_prefix: str = "") -> plt.Figure:
    """
    Four-panel figure: [Original | Prediction | After Agent | Ground Truth]
    """
    img_rgb = row["_img"]
    h, w    = row["_gt"].shape
    if img_rgb is None:
        img_rgb = np.full((h, w, 3), 80, dtype=np.uint8)

    panels = [img_rgb, row["_pred"], row["_mask_after"], row["_gt"]]
    titles = ["Original Image", "Pred (before)", "After Agent", "Ground Truth"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(_BG)

    header = f"{title_prefix}{row['sample_id']}  |  " \
             f"issue: {row['issue_detected']}  |  " \
             f"strategy: {row['selected_strategy']}"
    fig.suptitle(header, fontsize=10, color="white", fontweight="bold")

    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(img if img.ndim == 3 else img, cmap="gray" if img.ndim == 2 else None,
                  vmin=(0 if img.ndim == 2 else None), vmax=(255 if img.ndim == 2 else None))
        ax.set_title(title, fontsize=8, color="white", pad=3)
        ax.axis("off")
        ax.set_facecolor(_BG)

    # Overlay ONLY on "After Agent" panel
    dd    = row["dice_delta"]
    di    = row["iou_delta"]
    color = "#2ecc71" if dd >= 0 else "#e74c3c"
    sign_d, sign_i = ("+" if dd >= 0 else ""), ("+" if di >= 0 else "")
    note  = (
        f"action: {row['action_applied']}\n"
        f"conf: {row['confidence']:.2f}\n"
        f"Dice: {row['dice_before']:.4f} -> {row['dice_after']:.4f}  ({sign_d}{dd:.4f})\n"
        f"IoU:  {row['iou_before']:.4f} -> {row['iou_after']:.4f}   ({sign_i}{di:.4f})"
    )
    axes[2].text(
        0.5, -0.02, note,
        transform=axes[2].transAxes,
        ha="center", va="top", fontsize=7.5, color=color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a", alpha=0.92),
    )

    plt.tight_layout(rect=[0, 0.13, 1, 0.93])
    return fig


def save_single_viz(row: dict, label: str, out_path: Path) -> None:
    fig = _make_panel_fig(row, title_prefix=f"[{label}]  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


def save_combined_viz(cases: list[tuple[str, dict]], out_path: Path) -> None:
    """
    Grid of all selected cases stacked vertically into one combined PNG.
    """
    n = len(cases)
    fig, axes_grid = plt.subplots(n, 4, figsize=(18, 5 * n))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Selected Cases - Agent Post-processing Results",
                 fontsize=13, color="white", fontweight="bold", y=1.002)

    col_titles = ["Original Image", "Pred (before)", "After Agent", "Ground Truth"]
    for j, ct in enumerate(col_titles):
        axes_grid[0][j].set_title(ct, fontsize=9, color="white", pad=4, fontweight="bold")

    for i, (label, row) in enumerate(cases):
        img_rgb = row["_img"]
        h, w    = row["_gt"].shape
        if img_rgb is None:
            img_rgb = np.full((h, w, 3), 80, dtype=np.uint8)

        panels = [img_rgb, row["_pred"], row["_mask_after"], row["_gt"]]
        for j, img in enumerate(panels):
            ax = axes_grid[i][j]
            kwargs = {"cmap": "gray", "vmin": 0, "vmax": 255} if img.ndim == 2 else {}
            ax.imshow(img, **kwargs)
            ax.axis("off")
            ax.set_facecolor(_BG)

        # Row label
        dd    = row["dice_delta"]
        color = "#2ecc71" if dd >= 0 else "#e74c3c"
        sign  = "+" if dd >= 0 else ""
        row_label = (f"[{label}] {row['sample_id']}\n"
                     f"{row['issue_detected']} / {row['selected_strategy']}\n"
                     f"D{sign}{dd:.4f}")
        axes_grid[i][0].set_ylabel(row_label, fontsize=7.5, color=color,
                                   rotation=0, labelpad=60, va="center")

        # Overlay on After panel
        note = (
            f"conf={row['confidence']:.2f}\n"
            f"Dice {row['dice_before']:.4f}->{row['dice_after']:.4f} ({sign}{dd:.4f})"
        )
        axes_grid[i][2].text(
            0.5, -0.03, note,
            transform=axes_grid[i][2].transAxes,
            ha="center", va="top", fontsize=7, color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", alpha=0.9),
        )

    plt.tight_layout(rect=[0.05, 0.01, 1.0, 0.995])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


# ── CSV ───────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if not r.get("_error")]
    with open(str(path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid)


# ── Summary statistics ────────────────────────────────────────────────────────

def print_summary(rows: list[dict]) -> None:
    valid   = [r for r in rows if not r.get("_error")]
    errors  = [r for r in rows if r.get("_error")]
    n       = len(valid)
    if n == 0:
        print("  No valid samples.")
        return

    improved  = [r for r in valid if r["dice_delta"] >  IMPROVED_THRESH]
    unchanged = [r for r in valid if abs(r["dice_delta"]) <= UNCHANGED_THRESH]
    worsened  = [r for r in valid if r["dice_delta"] < -IMPROVED_THRESH]
    rollbacks = [r for r in valid if r["rollback_happened"]]

    deltas     = [r["dice_delta"] for r in valid]
    avg_delta  = sum(deltas) / n
    max_delta  = max(deltas)
    min_delta  = min(deltas)
    avg_before = sum(r["dice_before"] for r in valid) / n
    avg_after  = sum(r["dice_after"]  for r in valid) / n

    print()
    print("=" * 68)
    print("  SUMMARY STATISTICS")
    print("=" * 68)
    print(f"  Total samples          : {n}")
    if errors:
        print(f"  Errors / skipped       : {len(errors)}")
    print(f"  Improved  (D > +0.002) : {len(improved):>3}  ({100*len(improved)/n:.0f}%)")
    print(f"  Unchanged (|D| <= 0.002): {len(unchanged):>3}  ({100*len(unchanged)/n:.0f}%)")
    print(f"  Worsened  (D < -0.002) : {len(worsened):>3}  ({100*len(worsened)/n:.0f}%)")
    print(f"  Rollbacks              : {len(rollbacks):>3}")
    print()
    print(f"  Avg Dice before        : {avg_before:.4f}")
    print(f"  Avg Dice after         : {avg_after:.4f}")
    print(f"  Avg Dice delta         : {avg_delta:+.4f}")
    print(f"  Max improvement        : {max_delta:+.4f}")
    print(f"  Min / worst delta      : {min_delta:+.4f}")
    print()

    # Action distribution
    action_counts: dict[str, int] = {}
    for r in valid:
        a = r["action_applied"]
        action_counts[a] = action_counts.get(a, 0) + 1

    print("  Action distribution:")
    for action, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        bar = "#" * cnt
        print(f"    {action:<42} {cnt:>2}  {bar}")

    # Strategy distribution
    strat_counts: dict[str, int] = {}
    for r in valid:
        s = r["selected_strategy"]
        strat_counts[s] = strat_counts.get(s, 0) + 1
    print()
    print("  Strategy distribution:")
    for strat, cnt in sorted(strat_counts.items(), key=lambda x: -x[1]):
        print(f"    {strat:<12} {cnt:>2}")

    if errors:
        print()
        print("  Errors:")
        for r in errors:
            print(f"    {r['sample_id']}: {r['_error']}")

    print("=" * 68)


# ── Ranked output ─────────────────────────────────────────────────────────────

def print_ranked(rows: list[dict]) -> None:
    valid    = [r for r in rows if not r.get("_error")]
    by_delta = sorted(valid, key=lambda r: -r["dice_delta"])

    hdr = f"  {'Sample':<12} {'Issue':<18} {'Strategy':<10} {'Action':<34} {'D Dice':>8}"
    sep = f"  {'-'*12} {'-'*18} {'-'*10} {'-'*34} {'-'*8}"

    print()
    print("=" * 68)
    print("  TOP 5 - Highest Dice Improvement")
    print("=" * 68)
    print(hdr)
    print(sep)
    for r in by_delta[:5]:
        sign = "+" if r["dice_delta"] >= 0 else ""
        act  = r["action_applied"][:34]
        print(f"  {r['sample_id']:<12} {r['issue_detected']:<18} "
              f"{r['selected_strategy']:<10} {act:<34} {sign}{r['dice_delta']:.4f}")

    worst = [r for r in by_delta if r["dice_delta"] < -UNCHANGED_THRESH]
    print()
    print("=" * 68)
    print("  WORST CASES - Dice Regression")
    print("=" * 68)
    if worst:
        print(hdr)
        print(sep)
        for r in sorted(worst, key=lambda r: r["dice_delta"])[:3]:
            act = r["action_applied"][:34]
            print(f"  {r['sample_id']:<12} {r['issue_detected']:<18} "
                  f"{r['selected_strategy']:<10} {act:<34} {r['dice_delta']:+.4f}")
    else:
        print("  None - no regressions detected.")

    rollbacks = [r for r in valid if r["rollback_happened"]]
    print()
    print("=" * 68)
    print(f"  ROLLBACKS - {len(rollbacks)} case(s) where original was kept")
    print("=" * 68)
    if rollbacks:
        print(f"  {'Sample':<12} {'Issue':<18} {'Reason (truncated)'}")
        print(f"  {'-'*12} {'-'*18} {'-'*38}")
        for r in rollbacks:
            reason = r.get("_strat_reason", "")[:55]
            print(f"  {r['sample_id']:<12} {r['issue_detected']:<18} {reason}")
    else:
        print("  None - all applied actions were accepted.")
    print("=" * 68)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(DATA_ROOT)
    print(f"\nFound {len(samples)} samples in {DATA_ROOT}")
    print(f"Pipeline: hybrid  |  confidence threshold: {CONF_THRESH}\n")

    reset_agent_session()

    rows: list[dict] = []
    for i, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        print(f"[{i:>2}/{len(samples)}] {sid} ... ", end="", flush=True)
        row = run_sample(sample)
        rows.append(row)
        if row.get("_error"):
            print(f"ERROR: {row['_error']}")
        else:
            sign = "+" if row["dice_delta"] >= 0 else ""
            act  = row["action_applied"][:38]
            print(f"issue={row['issue_detected']:<18}  "
                  f"Dice {row['dice_before']:.4f}->{row['dice_after']:.4f} "
                  f"({sign}{row['dice_delta']:.4f})  "
                  f"[{row['selected_strategy']}]  {act}")

    # CSV
    csv_path = EVAL_DIR / "full_report.csv"
    write_csv(rows, csv_path)
    print(f"\nCSV saved -> {csv_path}")

    # Summary + rankings
    print_summary(rows)
    print_ranked(rows)

    # Visualizations
    cases = select_cases(rows)
    print(f"\nGenerating {len(cases)} individual visualization(s)...")
    for label, row in cases:
        out_path = VIZ_DIR / f"{row['sample_id']}_{label}_eval.png"
        save_single_viz(row, label, out_path)
        sign = "+" if row["dice_delta"] >= 0 else ""
        print(f"  [{label:<16}]  {row['sample_id']:<12}  "
              f"D{sign}{row['dice_delta']:.4f}  ->  {out_path.name}")

    combined_path = VIZ_DIR / "combined.png"
    if cases:
        print(f"\nGenerating combined.png ({len(cases)} cases)...")
        save_combined_viz(cases, combined_path)
        print(f"  Saved -> {combined_path}")
    else:
        print("\nNo cases selected for combined visualization.")

    print(f"\nDone.")
    print(f"  CSV      -> {csv_path}")
    print(f"  Visuals  -> {VIZ_DIR}")


if __name__ == "__main__":
    main()
