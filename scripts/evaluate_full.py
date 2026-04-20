"""
Full evaluation of the hybrid post-processing pipeline on all GlaS samples.

Folder structure expected:
    data/glas_sample/
    |-- image/    *.png   (original H&E images, for visualization)
    |-- mask/     *.png   (binary / instance ground-truth masks)
    `-- pred/     *.png   (binary predictions: 0 or 255)

Run:
    cd "d:/SidePrj/Agentic AI"
    python scripts/evaluate_full.py

Outputs:
    outputs/glas_eval/full_report.csv
    outputs/visualizations/<case>_eval.png   (selected cases)
"""

from __future__ import annotations

import os
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.hybrid_pipeline import run_hybrid_pipeline
from src.pipeline import reset_agent_session

# ── Config ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
DATA_ROOT   = ROOT / "data" / "glas_sample"
EVAL_DIR    = ROOT / "outputs" / "glas_eval"
VIZ_DIR     = ROOT / "outputs" / "visualizations"
CONF_THRESH = 0.30

IMPROVED_THRESH  = 0.001    # dice_delta >  this  -> "improved"
UNCHANGED_THRESH = 0.001    # |dice_delta| <= this -> "unchanged"

CSV_FIELDS = [
    "sample_id",
    "issue_detected", "action_applied", "confidence",
    "selected_strategy", "rollback_happened",
    "dice_before", "dice_after", "dice_delta",
    "iou_before",  "iou_after",  "iou_delta",
    "objects_before", "objects_after",
]


# ── Dataset loading ───────────────────────────────────────────────────────────

def discover_samples(data_root: Path) -> list[dict]:
    """
    Match sample IDs across image/, mask/, pred/ by filename stem.
    Returns list of dicts; img_path is optional (None if missing).
    Raises FileNotFoundError if no matching pairs found.
    """
    def index_dir(folder: Path) -> dict[str, Path]:
        if not folder.is_dir():
            return {}
        return {p.stem: p for p in sorted(folder.iterdir()) if p.is_file()}

    images = index_dir(data_root / "image")
    masks  = index_dir(data_root / "mask")
    preds  = index_dir(data_root / "pred")

    common = sorted(set(masks) & set(preds))
    if not common:
        raise FileNotFoundError(
            f"No matching sample IDs found between mask/ and pred/ in {data_root}"
        )

    return [
        {
            "sample_id": sid,
            "img_path":  images.get(sid),
            "gt_path":   masks[sid],
            "pred_path": preds[sid],
        }
        for sid in common
    ]


def load_mask_binary(path: Path, threshold: int = 0) -> np.ndarray:
    """Load a mask PNG and binarize to 0/255. threshold=0 handles instance maps."""
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise IOError(f"Cannot read mask: {path}")
    return (raw > threshold).astype(np.uint8) * 255


def load_image_rgb(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Single-sample evaluation ──────────────────────────────────────────────────

def evaluate_sample(sample: dict) -> dict:
    """
    Run the hybrid pipeline on one sample and return a flat result dict.
    Returns an error dict on failure.
    """
    sid = sample["sample_id"]
    try:
        pred_mask = load_mask_binary(sample["pred_path"], threshold=127)
        gt_mask   = load_mask_binary(sample["gt_path"],   threshold=0)
        image_rgb = load_image_rgb(sample["img_path"])

        # Align spatial resolution if model output differs from GT.
        if pred_mask.shape != gt_mask.shape:
            h, w = gt_mask.shape
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        result = run_hybrid_pipeline(
            pred_mask,
            gt_mask=gt_mask,
            confidence_threshold=CONF_THRESH,
        )

        imp = result["improvement_summary"]
        dec = result["decision"]

        # Did the hybrid reject all candidates and keep the original?
        rollback = int(result.get("selected_strategy", "original") == "original")

        return {
            # identifiers
            "sample_id":         sid,
            # agent decision
            "issue_detected":    dec["issue"],
            "action_applied":    result["action_applied"],
            "confidence":        result["confidence"],
            # strategy
            "selected_strategy": result.get("selected_strategy", "original"),
            "rollback_happened": rollback,
            # Dice / IoU
            "dice_before":       imp["dice_before"],
            "dice_after":        imp["dice_after"],
            "dice_delta":        imp["dice_delta"],
            "iou_before":        imp["iou_before"],
            "iou_after":         imp["iou_after"],
            "iou_delta":         imp["iou_delta"],
            # objects
            "objects_before":    imp["objects_before"],
            "objects_after":     imp["objects_after"],
            # extra (not in CSV, used for viz selection)
            "_pred_mask":        pred_mask,
            "_gt_mask":          gt_mask,
            "_image_rgb":        image_rgb,
            "_mask_before":      result["mask_before"],
            "_mask_after":       result["mask_after"],
            "_spread":           result.get("spread", {}),
            "_region_log":       result.get("region_log", []),
            "_score_global":     result.get("score_global", 0.0),
            "_score_region":     result.get("score_region", 0.0),
            "_strategy_reason":  result.get("strategy_reason", ""),
            "_acceptance_reason":result.get("acceptance_reason", ""),
            "_error":            None,
        }

    except Exception as exc:
        return {"sample_id": sid, "_error": str(exc)}


# ── Visualization ─────────────────────────────────────────────────────────────

def save_viz(row: dict, out_path: Path) -> None:
    """
    Four-panel figure: [Original H&E | Prediction | After Agent | Ground Truth]
    If no H&E image is available the panel is replaced by a plain grey square.
    """
    image_rgb   = row["_image_rgb"]
    pred_mask   = row["_pred_mask"]
    mask_after  = row["_mask_after"]
    gt_mask     = row["_gt_mask"]

    if image_rgb is None:
        h, w    = gt_mask.shape
        image_rgb = np.full((h, w, 3), 80, dtype=np.uint8)

    panels = [image_rgb, pred_mask, mask_after, gt_mask]
    titles = ["Original Image", "Pred Mask (before)", "After Agent", "Ground Truth"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#111")
    fig.suptitle(
        f"{row['sample_id']}  |  issue: {row['issue_detected']}  |  "
        f"action: {row['action_applied']}  |  strategy: {row['selected_strategy']}",
        fontsize=11, color="white", fontweight="bold",
    )

    for ax, img, title in zip(axes, panels, titles):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9, color="white", pad=4)
        ax.axis("off")
        ax.set_facecolor("#111")

    # Metrics overlay on the "After" panel
    dd      = row["dice_delta"]
    di      = row["iou_delta"]
    sign_d  = "+" if dd >= 0 else ""
    sign_i  = "+" if di >= 0 else ""
    color   = "#2ecc71" if dd >= 0 else "#e74c3c"
    note = (
        f"conf={row['confidence']:.2f}\n"
        f"Dice: {row['dice_before']:.4f} -> {row['dice_after']:.4f}  ({sign_d}{dd:.4f})\n"
        f"IoU:  {row['iou_before']:.4f} -> {row['iou_after']:.4f}   ({sign_i}{di:.4f})"
    )
    axes[2].text(
        0.5, -0.03, note,
        transform=axes[2].transAxes,
        ha="center", va="top", fontsize=8, color=color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#222", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close(fig)


# ── Case selection ────────────────────────────────────────────────────────────

def _is_interesting(row: dict) -> bool:
    """Multi-step, fragmented, mixed issues, or multiple regions modified."""
    action  = row.get("action_applied", "")
    issue   = row.get("issue_detected", "")
    spread  = row.get("_spread", {})
    r_log   = row.get("_region_log", [])
    n_mod   = sum(1 for e in r_log if e.get("applied"))

    return (
        "connect_fragments" in action
        or "watershed_split" in action
        or issue in ("fragmented", "merged", "thin_bridge")
        or spread.get("is_mixed", False)
        or n_mod >= 2
    )


def select_viz_cases(rows: list[dict]) -> list[tuple[str, dict]]:
    """
    Return a labelled list of (label, row) tuples for visualization.

    Selection:
      - Top 3 by Dice improvement
      - Up to 2 interesting cases (not already in top 3)
      - 1 safety / rollback case (rollback or no_action)
    """
    valid = [r for r in rows if r.get("_error") is None]
    by_dice = sorted(valid, key=lambda r: -r["dice_delta"])

    selected: dict[str, dict] = {}

    # Top 3 improvements
    for r in by_dice:
        if len(selected) >= 3:
            break
        selected[r["sample_id"]] = ("top_improvement", r)

    # Interesting cases
    n_interesting = 0
    for r in sorted(valid, key=lambda r: -r["dice_delta"]):
        if n_interesting >= 2:
            break
        if r["sample_id"] in selected:
            continue
        if _is_interesting(r):
            selected[r["sample_id"]] = ("interesting", r)
            n_interesting += 1

    # Safety / rollback case
    safety_candidates = sorted(
        [r for r in valid if r["rollback_happened"] or r["action_applied"] == "no_action"],
        key=lambda r: abs(r["dice_delta"]),
    )
    for r in safety_candidates:
        if r["sample_id"] not in selected:
            selected[r["sample_id"]] = ("safety_rollback", r)
            break

    return [(label, r) for label, r in selected.values()]


# ── CSV ───────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if row.get("_error") is None:
                writer.writerow({k: row[k] for k in CSV_FIELDS})


# ── Summary statistics ────────────────────────────────────────────────────────

def print_summary(rows: list[dict]) -> None:
    valid   = [r for r in rows if r.get("_error") is None]
    errors  = [r for r in rows if r.get("_error") is not None]
    n       = len(valid)

    improved  = [r for r in valid if r["dice_delta"] >  IMPROVED_THRESH]
    unchanged = [r for r in valid if abs(r["dice_delta"]) <= UNCHANGED_THRESH]
    worsened  = [r for r in valid if r["dice_delta"] < -UNCHANGED_THRESH]
    rollbacks = [r for r in valid if r["rollback_happened"]]

    deltas = [r["dice_delta"] for r in valid]
    avg_delta = sum(deltas) / n if n else 0.0
    max_delta = max(deltas) if deltas else 0.0
    min_delta = min(deltas) if deltas else 0.0

    avg_before = sum(r["dice_before"] for r in valid) / n if n else 0.0
    avg_after  = sum(r["dice_after"]  for r in valid) / n if n else 0.0

    print()
    print("=" * 65)
    print("  SUMMARY STATISTICS")
    print("=" * 65)
    print(f"  Total samples        : {n}")
    print(f"  Errors / skipped     : {len(errors)}")
    print(f"  Improved  (D >+0.001): {len(improved):>3}  ({100*len(improved)/n:.0f}%)")
    print(f"  Unchanged (|D|<=0.001): {len(unchanged):>3}  ({100*len(unchanged)/n:.0f}%)")
    print(f"  Worsened  (D <-0.001): {len(worsened):>3}  ({100*len(worsened)/n:.0f}%)")
    print(f"  Rollbacks            : {len(rollbacks):>3}")
    print()
    print(f"  Avg Dice before      : {avg_before:.4f}")
    print(f"  Avg Dice after       : {avg_after:.4f}")
    print(f"  Avg Dice delta       : {avg_delta:+.4f}")
    print(f"  Max improvement      : {max_delta:+.4f}")
    print(f"  Min / worst delta    : {min_delta:+.4f}")
    print()

    # Action distribution
    action_counts: dict[str, int] = {}
    for r in valid:
        a = r["action_applied"]
        action_counts[a] = action_counts.get(a, 0) + 1

    print("  Action distribution:")
    for action, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        bar = "#" * cnt
        print(f"    {action:<35} {cnt:>2}  {bar}")

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
    print("=" * 65)


def print_ranked(rows: list[dict]) -> None:
    valid = [r for r in rows if r.get("_error") is None]
    by_dice = sorted(valid, key=lambda r: -r["dice_delta"])

    print()
    print("=" * 65)
    print("  TOP 5  -  Highest Dice Improvement")
    print("=" * 65)
    print(f"  {'Sample':<12} {'Issue':<18} {'Action':<32} {'D Dice':>8}")
    print(f"  {'-'*12} {'-'*18} {'-'*32} {'-'*8}")
    for r in by_dice[:5]:
        sign = "+" if r["dice_delta"] >= 0 else ""
        print(f"  {r['sample_id']:<12} {r['issue_detected']:<18} "
              f"{r['action_applied']:<32} {sign}{r['dice_delta']:.4f}")

    worst = [r for r in by_dice if r["dice_delta"] < -UNCHANGED_THRESH]
    if worst:
        print()
        print("=" * 65)
        print("  WORST CASES  -  Dice Regression")
        print("=" * 65)
        print(f"  {'Sample':<12} {'Issue':<18} {'Action':<32} {'D Dice':>8}")
        print(f"  {'-'*12} {'-'*18} {'-'*32} {'-'*8}")
        for r in sorted(worst, key=lambda r: r["dice_delta"])[:3]:
            print(f"  {r['sample_id']:<12} {r['issue_detected']:<18} "
                  f"{r['action_applied']:<32} {r['dice_delta']:+.4f}")
    else:
        print()
        print("  No regressions detected.")

    rollbacks = [r for r in valid if r["rollback_happened"]]
    print()
    print("=" * 65)
    print(f"  ROLLBACKS  -  {len(rollbacks)} case(s) kept original")
    print("=" * 65)
    if rollbacks:
        print(f"  {'Sample':<12} {'Issue':<18} {'Strategy Reason'}")
        print(f"  {'-'*12} {'-'*18} {'-'*35}")
        for r in rollbacks:
            reason = r.get("_strategy_reason", "")[:55]
            print(f"  {r['sample_id']:<12} {r['issue_detected']:<18} {reason}")
    else:
        print("  None - all applied actions were accepted.")
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(DATA_ROOT)
    print(f"\nFound {len(samples)} samples in {DATA_ROOT}")
    print(f"Running hybrid pipeline (confidence threshold={CONF_THRESH})...\n")

    reset_agent_session()

    rows: list[dict] = []
    for i, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        print(f"[{i:>2}/{len(samples)}] {sid} ... ", end="", flush=True)
        row = evaluate_sample(sample)
        rows.append(row)
        if row.get("_error"):
            print(f"ERROR: {row['_error']}")
        else:
            sign = "+" if row["dice_delta"] >= 0 else ""
            print(
                f"issue={row['issue_detected']:<18}  "
                f"action={row['action_applied']:<32}  "
                f"Dice {row['dice_before']:.4f}->{row['dice_after']:.4f} "
                f"({sign}{row['dice_delta']:.4f})  "
                f"[{row['selected_strategy']}]"
            )

    # ── CSV report ────────────────────────────────────────────────────────────
    csv_path = EVAL_DIR / "full_report.csv"
    write_csv(rows, csv_path)
    print(f"\nCSV saved -> {csv_path}")

    # ── Summary + rankings ────────────────────────────────────────────────────
    print_summary(rows)
    print_ranked(rows)

    # ── Selected visualizations ───────────────────────────────────────────────
    cases = select_viz_cases(rows)
    print(f"\nGenerating {len(cases)} visualization(s)...")
    for label, row in cases:
        out_path = VIZ_DIR / f"{row['sample_id']}_{label}_eval.png"
        save_viz(row, out_path)
        print(f"  [{label}]  {row['sample_id']}  D={row['dice_delta']:+.4f}  -> {out_path.name}")

    print(f"\nDone. Visualizations saved -> {VIZ_DIR}")


if __name__ == "__main__":
    main()
