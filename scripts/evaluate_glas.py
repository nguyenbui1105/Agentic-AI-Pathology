"""
Real evaluation of the agent-guided post-processing pipeline on GlaS sample data.

Folder structure expected:
    data/glas_sample/
    ├── gt_masks/      *.bmp   (instance label maps: 0=bg, 1..N=gland IDs)
    ├── images/        *.bmp   (original H&E images, for visualization only)
    └── pred_masks/    *.png   (binary predictions: 0 or 255)

Run:
    cd "d:/SidePrj/Agentic AI"
    python scripts/evaluate_glas.py

Outputs saved to:
    outputs/glas_eval/
    ├── report.csv
    └── <sample_id>_comparison.png
"""

import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline import run_postprocessing_pipeline, reset_agent_session
from src.region_pipeline import run_region_pipeline
from src.hybrid_pipeline import run_hybrid_pipeline

# PIPELINE_MODE: "global" | "region" | "hybrid"
PIPELINE_MODE = "hybrid"

# ── Config ────────────────────────────────────────────────────────────────────

GLAS_ROOT          = r"d:\SidePrj\Agentic AI\data\glas_sample"
OUTPUT_DIR         = r"d:\SidePrj\Agentic AI\outputs\glas_eval"
CONFIDENCE_THRESH  = 0.30

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_samples(root: str) -> list[dict]:
    """
    Match sample IDs across gt_masks/, pred_masks/, images/.
    Returns list of dicts with keys: sample_id, pred_path, gt_path, img_path.
    """
    gt_dir   = os.path.join(root, "gt_masks")
    pred_dir = os.path.join(root, "pred_masks")
    img_dir  = os.path.join(root, "images")

    # Build {sample_id: path} for GT (strip extension to get ID)
    gt_files = {os.path.splitext(f)[0]: os.path.join(gt_dir, f)
                for f in sorted(os.listdir(gt_dir))}

    pred_files = {os.path.splitext(f)[0]: os.path.join(pred_dir, f)
                  for f in sorted(os.listdir(pred_dir))}

    img_files = {os.path.splitext(f)[0]: os.path.join(img_dir, f)
                 for f in sorted(os.listdir(img_dir)) if os.path.isfile(os.path.join(img_dir, f))}

    # Keep only samples that have both GT and pred
    common_ids = sorted(set(gt_files) & set(pred_files))
    if not common_ids:
        raise FileNotFoundError("No matching sample IDs found between gt_masks/ and pred_masks/.")

    samples = []
    for sid in common_ids:
        samples.append({
            "sample_id": sid,
            "gt_path":   gt_files[sid],
            "pred_path": pred_files[sid],
            "img_path":  img_files.get(sid),   # optional
        })
    return samples


def load_gt_binary(path: str) -> np.ndarray:
    """
    Load GT mask. GlaS GT is an instance label map (0=bg, 1..N=gland IDs).
    Binarize to 0/255.
    """
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise IOError(f"Cannot read GT mask: {path}")
    return (raw > 0).astype(np.uint8) * 255


def load_pred_binary(path: str) -> np.ndarray:
    """
    Load prediction mask. Already 0/255; threshold at 127 for safety.
    """
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise IOError(f"Cannot read pred mask: {path}")
    return (raw > 127).astype(np.uint8) * 255


def load_image_rgb(path: str | None) -> np.ndarray | None:
    if path is None or not os.path.exists(path):
        return None
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Visualization ─────────────────────────────────────────────────────────────

def save_comparison(
    sample_id: str,
    image_rgb: np.ndarray | None,
    pred_mask: np.ndarray,
    refined_mask: np.ndarray,
    gt_mask: np.ndarray,
    action: str,
    confidence: float,
    dice_before: float,
    dice_after: float,
    iou_before: float,
    iou_after: float,
    out_path: str,
) -> None:
    panels = []
    titles = []

    if image_rgb is not None:
        panels.append(image_rgb)
        titles.append("Original Image")

    panels += [pred_mask, refined_mask, gt_mask]
    titles += ["Pred Mask (before)", "After Agent Action", "Ground Truth"]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    fig.patch.set_facecolor("#111")
    fig.suptitle(f"Sample: {sample_id}", fontsize=12, color="white", fontweight="bold")

    for ax, img, title in zip(axes, panels, titles):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9, color="white", pad=4)
        ax.axis("off")
        ax.set_facecolor("#111")

    # Annotation under the "after" panel (second-to-last)
    ax_after = axes[-2]
    delta_d = dice_after - dice_before
    delta_i = iou_after  - iou_before
    sign_d  = "+" if delta_d >= 0 else ""
    sign_i  = "+" if delta_i >= 0 else ""
    color   = "#2ecc71" if delta_d >= 0 else "#e74c3c"
    note = (
        f"action: {action}  conf={confidence:.2f}\n"
        f"Dice: {dice_before:.4f} -> {dice_after:.4f}  ({sign_d}{delta_d:.4f})\n"
        f"IoU:  {iou_before:.4f} -> {iou_after:.4f}  ({sign_i}{delta_i:.4f})"
    )
    ax_after.text(
        0.5, -0.03, note,
        transform=ax_after.transAxes,
        ha="center", va="top", fontsize=8, color=color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#222", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close(fig)


# ── CSV ───────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "sample_id", "issue_detected", "action_applied", "confidence",
    "rollback_happened", "selected_checkpoint", "selected_strategy",
    "dice_before", "dice_after", "dice_delta",
    "iou_before",  "iou_after",  "iou_delta",
    "objects_before", "objects_after",
]


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reset_agent_session()   # fresh diversity counters for this run

    samples = load_samples(GLAS_ROOT)
    print(f"\nFound {len(samples)} matched samples in {GLAS_ROOT}\n")

    csv_rows = []

    for s in samples:
        sid = s["sample_id"]
        print(f"--- {sid} ---")

        pred_mask = load_pred_binary(s["pred_path"])
        gt_mask   = load_gt_binary(s["gt_path"])
        image_rgb = load_image_rgb(s["img_path"])

        # Align pred to GT spatial resolution (new model outputs may differ in size).
        if pred_mask.shape != gt_mask.shape:
            h, w = gt_mask.shape
            pred_mask = cv2.resize(
                pred_mask, (w, h), interpolation=cv2.INTER_NEAREST
            )
            print(f"  [resize] pred_mask -> {gt_mask.shape}")

        if PIPELINE_MODE == "hybrid":
            result = run_hybrid_pipeline(
                pred_mask,
                gt_mask=gt_mask,
                confidence_threshold=CONFIDENCE_THRESH,
            )
        elif PIPELINE_MODE == "region":
            result = run_region_pipeline(
                pred_mask,
                gt_mask=gt_mask,
                confidence_threshold=CONFIDENCE_THRESH,
            )
        else:
            result = run_postprocessing_pipeline(
                pred_mask,
                gt_mask=gt_mask,
                confidence_threshold=CONFIDENCE_THRESH,
            )

        imp  = result["improvement_summary"]
        dec  = result["decision"]

        dice_before = imp["dice_before"]
        dice_after  = imp["dice_after"]
        iou_before  = imp["iou_before"]
        iou_after   = imp["iou_after"]
        dice_delta  = imp["dice_delta"]
        iou_delta   = imp["iou_delta"]
        action      = result["action_applied"]
        confidence  = result["confidence"]

        selected_checkpoint = result.get("selected_checkpoint", "original")

        # Per-sample console output
        breakdown = dec.get("score_breakdown", {})
        print(f"  Issue      : {dec['issue']}")
        print(f"  Action     : {action}  (confidence={confidence})")

        # Tool score distribution — shows why winner beats alternatives
        if breakdown:
            all_s = breakdown.get("all_scores", {})
            score_str = "  ".join(
                f"{t}={s:.3f}" for t, s in all_s.items()
            )
            margin = breakdown.get("margin", 0)
            ru     = breakdown.get("runner_up")
            if ru:
                print(f"  Scores     : {score_str}  |  margin={margin:.3f} over {ru}")
            else:
                print(f"  Scores     : {score_str}")
        if "safety_note" in result:
            print(f"  [SAFETY]   : {result['safety_note']}")
        if "rollback_note" in result:
            print(f"  [ROLLBACK] : {result['rollback_note']}")

        # Hybrid-specific output
        if PIPELINE_MODE == "hybrid":
            strategy = result.get("selected_strategy", "?")
            s_reason = result.get("strategy_reason", "")
            s_orig   = result.get("score_original", 0.0)
            s_glob   = result.get("score_global",   0.0)
            s_reg    = result.get("score_region",   0.0)
            spread   = result.get("spread", {})
            acc_reason = result.get("acceptance_reason", "")
            print(f"  Strategy   : [{strategy.upper()}]  {s_reason}")
            if acc_reason:
                print(f"  Acceptance : {acc_reason}")
            print(f"  Scores     : orig={s_orig:.4f}  global={s_glob:.4f}  region={s_reg:.4f}")
            if spread:
                print(f"  Spread     : dominant='{spread.get('dominant_issue','?')}' "
                      f"({spread.get('dominant_frac', 0):.0%} of "
                      f"{spread.get('n_significant', 0)} regions)  "
                      f"widespread={spread.get('is_widespread', False)}  "
                      f"mixed={spread.get('is_mixed', False)}")

        # Step-wise validation log
        step_log = result.get("step_log", [])
        if step_log:
            print(f"  Steps ({len(step_log)}):")
            for entry in step_log:
                status = "ACCEPTED" if entry["accepted"] else "REJECTED"
                sig    = entry.get("signals", {})
                area_r = sig.get("area_ratio", "?")
                n_prev = sig.get("n_prev", "?")
                n_curr = sig.get("n_curr", "?")
                comp   = sig.get("compactness", "?")
                area_r_str = f"{area_r:.3f}" if isinstance(area_r, float) else str(area_r)
                comp_str   = f"{comp:.3f}"   if isinstance(comp, float)   else str(comp)
                print(f"    Step {entry['step']} [{status}] {entry['action']}: "
                      f"area_ratio={area_r_str}, "
                      f"obj={n_prev}->{n_curr}, "
                      f"compactness={comp_str}")
                if entry.get("reason"):
                    print(f"      Reason: {entry['reason']}")

        print(f"  Selected   : {selected_checkpoint}")

        # Region log (region-aware pipeline only)
        region_log = result.get("region_log", [])
        n_regions  = result.get("n_regions", 0)
        n_modified = result.get("regions_modified", 0)
        if region_log:
            print(f"  Regions    : {n_regions} analyzed, {n_modified} modified")
            for entry in region_log:
                applied_str = "APPLIED" if entry["applied"] else "skipped"
                ts = entry.get("tool_scores", {})
                ts_str = "  ".join(f"{t}={s:.2f}" for t, s in ts.items())
                reason_str = f"  [{entry['reason']}]" if entry.get("reason") and not entry["applied"] else ""
                print(
                    f"    r{entry['label']} area={entry['area']:>6}px "
                    f"issue={entry['issue']:<15} action={entry['action']:<20} "
                    f"conf={entry['confidence']:.3f} {applied_str}{reason_str}"
                )
                if ts_str:
                    print(f"       scores: {ts_str}")

        print(f"  Dice       : {dice_before} -> {dice_after}  (delta={dice_delta:+.4f})")
        print(f"  IoU        : {iou_before}  -> {iou_after}   (delta={iou_delta:+.4f})")
        print(f"  Objects    : {imp['objects_before']} -> {imp['objects_after']}")
        print()

        # Save comparison figure
        fig_path = os.path.join(OUTPUT_DIR, f"{sid}_comparison.png")
        save_comparison(
            sample_id=sid,
            image_rgb=image_rgb,
            pred_mask=pred_mask,
            refined_mask=result["mask_after"],
            gt_mask=gt_mask,
            action=action,
            confidence=confidence,
            dice_before=dice_before,
            dice_after=dice_after,
            iou_before=iou_before,
            iou_after=iou_after,
            out_path=fig_path,
        )

        csv_rows.append({
            "sample_id":          sid,
            "issue_detected":     dec["issue"],
            "action_applied":     action,
            "confidence":         confidence,
            "rollback_happened":  1 if "rollback_note" in result else 0,
            "selected_checkpoint": selected_checkpoint,
            "selected_strategy":  result.get("selected_strategy", PIPELINE_MODE),
            "dice_before":        dice_before,
            "dice_after":         dice_after,
            "dice_delta":         dice_delta,
            "iou_before":         iou_before,
            "iou_after":          iou_after,
            "iou_delta":          iou_delta,
            "objects_before":     imp["objects_before"],
            "objects_after":      imp["objects_after"],
        })

    # Aggregate summary
    avg_dice_before = round(sum(r["dice_before"] for r in csv_rows) / len(csv_rows), 4)
    avg_dice_after  = round(sum(r["dice_after"]  for r in csv_rows) / len(csv_rows), 4)
    avg_iou_before  = round(sum(r["iou_before"]  for r in csv_rows) / len(csv_rows), 4)
    avg_iou_after   = round(sum(r["iou_after"]   for r in csv_rows) / len(csv_rows), 4)
    n_improved      = sum(1 for r in csv_rows if r["dice_delta"] > 0)
    n_unchanged     = sum(1 for r in csv_rows if r["dice_delta"] == 0)
    n_worsened      = sum(1 for r in csv_rows if r["dice_delta"] < 0)

    # Action distribution
    action_counts = {}
    for r in csv_rows:
        action = r["action_applied"]
        action_counts[action] = action_counts.get(action, 0) + 1

    print("=" * 65)
    print("  AGGREGATE RESULTS")
    print("=" * 65)
    print(f"  Samples evaluated : {len(csv_rows)}")
    print(f"  Samples improved  : {n_improved} / {len(csv_rows)}")
    print(f"  Samples unchanged : {n_unchanged} / {len(csv_rows)}")
    print(f"  Samples worsened  : {n_worsened} / {len(csv_rows)}")
    print()
    print(f"  Avg Dice  before  : {avg_dice_before}")
    print(f"  Avg Dice  after   : {avg_dice_after}  (delta={avg_dice_after-avg_dice_before:+.4f})")
    print(f"  Avg IoU   before  : {avg_iou_before}")
    print(f"  Avg IoU   after   : {avg_iou_after}  (delta={avg_iou_after-avg_iou_before:+.4f})")
    print()
    print("  Action distribution:")
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        pct = 100.0 * count / len(csv_rows)
        print(f"    {action:<30} {count:>2}  ({pct:>5.1f}%)")
    print()

    # Issue type statistics: show which issue types were detected and what actions they received
    issue_stats = {}
    for r in csv_rows:
        issue = r["issue_detected"]
        if issue not in issue_stats:
            issue_stats[issue] = {"count": 0, "actions": {}, "improved": 0}
        issue_stats[issue]["count"] += 1
        action = r["action_applied"]
        issue_stats[issue]["actions"][action] = issue_stats[issue]["actions"].get(action, 0) + 1
        if r["dice_delta"] > 0:
            issue_stats[issue]["improved"] += 1

    print("  Issue type summary (showing action diversity):")
    for issue in sorted(issue_stats.keys()):
        stats = issue_stats[issue]
        print(f"    {issue:<20} {stats['count']:>2} cases  ({stats['improved']} improved)")
        for action in sorted(stats["actions"].keys()):
            count = stats["actions"][action]
            print(f"      -> {action:<28} {count:>2}x")
    print()

    # ── Section 6: Explicit analysis ──────────────────────────────────────────
    n_total          = len(csv_rows)
    n_active         = sum(1 for r in csv_rows if r["action_applied"] not in ("no_action",))
    n_rollbacks      = sum(r["rollback_happened"] for r in csv_rows)
    n_meaningful     = sum(1 for r in csv_rows if r["dice_delta"] > 0.01)
    distinct_actions = set(r["action_applied"] for r in csv_rows)
    distinct_issues  = set(r["issue_detected"] for r in csv_rows)
    non_clean        = [r for r in csv_rows if r["issue_detected"] != "clean"]
    pct_active       = 100.0 * n_active / n_total if n_total else 0.0

    print("=" * 65)
    print("  ANALYSIS: AGENT UTILITY ON NEW MODEL PREDICTIONS")
    print("=" * 65)

    # --- Opportunity analysis -------------------------------------------------
    print()
    print("  [1] Agent opportunity")
    print(f"      The agent applied a corrective action on {n_active}/{n_total} samples "
          f"({pct_active:.0f}%).")
    if n_active == n_total:
        print("      -> Every sample triggered a tool. The new predictions are "
              "consistently imperfect — the agent has broad scope to help.")
    elif n_active == 0:
        print("      -> No samples triggered a tool (all clean or low-confidence). "
              "The new model may already be well-calibrated, or issues are subtle.")
    else:
        n_clean = sum(1 for r in csv_rows if r["issue_detected"] == "clean")
        print(f"      -> {n_clean} sample(s) were judged 'clean' (no action needed).")

    # --- Issue frequency breakdown -------------------------------------------
    print()
    print("  [2] Issue type frequency")
    for issue in sorted(issue_stats.keys(), key=lambda k: -issue_stats[k]["count"]):
        s = issue_stats[issue]
        pct = 100.0 * s["count"] / n_total
        bar = "#" * s["count"]
        print(f"      {issue:<20} {s['count']:>2}x  ({pct:>5.1f}%)  {bar}")

    # Characterise the dominant failure mode
    dominant_issue = max(issue_stats, key=lambda k: issue_stats[k]["count"])
    if dominant_issue == "clean":
        print("      -> Dominant outcome: clean. New predictions have few detectable artifacts.")
    elif dominant_issue == "noisy":
        print("      -> Dominant failure: noise/debris. New model produces many false-positive "
              "fragments — agent's remove_small_objects is the primary lever.")
    elif dominant_issue == "merged":
        print("      -> Dominant failure: merged glands. New model misses inter-gland "
              "boundaries — watershed_split is the critical corrective tool.")
    elif dominant_issue == "holey":
        print("      -> Dominant failure: internal holes. New model under-predicts lumen "
              "regions — fill_holes / morph_close are key.")
    elif dominant_issue == "under_segmented":
        print("      -> Dominant failure: under-segmentation. New model boundaries are too "
              "tight — dilation is the primary corrective action.")
    elif dominant_issue == "over_segmented":
        print("      -> Dominant failure: over-segmentation. New model over-fragments glands "
              "— morph_close is the primary corrective action.")
    elif dominant_issue == "thin_bridge":
        print("      -> Dominant failure: thin bridges. Glands are partially merged by narrow "
              "necks — morph_open is the targeted fix.")
    elif dominant_issue == "rough_boundary":
        print("      -> Dominant failure: rough boundaries. Gland shapes are correct but "
              "boundaries are jagged — morph_close smooths them.")

    # --- Action diversity ---------------------------------------------------
    print()
    print("  [3] Action diversity")
    print(f"      Distinct actions used  : {len(distinct_actions)}")
    print(f"      Distinct issue types   : {len(distinct_issues)}")
    if len(distinct_actions) >= 4:
        print("      -> HIGH diversity. The new predictions trigger a wide variety of "
              "corrections, showing the agent's full decision breadth is exercised.")
    elif len(distinct_actions) >= 2:
        print("      -> MODERATE diversity. The agent uses a subset of its tools; "
              "failure modes are somewhat consistent across samples.")
    else:
        print("      -> LOW diversity. A single failure mode dominates; "
              "action diversity is minimal.")

    # --- Meaningful improvement analysis -------------------------------------
    print()
    print("  [4] Meaningful improvement (Dice delta > 0.01)")
    print(f"      {n_meaningful}/{n_total} samples show a meaningful Dice gain.")
    if n_meaningful > 0:
        best = max(csv_rows, key=lambda r: r["dice_delta"])
        print(f"      Best case  : {best['sample_id']}  "
              f"Dice {best['dice_before']:.4f} -> {best['dice_after']:.4f}  "
              f"(+{best['dice_delta']:.4f})  action={best['action_applied']}")
    worst = min(csv_rows, key=lambda r: r["dice_delta"])
    if worst["dice_delta"] < 0:
        print(f"      Worst case : {worst['sample_id']}  "
              f"Dice {worst['dice_before']:.4f} -> {worst['dice_after']:.4f}  "
              f"({worst['dice_delta']:+.4f})  action={worst['action_applied']} "
              f"[rollback={'yes' if worst['rollback_happened'] else 'no'}]")
    if n_rollbacks:
        print(f"      Rollbacks  : {n_rollbacks} action(s) were reverted by the safety guard.")
    # Checkpoint distribution
    checkpoint_counts = {}
    for r in csv_rows:
        ck = r.get("selected_checkpoint", "original")
        checkpoint_counts[ck] = checkpoint_counts.get(ck, 0) + 1
    if any(v != "original" for v in checkpoint_counts):
        print(f"      Checkpoint : " + ", ".join(
            f"{ck}={cnt}" for ck, cnt in sorted(checkpoint_counts.items())
        ))

    # --- Overall verdict -----------------------------------------------------
    print()
    print("  [5] Overall verdict")
    dice_gain = avg_dice_after - avg_dice_before
    if dice_gain > 0.005:
        print(f"      The agent IMPROVED average Dice by {dice_gain:+.4f} on the new "
              "predictions. The weaker model leaves correctable artifacts that the "
              "agent successfully addresses.")
    elif dice_gain < -0.005:
        print(f"      The agent HURT average Dice by {dice_gain:+.4f}. The new predictions "
              "may have characteristics the agent's rules did not anticipate — "
              "review the worsened samples for feature mis-classification.")
    else:
        print(f"      The agent had NEUTRAL impact (Dice delta={dice_gain:+.4f}). "
              "Either the predictions are already near-optimal or the agent's "
              "confidence threshold blocked most interventions.")
    print("=" * 65)

    csv_path = os.path.join(OUTPUT_DIR, "report.csv")
    write_csv(csv_rows, csv_path)
    print(f"\n  CSV  saved -> {csv_path}")
    print(f"  PNGs saved -> {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
