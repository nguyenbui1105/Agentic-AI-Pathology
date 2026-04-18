"""
End-to-end post-processing pipeline with adaptive tool parameters,
multi-step sequential correction, and step-wise safety validation.

Flow:
    mask
      -> extract_features()
      -> PathologyReasoningAgent.decide()       # returns action_sequence (1-2 steps)
      -> confidence gate (skip if below threshold)
      -> for each step in sequence:
           apply tool with adaptive params
           validate result via proxy signals
           if rejected  → stop, keep last accepted checkpoint
           if accepted  → advance to next step
      -> select best checkpoint (last accepted)
      -> evaluate before / after
      -> return structured report

Step-wise validation
--------------------
After each tool application the result is scored using three cheap proxy
signals computed WITHOUT ground truth:

  area_ratio   = new_area  / prev_area
  obj_ratio    = new_count / prev_count
  compactness  = mean(4π·area / perimeter²) over all objects

Rejection rules (conservative):
  Universal:
    area_ratio < 0.70              → catastrophic area loss
    compactness < 0.05             → degenerate / fragmented mask

  remove_small_objects:
    obj_ratio < adaptive_threshold → too many real glands removed
    (threshold = 0.20 for n≤5, 0.25 for n≤10, 0.30 for n>10)

  morph_close / fill_holes / dilation  (expanding ops):
    area_ratio < 0.95              → unexpectedly shrank (indicates error)

  morph_open:
    area_ratio < 0.60              → removed too much boundary

  watershed_split:
    obj_ratio < 0.80               → objects collapsed instead of splitting

Checkpoint selection
--------------------
Checkpoints: mask0 (original), mask1 (after step1), mask2 (after step2).
The last accepted checkpoint is selected.

  step2 rejected → use mask1
  step1 rejected → use mask0 (original)

The report includes step_log (per-step decision + signals) and
selected_checkpoint ("original" / "step1" / "step2") for full auditability.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.measure import label as sk_label, regionprops as sk_rprops

from src.features.extractor import extract_features
from src.agent.reasoning import PathologyReasoningAgent, _DENSITY_UNDERSEG
from src.tools.postprocessing import (
    remove_small_objects, morph_close, watershed_split,
    morph_open, erosion, dilation, fill_holes, connect_fragments,
)
from src.evaluation.metrics import compute_dice, compute_iou, basic_stats


# ── Step validation thresholds ────────────────────────────────────────────────

_STEP_REJECT_MIN_AREA        = 0.70   # universal: max 30% area loss per step
_STEP_REJECT_EXPAND_MIN_AREA = 0.95   # morph_close / fill_holes / dilation must not shrink
_STEP_REJECT_OPEN_MIN_AREA   = 0.60   # morph_open: allow up to 40% removal
_STEP_REJECT_WATERSHED_COUNT = 0.80   # watershed must not reduce object count below 80%
_STEP_REJECT_MIN_COMPACT     = 0.05   # degenerate mask guard

# remove_small_objects adaptive minimum retained object fraction
# Tighter threshold for high object counts (more objects = more likely real glands exist)
def _rso_min_obj_ratio(n_before: int) -> float:
    if n_before <= 5:
        return 0.20
    if n_before <= 10:
        return 0.25
    if n_before <= 15:
        return 0.30
    return 0.35   # n>15: many objects present — keep at least 35% to avoid over-pruning


_DEFAULT_AGENT = PathologyReasoningAgent()


def reset_agent_session() -> None:
    """Reset the default agent's session-level diversity counters."""
    _DEFAULT_AGENT.reset_session()


# ── Proxy signal computation ──────────────────────────────────────────────────

def _quick_compactness(mask: np.ndarray) -> float:
    """
    Fast mean compactness (4π·area / perimeter²) over all objects.
    Returns 0.0 for an empty mask.
    """
    labeled = sk_label(mask > 0)
    props   = sk_rprops(labeled)
    vals = [
        4.0 * math.pi * p.area / (p.perimeter ** 2)
        for p in props
        if p.area > 0 and p.perimeter > 0
    ]
    return float(np.mean(vals)) if vals else 0.0


def _step_validate(
    action: str,
    mask_prev: np.ndarray,
    mask_curr: np.ndarray,
) -> tuple[bool, dict, str | None]:
    """
    Validate a single pipeline step using cheap proxy signals (no GT needed).

    Parameters
    ----------
    action    : name of the tool just applied
    mask_prev : mask before this step (previous checkpoint)
    mask_curr : mask produced by this step

    Returns
    -------
    accept  : True → keep mask_curr, False → reject and revert to mask_prev
    signals : dict of computed proxy values (for logging)
    reason  : human-readable rejection reason, or None if accepted
    """
    area_prev  = float((mask_prev > 0).sum())
    area_curr  = float((mask_curr > 0).sum())
    n_prev     = int(sk_label(mask_prev > 0).max())
    n_curr     = int(sk_label(mask_curr > 0).max())
    area_ratio = area_curr / max(area_prev, 1.0)
    obj_ratio  = n_curr   / max(n_prev, 1)
    compact    = _quick_compactness(mask_curr)

    signals = {
        "area_ratio":  round(area_ratio, 4),
        "n_prev":      n_prev,
        "n_curr":      n_curr,
        "obj_ratio":   round(obj_ratio, 4),
        "compactness": round(compact, 4),
    }

    # ── Universal checks ──────────────────────────────────────────────────────

    if area_ratio < _STEP_REJECT_MIN_AREA:
        return False, signals, (
            f"area dropped to {area_ratio:.1%} of previous step "
            f"(limit {_STEP_REJECT_MIN_AREA:.0%})"
        )

    if compact > 0 and compact < _STEP_REJECT_MIN_COMPACT:
        return False, signals, (
            f"compactness={compact:.4f} is unrealistically low — mask likely degenerate"
        )

    # ── Action-specific checks ────────────────────────────────────────────────

    if action == "remove_small_objects":
        if n_prev > 1:
            min_ratio = _rso_min_obj_ratio(n_prev)
            if obj_ratio < min_ratio:
                return False, signals, (
                    f"objects collapsed {n_prev} -> {n_curr} "
                    f"({obj_ratio:.1%} retained, limit {min_ratio:.0%})"
                )

    elif action == "connect_fragments":
        # Area can only increase (bridge pixels added); object count decreases
        # intentionally as fragments merge — no obj_ratio check needed here.
        if area_ratio < _STEP_REJECT_EXPAND_MIN_AREA:
            return False, signals, (
                f"connect_fragments unexpectedly shrank area to {area_ratio:.1%} "
                f"(should be >= {_STEP_REJECT_EXPAND_MIN_AREA:.0%})"
            )

    elif action in ("morph_close", "fill_holes", "dilation"):
        if area_ratio < _STEP_REJECT_EXPAND_MIN_AREA:
            return False, signals, (
                f"{action} unexpectedly shrank area to {area_ratio:.1%} "
                f"(should be >= {_STEP_REJECT_EXPAND_MIN_AREA:.0%})"
            )

    elif action == "morph_open":
        if area_ratio < _STEP_REJECT_OPEN_MIN_AREA:
            return False, signals, (
                f"morph_open removed {1 - area_ratio:.1%} of area "
                f"(limit {1 - _STEP_REJECT_OPEN_MIN_AREA:.0%} removable)"
            )

    elif action == "watershed_split":
        if n_prev > 0 and obj_ratio < _STEP_REJECT_WATERSHED_COUNT:
            return False, signals, (
                f"watershed_split reduced objects {n_prev} -> {n_curr} "
                f"({obj_ratio:.1%} remaining, expected >= "
                f"{_STEP_REJECT_WATERSHED_COUNT:.0%})"
            )

    return True, signals, None


# ── Adaptive parameter computation ────────────────────────────────────────────

def _adaptive_params(action: str, features: dict) -> dict:
    """
    Compute feature-driven parameters for each tool.
    All formulas are deterministic given the feature dict.

    morph_close
        kernel = max(hole-driven, roughness-driven, frag-driven), capped at 19, odd.

    remove_small_objects
        Bimodal-aware: blends conservative (10% avg) and aggressive (12% max)
        weighted by size_cv/2. Capped at 2000 px.

    watershed_split
        min_distance = 28% of approximate gland radius. Capped at 40 px.

    fill_holes
        max_hole_size = 15% of avg_object_size (below 20% lumen threshold).

    dilation
        kernel 3/5/7 based on sparseness level.

    morph_open
        kernel adapts to boundary roughness excess, capped at 11, odd.

    erosion   Always conservative (kernel=3).
    """
    if action == "remove_small_objects":
        max_size = features.get("max_object_size", 500)
        avg_size = features.get("avg_object_size", 500)
        size_cv  = features.get("size_cv", 0.0)
        cv_w         = min(1.0, size_cv / 2.0)
        conservative = int(avg_size * 0.10)
        aggressive   = int(max_size * 0.12)
        threshold    = int(conservative + cv_w * (aggressive - conservative))
        return {"min_size": max(50, min(threshold, 2000))}

    elif action == "morph_close":
        h_size    = features.get("avg_hole_size", 0.0)
        roughness = features.get("boundary_roughness", 1.0)
        n_obj     = features.get("num_objects", 1)
        if h_size > 0:
            hole_r = math.sqrt(h_size / math.pi)
            k_hole = max(5, 2 * int(hole_r * 0.4) + 1)
        else:
            k_hole = 5
        excess  = max(0.0, roughness - 1.0)
        k_rough = max(5, 2 * int(excess * 5) + 1)
        frag    = min(1.0, max(0.0, (n_obj - 2) / 12.0))
        k_frag  = max(5, 2 * int(frag * 6) + 1)
        kernel  = min(max(k_hole, k_rough, k_frag), 19)
        if kernel % 2 == 0:
            kernel += 1
        return {"kernel_size": kernel, "iterations": 1}

    elif action == "watershed_split":
        avg_size = features.get("avg_object_size", 1000)
        radius   = math.sqrt(avg_size / math.pi)
        return {"min_distance": max(10, min(int(radius * 0.28), 40))}

    elif action == "fill_holes":
        avg_size = features.get("avg_object_size", 1000)
        return {"max_hole_size": max(100, min(int(avg_size * 0.15), 2000))}

    elif action == "morph_open":
        roughness = features.get("boundary_roughness", 1.0)
        excess    = max(0.0, roughness - 1.0)
        kernel    = min(max(3, 2 * int(excess * 5) + 1), 11)
        if kernel % 2 == 0:
            kernel += 1
        return {"kernel_size": kernel, "iterations": 1}

    elif action == "dilation":
        density    = features.get("mask_density", 0.01)
        sparseness = max(0.0, _DENSITY_UNDERSEG - density) / _DENSITY_UNDERSEG
        kernel     = 3 if sparseness < 0.3 else (5 if sparseness < 0.6 else 7)
        return {"kernel_size": kernel, "iterations": 1}

    elif action == "erosion":
        return {"kernel_size": 3, "iterations": 1}

    elif action == "connect_fragments":
        avg_size  = features.get("avg_object_size", 1000)
        near_frag = features.get("nearby_fragment_ratio", 0.5)
        # Gap scales with sqrt(gland size): larger glands tolerate wider fragment gaps.
        max_gap = max(15, min(int(math.sqrt(avg_size) * 0.45), 40))
        # Stricter size ratio when proximity is low (fewer clear fragments present).
        size_ratio = 0.12 if near_frag >= 0.7 else 0.08
        return {
            "max_gap_px":     max_gap,
            "size_ratio_max": size_ratio,
            "min_main_size":  max(300, int(avg_size * 0.5)),
        }

    return {}


def _apply_adaptive(mask: np.ndarray, action: str, features: dict) -> np.ndarray:
    """Dispatch action → tool with feature-adaptive parameters."""
    params = _adaptive_params(action, features)
    if action == "remove_small_objects":
        return remove_small_objects(mask, **params)
    elif action == "fill_holes":
        return fill_holes(mask, **params)
    elif action == "morph_close":
        return morph_close(mask, **params)
    elif action == "morph_open":
        return morph_open(mask, **params)
    elif action == "erosion":
        return erosion(mask, **params)
    elif action == "dilation":
        return dilation(mask, **params)
    elif action == "watershed_split":
        return watershed_split(mask, **params)
    elif action == "connect_fragments":
        return connect_fragments(mask, **params)
    return mask.copy()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_postprocessing_pipeline(
    mask: np.ndarray,
    gt_mask: np.ndarray | None = None,
    agent: PathologyReasoningAgent | None = None,
    confidence_threshold: float = 0.30,
) -> dict:
    """
    Run the full agent-guided post-processing pipeline on a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Input segmentation mask (H, W), uint8, foreground = 255 or 1.
    gt_mask : np.ndarray or None
        Ground truth mask for evaluation.
    agent : PathologyReasoningAgent or None
        Agent instance. Uses the default singleton if None.
    confidence_threshold : float
        Steps are skipped if agent confidence < this value.

    Returns
    -------
    dict with keys:
        features, decision, action_applied, action_sequence, params_used,
        confidence, confidence_threshold, selected_checkpoint, step_log,
        mask_before, mask_after, metrics_before, metrics_after,
        improvement_summary
        (+ optional: safety_note, rollback_note)
    """
    if agent is None:
        agent = _DEFAULT_AGENT

    # ── Stage 1: feature extraction ───────────────────────────────────────────
    features = extract_features(mask)

    # ── Stage 2: agent decision ───────────────────────────────────────────────
    decision        = agent.decide(features)
    action_sequence = decision.get("action_sequence", [decision["selected_action"]])
    confidence      = decision["confidence"]

    # ── Stage 3: confidence gate ──────────────────────────────────────────────
    safety_note = None
    if confidence < confidence_threshold and action_sequence != ["no_action"]:
        skipped         = " -> ".join(action_sequence)
        action_sequence = ["no_action"]
        safety_note     = (
            f"Confidence {confidence:.3f} < threshold {confidence_threshold:.2f}. "
            f"Skipped '{skipped}', applied no_action instead."
        )

    # ── Stage 4: step-wise application with checkpoint management ─────────────
    #
    # checkpoints: list of (label, mask_array)
    #   "original" → mask before any processing
    #   "step1"    → mask after step 1 (if accepted)
    #   "step2"    → mask after step 2 (if accepted)
    #
    # Execution stops at the first rejected step.
    # The report always contains the LAST ACCEPTED checkpoint.

    checkpoints: list[tuple[str, np.ndarray]] = [("original", mask.copy())]
    params_used: dict[str, dict]              = {}
    step_log:    list[dict]                   = []
    rollback_note: str | None                 = None

    for i, step in enumerate(action_sequence):
        step_label = f"step{i + 1}"

        if step == "no_action":
            step_log.append({
                "step": i + 1, "action": "no_action",
                "accepted": True, "signals": {}, "reason": None, "params": {},
            })
            # No further processing needed; checkpoint stays as-is
            break

        step_params       = _adaptive_params(step, features)
        params_used[step] = step_params
        mask_prev         = checkpoints[-1][1]
        mask_next         = _apply_adaptive(mask_prev, step, features)

        accept, signals, reason = _step_validate(step, mask_prev, mask_next)

        step_log.append({
            "step":     i + 1,
            "action":   step,
            "accepted": accept,
            "signals":  signals,
            "reason":   reason,
            "params":   step_params,
        })

        if accept:
            checkpoints.append((step_label, mask_next))
            agent.record_action(step)   # update session diversity tracker
        else:
            # Rejected: stop sequence, use the previous accepted checkpoint
            rollback_note = (
                f"Step {i + 1} ({step}) rejected — {reason}. "
                f"Reverting to {checkpoints[-1][0]}."
            )
            break

    # ── Determine selected checkpoint and build action string ─────────────────
    selected_label, mask_after = checkpoints[-1]

    accepted_actions = [
        s["action"] for s in step_log
        if s["accepted"] and s["action"] != "no_action"
    ]
    rejected_steps = [s for s in step_log if not s["accepted"]]

    if action_sequence == ["no_action"] or (not accepted_actions and not rejected_steps):
        action_applied = "no_action"
    elif not accepted_actions:
        # Every attempted step was rejected → fell back to original
        rej = rejected_steps[0]
        action_applied = f"no_action (step {rej['step']} '{rej['action']}' rejected)"
    else:
        action_applied = " -> ".join(accepted_actions)
        if rejected_steps:
            # Some steps accepted, then a later step rejected
            rej = rejected_steps[0]
            action_applied += f" [step {rej['step']} '{rej['action']}' rejected]"

    # ── Stage 5: evaluate before / after ─────────────────────────────────────
    metrics_before = _evaluate(mask, gt_mask)
    metrics_after  = _evaluate(mask_after, gt_mask)
    improvement    = _summarise(metrics_before, metrics_after, gt_mask is not None)

    report = {
        "features":             features,
        "decision":             decision,
        "action_applied":       action_applied,
        "action_sequence":      action_sequence,
        "params_used":          params_used,
        "confidence":           confidence,
        "confidence_threshold": confidence_threshold,
        "selected_checkpoint":  selected_label,
        "step_log":             step_log,
        "mask_before":          mask,
        "mask_after":           mask_after,
        "metrics_before":       metrics_before,
        "metrics_after":        metrics_after,
        "improvement_summary":  improvement,
    }
    if safety_note:
        report["safety_note"]  = safety_note
    if rollback_note:
        report["rollback_note"] = rollback_note

    return report


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _evaluate(mask: np.ndarray, gt: np.ndarray | None) -> dict:
    stats = basic_stats(mask)
    if gt is not None:
        stats["dice"] = compute_dice(mask, gt)
        stats["iou"]  = compute_iou(mask, gt)
    return stats


def _summarise(before: dict, after: dict, has_gt: bool) -> dict:
    summary: dict = {}
    if has_gt:
        dice_delta = round(after["dice"] - before["dice"], 4)
        iou_delta  = round(after["iou"]  - before["iou"],  4)
        summary["dice_before"] = before["dice"]
        summary["dice_after"]  = after["dice"]
        summary["dice_delta"]  = dice_delta
        summary["iou_before"]  = before["iou"]
        summary["iou_after"]   = after["iou"]
        summary["iou_delta"]   = iou_delta
        summary["improved"]    = dice_delta > 0
    summary["objects_before"] = before["num_objects"]
    summary["objects_after"]  = after["num_objects"]
    summary["area_before"]    = before["total_area"]
    summary["area_after"]     = after["total_area"]
    return summary
