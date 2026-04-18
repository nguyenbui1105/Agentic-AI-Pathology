"""
Region-aware post-processing pipeline.

Two-tier decision flow
----------------------
Tier 1 — Global:
    Classify the whole mask.  Handle issues that require global context:
    - noisy          → remove_small_objects (debris is only meaningful globally)
    - over_segmented → morph_close (closing many similar fragments at once)
    If the global action is accepted, pass the cleaned mask to Tier 2.
    If rejected, proceed to Tier 2 on the original mask (debris stays; the
    region tier focuses on real glands and ignores tiny objects).

Tier 2 — Region:
    For each significant connected component (area >= _SMALL_REGION_PX):
    1. Extract per-region features (compactness, roughness, holes, …).
    2. Agent classifies the region's issue and selects a tool.
    3. Apply the tool locally: operate on the isolated component, then
       composite the result back into the full mask.
    4. Validate the global mask after each local modification; reject if
       proxy signals indicate a regression.

Overlap handling
----------------
When an expanding tool (fill_holes, morph_close, dilation) is applied to
one region, its output is clipped so it cannot overwrite pixels that belong
to other connected components in the current labeled mask.

Checkpoint
----------
The final mask is the working_mask after all accepted region modifications.
If every region-level step is rejected, the result equals the mask coming
out of the global tier (or the original if the global tier also failed).
"""

from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label as sk_label, regionprops as sk_rprops

from src.features.extractor import extract_features
from src.agent.reasoning import PathologyReasoningAgent
from src.agent.region_features import extract_region_features, _SMALL_REGION_PX
from src.tools.postprocessing import (
    remove_small_objects, fill_holes, morph_close,
    morph_open, erosion, dilation, watershed_split, connect_fragments,
)
from src.pipeline import (
    _DEFAULT_AGENT, _step_validate, _adaptive_params,
    _apply_adaptive, _evaluate, _summarise, reset_agent_session,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# Issue types that require global mask context; skip at region level.
# connect_fragments is inherently global: it merges multiple components.
_GLOBAL_ONLY_ISSUES = frozenset({"noisy", "over_segmented", "fragmented"})

# ── Helpers ───────────────────────────────────────────────────────────────────

def _region_adaptive_params(action: str, rf: dict) -> dict:
    """
    Adaptive parameters for a tool applied to a single region.

    Mirrors pipeline._adaptive_params() but tuned for single-object context:
    smaller kernels (no need to bridge across multiple objects) and
    hole-size computation relative to the single region's area.
    """
    area      = rf.get("avg_object_size", 1000)
    roughness = rf.get("boundary_roughness", 1.0)
    density   = rf.get("mask_density", 0.5)
    h_size    = rf.get("avg_hole_size", 0.0)

    if action == "fill_holes":
        return {"max_hole_size": max(50, min(int(area * 0.15), 2000))}

    elif action == "morph_close":
        if h_size > 0:
            k_hole = max(3, 2 * int(math.sqrt(h_size / math.pi) * 0.4) + 1)
        else:
            k_hole = 3
        excess  = max(0.0, roughness - 1.0)
        k_rough = max(3, 2 * int(excess * 5) + 1)
        kernel  = min(max(k_hole, k_rough), 13)
        if kernel % 2 == 0:
            kernel += 1
        return {"kernel_size": kernel, "iterations": 1}

    elif action == "morph_open":
        excess = max(0.0, roughness - 1.0)
        kernel = min(max(3, 2 * int(excess * 5) + 1), 9)
        if kernel % 2 == 0:
            kernel += 1
        return {"kernel_size": kernel, "iterations": 1}

    elif action == "watershed_split":
        radius = math.sqrt(area / math.pi)
        return {"min_distance": max(8, min(int(radius * 0.28), 40))}

    elif action == "dilation":
        sparseness = max(0.0, 0.05 - density) / 0.05
        kernel = 3 if sparseness < 0.3 else (5 if sparseness < 0.6 else 7)
        return {"kernel_size": kernel, "iterations": 1}

    elif action == "erosion":
        return {"kernel_size": 3, "iterations": 1}

    return {}


def _apply_locally(
    working_mask:  np.ndarray,
    full_labeled:  np.ndarray,
    region_label:  int,
    action:        str,
    rf:            dict,
) -> np.ndarray:
    """
    Apply a tool to one connected component and composite back.

    Parameters
    ----------
    working_mask  : current full mask (uint8, 0/255)
    full_labeled  : labeled version of working_mask (output of sk_label)
    region_label  : label value for the target region
    action        : tool name
    rf            : per-region features dict

    Returns
    -------
    np.ndarray — updated full mask (same shape/dtype).  If the tool produces
    an empty result for the region the original mask is returned unchanged.
    """
    # Isolate the region as a standalone binary mask.
    region_bool = full_labeled == region_label
    region_mask = region_bool.astype(np.uint8) * 255

    # Apply the tool.
    params = _region_adaptive_params(action, rf)
    if action == "fill_holes":
        result = fill_holes(region_mask, **params)
    elif action == "morph_close":
        result = morph_close(region_mask, **params)
    elif action == "morph_open":
        result = morph_open(region_mask, **params)
    elif action == "erosion":
        result = erosion(region_mask, **params)
    elif action == "dilation":
        result = dilation(region_mask, **params)
    elif action == "watershed_split":
        result = watershed_split(region_mask, **params)
    else:
        return working_mask.copy()

    # Safety: if tool wiped the region completely, skip.
    if not np.any(result > 0):
        return working_mask.copy()

    # Pixels belonging to OTHER components (must not be overwritten).
    other_objects = (full_labeled > 0) & ~region_bool

    # Build updated mask:
    #   • remove original region pixels
    #   • add tool output, clipped away from other components
    updated = working_mask.copy()
    updated[region_bool] = 0
    expansion = (result > 0) & ~other_objects
    updated[expansion] = 255

    return updated


def _decide_regions(
    mask:               np.ndarray,
    agent:              PathologyReasoningAgent,
    confidence_thresh:  float,
) -> list[dict]:
    """
    Classify each significant region and return per-region decision dicts.

    Only issues that make sense at the single-object level are kept.
    Global-only issues (noisy, over_segmented) are forced to no_action.
    """
    region_features_list = extract_region_features(mask)
    decisions: list[dict] = []

    for rf in region_features_list:
        d     = agent.decide(rf)
        issue = d["issue"]
        conf  = d["confidence"]

        # Suppress issues that only make sense with multiple objects.
        if issue in _GLOBAL_ONLY_ISSUES:
            issue  = "clean"
            action = "no_action"
        else:
            # At region level, always select the single best-scoring candidate.
            # Multi-step sequences are not applied per-region (they require
            # global-scope coordination between independent objects).
            ts = d["tool_scores"]
            action = max(ts, key=ts.__getitem__) if ts else "no_action"

            # If every candidate tool scored 0 (all gates suppressed), force no_action.
            # This handles cases like lumen safety where both fill_holes and morph_close
            # are suppressed — we should not attempt to apply a zero-scoring tool.
            if action != "no_action" and ts.get(action, 0) <= 0.0:
                action = "no_action"

        apply = (action != "no_action") and (conf >= confidence_thresh)

        decisions.append({
            "label":       rf["_label"],
            "area":        rf["_area"],
            "centroid":    rf["_centroid"],
            "issue":       issue,
            "action":      action,
            "confidence":  round(conf, 4),
            "tool_scores": d["tool_scores"],
            "score_breakdown": d.get("score_breakdown", {}),
            "apply":       apply,
            "features":    rf,
        })

    # Largest regions first — most important glands get priority.
    decisions.sort(key=lambda x: -x["area"])
    return decisions


# ── Main entry point ──────────────────────────────────────────────────────────

def run_region_pipeline(
    mask:                np.ndarray,
    gt_mask:             np.ndarray | None = None,
    agent:               PathologyReasoningAgent | None = None,
    confidence_threshold: float = 0.30,
) -> dict:
    """
    Region-aware post-processing pipeline.

    Parameters
    ----------
    mask                 : Input binary mask (H, W), uint8, foreground = 255.
    gt_mask              : Optional ground-truth for Dice/IoU evaluation.
    agent                : Agent instance; uses module-level singleton if None.
    confidence_threshold : Per-region actions require confidence >= this.

    Returns
    -------
    dict — same keys as run_postprocessing_pipeline(), plus:
        n_regions        : int   — significant regions analysed
        regions_modified : int   — regions where a tool was applied
        region_log       : list  — per-region issue / action / outcome
        global_log       : list  — global-tier step log (may be empty)
    """
    if agent is None:
        agent = _DEFAULT_AGENT

    # ── Tier 1: global classification and noise/fragmentation handling ─────────
    global_features = extract_features(mask)
    global_decision = agent.decide(global_features)
    global_issue    = global_decision["issue"]

    working_mask = mask.copy()
    global_log:   list[dict] = []
    params_used:  dict       = {}
    global_applied_action    = "no_action"

    if global_issue in _GLOBAL_ONLY_ISSUES and \
            global_decision["confidence"] >= confidence_threshold:

        g_action   = global_decision["action_sequence"][0]
        g_params   = _adaptive_params(g_action, global_features)
        g_result   = _apply_adaptive(working_mask, g_action, global_features)
        accept, signals, reason = _step_validate(g_action, working_mask, g_result)

        global_log.append({
            "step":     "global",
            "action":   g_action,
            "accepted": accept,
            "signals":  signals,
            "reason":   reason,
            "params":   g_params,
        })

        if accept:
            working_mask          = g_result
            global_applied_action = g_action
            params_used[g_action] = g_params
            agent.record_action(g_action)
        # If rejected: keep working_mask as-is; continue to region tier.

    # ── Tier 2: region-level analysis and local tool application ───────────────
    region_decisions = _decide_regions(working_mask, agent, confidence_threshold)
    region_log:   list[dict] = []
    regions_modified = 0

    # Labeled mask refreshed after any global modification.
    full_labeled = sk_label(working_mask > 0)

    for dec in region_decisions:
        orig_label = dec["label"]
        action     = dec["action"]
        conf       = dec["confidence"]
        rf         = dec["features"]

        log_entry: dict = {
            "label":           orig_label,
            "area":            dec["area"],
            "centroid":        dec["centroid"],
            "issue":           dec["issue"],
            "action":          action,
            "confidence":      conf,
            "tool_scores":     dec["tool_scores"],
            "score_breakdown": dec["score_breakdown"],
            "applied":         False,
            "signals":         {},
            "reason":          None,
        }

        if not dec["apply"]:
            log_entry["reason"] = (
                f"conf={conf:.3f} < threshold={confidence_threshold}"
                if conf < confidence_threshold else "clean — no action needed"
            )
            region_log.append(log_entry)
            continue

        # Guard: region may have been merged away by a prior local step.
        if not np.any(full_labeled == orig_label):
            log_entry["reason"] = "region no longer present after prior modification"
            region_log.append(log_entry)
            continue

        # Apply the tool locally.
        prev_mask = working_mask.copy()
        try:
            updated = _apply_locally(
                working_mask, full_labeled, orig_label, action, rf
            )
        except Exception as exc:
            log_entry["reason"] = f"tool raised exception: {exc}"
            region_log.append(log_entry)
            continue

        # Validate the full-mask result (global proxy signals).
        accept, signals, reason = _step_validate(action, prev_mask, updated)
        log_entry["signals"] = signals

        if accept:
            working_mask = updated
            # Refresh labels so subsequent regions see the updated layout.
            full_labeled = sk_label(working_mask > 0)
            log_entry["applied"] = True
            agent.record_action(action)
            regions_modified += 1
        else:
            log_entry["reason"] = f"step rejected: {reason}"

        region_log.append(log_entry)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics_before = _evaluate(mask, gt_mask)
    metrics_after  = _evaluate(working_mask, gt_mask)
    improvement    = _summarise(metrics_before, metrics_after, gt_mask is not None)

    # ── Build action string for compatibility ─────────────────────────────────
    applied_parts: list[str] = []
    if global_applied_action != "no_action":
        applied_parts.append(f"[global] {global_applied_action}")
    for entry in region_log:
        if entry["applied"]:
            applied_parts.append(f"[r{entry['label']}:{entry['issue']}] {entry['action']}")
    action_applied = " | ".join(applied_parts) if applied_parts else "no_action"

    checkpoint = "region_result" if (regions_modified > 0 or global_applied_action != "no_action") else "original"

    return {
        # ── Compatibility with evaluate_glas.py ──────────────────────────
        "features":             global_features,
        "decision":             global_decision,
        "action_applied":       action_applied,
        "action_sequence":      global_decision.get("action_sequence", ["no_action"]),
        "params_used":          params_used,
        "confidence":           global_decision["confidence"],
        "confidence_threshold": confidence_threshold,
        "selected_checkpoint":  checkpoint,
        "step_log":             global_log,
        "mask_before":          mask,
        "mask_after":           working_mask,
        "metrics_before":       metrics_before,
        "metrics_after":        metrics_after,
        "improvement_summary":  improvement,
        # ── Region-specific additions ─────────────────────────────────────
        "n_regions":            len(region_decisions),
        "regions_modified":     regions_modified,
        "region_log":           region_log,
        "global_log":           global_log,
    }
