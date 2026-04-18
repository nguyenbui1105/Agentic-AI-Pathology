"""
Hybrid post-processing pipeline.

Combines global and region-aware reasoning to get the best of both:
  • Global pipeline:  strong corrections when a single issue dominates all glands
  • Region pipeline:  precise per-component fixes for spatially heterogeneous errors
  • Hybrid selector:  compares original / global / region results and picks the best

Decision flow
-------------
1. Measure issue spread: what fraction of significant regions share the same
   dominant issue.  A high spread means global treatment is appropriate; a low
   spread (mixed issues) favours per-region precision.

2. Apply routing rules (evaluated in priority order):
   a. noisy        — debris is a global property; always prefer global RSO
   b. over_segmented widespread (>= _WIDESPREAD_THRESH) — global morph_close
   c. holey widespread (>= _WIDESPREAD_THRESH)          — global morph_close
   d. mixed issues (spread < _MIXED_ISSUE_THRESH)        — prefer region pipeline
   e. fallback                                           — score both and pick winner

3. Accept / reject each candidate with a risk-aware three-tier logic:
   Tier 1 — Hard veto:   area < 50 % of original  OR  object count collapses → reject.
   Tier 2 — Dice-first:  if Dice gain ≥ _DICE_GAIN_THRESHOLD (+0.002) → accept.
   Tier 3 — Score gate:  proxy score within _SCORE_TOLERANCE of original → accept.

   Proxy score = f(area_score, compactness_delta, dice_delta):
     area_score is tolerant of ±25 % area change (normal for RSO / fill_holes) and
     only penalises large deviations — preventing the old over-penalisation of RSO.

4. Return the best-accepted candidate with `selected_strategy` and `strategy_reason`.
   If no candidate passes acceptance, the original mask is returned unchanged.

Risk-aware safety rules
-----------------------
• Hard veto catches catastrophic outcomes (per-step validation already ran inside
  both sub-pipelines; this is a final sanity check on the net result).
• Dice-first acceptance means a meaningful improvement is never blocked by minor
  proxy noise (area ±5 %, compactness ±3 %) — a key benefit when GT is available.
• Score-gate acceptance allows structural improvements that shift area/compactness
  slightly but stay within the normal operating envelope of the tools.
• If GT is not available, only score-gate and hard-veto are active.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.measure import label as sk_label

from src.agent.region_features import extract_region_features
from src.pipeline import (
    run_postprocessing_pipeline, _evaluate, _summarise,
    _DEFAULT_AGENT, reset_agent_session, _quick_compactness,
)
from src.region_pipeline import run_region_pipeline

# ── Routing thresholds ────────────────────────────────────────────────────────

_WIDESPREAD_THRESH  = 0.50   # dominant issue in >= 50 % of regions → "widespread"
_MIXED_ISSUE_THRESH = 0.50   # dominant issue in <  50 % of regions → "mixed"
_FORCE_GLOBAL_ISSUES = frozenset({"noisy"})

# ── Proxy score weights ───────────────────────────────────────────────────────

_W_AREA    = 0.20   # area score   (tolerant range, soft penalty outside it)
_W_COMPACT = 0.20   # compactness improvement
_W_DICE    = 0.60   # Dice delta   (primary signal when GT available)

# ── Risk-aware acceptance thresholds ─────────────────────────────────────────

# Tier 1 — Hard veto (catastrophic outcome)
_HARD_MIN_AREA_RATIO  = 0.50   # final mask retains at least 50 % of original area
_HARD_MIN_OBJ_RATIO   = 0.10   # final mask retains at least 10 % of original objects

# Tier 2 — Dice-first acceptance
_DICE_GAIN_THRESHOLD  = 0.002  # Dice must improve by at least +0.002 to accept via Tier 2

# Tier 3 — Score-gate acceptance
_SCORE_TOLERANCE      = 0.04   # candidate score within 4 % of original score → accepted

# Area tolerance band: within this ratio range score is 1.0 (no penalty).
# Covers normal RSO (−20 %), fill_holes (+10 %), morph_close (+5 %).
_AREA_SCORE_LOWER = 0.70   # below this → linear penalty
_AREA_SCORE_UPPER = 1.30   # above this → linear penalty


# ── Issue spread analysis ─────────────────────────────────────────────────────

def _measure_issue_spread(mask: np.ndarray, agent) -> dict:
    """
    Classify each significant region's issue and compute spread statistics.

    Returns
    -------
    dict with:
        dominant_issue  : str   — most common issue across regions
        dominant_frac   : float — fraction of significant regions with dominant issue
        issue_counts    : dict  — {issue: count}
        n_significant   : int   — number of regions analysed
        is_widespread   : bool  — dominant_frac >= _WIDESPREAD_THRESH
        is_mixed        : bool  — dominant_frac <  _MIXED_ISSUE_THRESH
    """
    region_features = extract_region_features(mask)
    n_sig = len(region_features)

    if n_sig == 0:
        return {
            "dominant_issue": "clean",
            "dominant_frac":  1.0,
            "issue_counts":   {"clean": 0},
            "n_significant":  0,
            "is_widespread":  False,
            "is_mixed":       False,
        }

    issue_counts: dict[str, int] = {}
    for rf in region_features:
        d     = agent.decide(rf)
        issue = d["issue"]
        if issue in ("noisy", "over_segmented"):
            issue = "clean"
        issue_counts[issue] = issue_counts.get(issue, 0) + 1

    dominant_issue = max(issue_counts, key=issue_counts.__getitem__)
    dominant_frac  = issue_counts[dominant_issue] / n_sig

    return {
        "dominant_issue": dominant_issue,
        "dominant_frac":  round(dominant_frac, 4),
        "issue_counts":   issue_counts,
        "n_significant":  n_sig,
        "is_widespread":  dominant_frac >= _WIDESPREAD_THRESH,
        "is_mixed":       dominant_frac <  _MIXED_ISSUE_THRESH,
    }


# ── Proxy scoring ─────────────────────────────────────────────────────────────

def _area_score(area_ratio: float) -> float:
    """
    Tolerant area score.  Within [_AREA_SCORE_LOWER, _AREA_SCORE_UPPER] → 1.0.
    Linear decay outside; reaches 0.0 at ratio 0.0 (below) or 2.0 (above).
    """
    if _AREA_SCORE_LOWER <= area_ratio <= _AREA_SCORE_UPPER:
        return 1.0
    if area_ratio < _AREA_SCORE_LOWER:
        return max(0.0, area_ratio / _AREA_SCORE_LOWER)
    # area_ratio > upper
    return max(0.0, 1.0 - (area_ratio - _AREA_SCORE_UPPER) / (2.0 - _AREA_SCORE_UPPER))


def _proxy_score(
    mask_orig:   np.ndarray,
    mask_cand:   np.ndarray,
    gt_mask:     np.ndarray | None,
    dice_before: float | None,
) -> float:
    """
    Composite proxy score for a candidate mask relative to the original.

    Uses a tolerant area score so that moderate area changes (e.g. RSO reducing
    debris by ~20 %) do not unfairly penalise a good correction.
    """
    if not np.any(mask_cand > 0):
        return -1.0

    area_orig  = float((mask_orig > 0).sum())
    area_cand  = float((mask_cand > 0).sum())
    area_ratio = area_cand / max(area_orig, 1.0)
    a_score    = _area_score(area_ratio)

    compact_orig  = _quick_compactness(mask_orig)
    compact_cand  = _quick_compactness(mask_cand)
    if compact_orig > 0:
        compact_delta = (compact_cand - compact_orig) / compact_orig
    else:
        compact_delta = 0.0
    compact_delta = max(-1.0, min(1.0, compact_delta))

    if gt_mask is not None and dice_before is not None:
        from src.evaluation.metrics import compute_dice
        dice_after = compute_dice(mask_cand, gt_mask)
        dice_delta = dice_after - dice_before
        score = (
            _W_AREA    * a_score +
            _W_COMPACT * (compact_delta + 1.0) / 2.0 +
            _W_DICE    * (dice_delta     + 1.0) / 2.0
        )
    else:
        w_a = _W_AREA    / (_W_AREA + _W_COMPACT)
        w_c = _W_COMPACT / (_W_AREA + _W_COMPACT)
        score = (
            w_a * a_score +
            w_c * (compact_delta + 1.0) / 2.0
        )

    return round(score, 6)


# ── Risk-aware acceptance ─────────────────────────────────────────────────────

def _accept_candidate(
    mask_orig:   np.ndarray,
    mask_cand:   np.ndarray,
    score_orig:  float,
    score_cand:  float,
    gt_mask:     np.ndarray | None,
    dice_before: float | None,
) -> tuple[bool, str]:
    """
    Three-tier risk-aware acceptance decision.

    Returns (accepted, reason_string).

    Tier 1 — Hard veto
        Reject if the candidate represents a catastrophic change:
          • area dropped below 50 % of original
          • object count collapsed below 10 % of original (2+ objects only)

    Tier 2 — Dice-first acceptance  (GT required)
        Accept if Dice improved by at least _DICE_GAIN_THRESHOLD (+0.002).
        This allows meaningful improvements even when area/compactness shift
        slightly (e.g. RSO removing large debris reduces area but improves GT fit).

    Tier 3 — Score-gate acceptance
        Accept if proxy score is within _SCORE_TOLERANCE of original score.
        Catches structural improvements not captured by Dice alone.
    """
    # ── Tier 1: hard veto ────────────────────────────────────────────────────
    area_orig = float((mask_orig > 0).sum())
    area_cand = float((mask_cand > 0).sum())
    area_ratio = area_cand / max(area_orig, 1.0)

    if area_ratio < _HARD_MIN_AREA_RATIO:
        return False, f"hard veto: area dropped to {area_ratio:.1%} (limit {_HARD_MIN_AREA_RATIO:.0%})"

    n_orig = int(sk_label(mask_orig > 0).max())
    n_cand = int(sk_label(mask_cand > 0).max())
    if n_orig >= 3 and n_cand / max(n_orig, 1) < _HARD_MIN_OBJ_RATIO:
        return False, f"hard veto: objects collapsed {n_orig}->{n_cand} ({n_cand/n_orig:.1%} retained)"

    # ── Tier 2: Dice-first acceptance ────────────────────────────────────────
    if gt_mask is not None and dice_before is not None:
        from src.evaluation.metrics import compute_dice
        dice_after = compute_dice(mask_cand, gt_mask)
        dice_delta = dice_after - dice_before
        if dice_delta >= _DICE_GAIN_THRESHOLD:
            return True, f"dice-first: gain={dice_delta:+.4f} >= {_DICE_GAIN_THRESHOLD}"

    # ── Tier 3: score-gate acceptance ────────────────────────────────────────
    threshold = score_orig - _SCORE_TOLERANCE
    if score_cand >= threshold:
        return True, f"score-gate: {score_cand:.4f} >= {threshold:.4f}"

    return False, f"rejected: score {score_cand:.4f} < gate {threshold:.4f}"


# ── Strategy routing ──────────────────────────────────────────────────────────

def _route_strategy(global_issue: str, spread: dict) -> tuple[str, str]:
    """
    Routing preference based on global issue and spread stats.

    Returns (strategy_preference, reason):
        "global" — prefer global pipeline result
        "region" — prefer region pipeline result
        "score"  — evaluate both with acceptance logic and pick best
    """
    if global_issue in _FORCE_GLOBAL_ISSUES:
        return (
            "global",
            f"global issue '{global_issue}' detected — debris requires global RSO",
        )

    dom   = spread["dominant_issue"]
    frac  = spread["dominant_frac"]
    n_sig = spread["n_significant"]

    if n_sig == 0:
        return ("region", "no significant regions — region pipeline handles degenerate mask")

    if dom == "holey" and frac >= _WIDESPREAD_THRESH:
        return (
            "global",
            f"holey widespread ({frac:.0%} of {n_sig} regions) — global morph_close preferred",
        )

    if dom == "over_segmented" and frac >= _WIDESPREAD_THRESH:
        return (
            "global",
            f"over_segmented widespread ({frac:.0%} of {n_sig} regions) — global morph_close preferred",
        )

    if spread["is_mixed"]:
        return (
            "region",
            f"mixed issues (dominant='{dom}' covers only {frac:.0%} of {n_sig} regions) — per-region precision preferred",
        )

    return (
        "score",
        f"dominant='{dom}' ({frac:.0%} of {n_sig} regions) — scoring both strategies to pick best",
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_hybrid_pipeline(
    mask:                 np.ndarray,
    gt_mask:              np.ndarray | None = None,
    agent=None,
    confidence_threshold: float = 0.30,
) -> dict:
    """
    Hybrid post-processing pipeline.

    Parameters
    ----------
    mask                 : Input binary mask (H, W), uint8, foreground = 255.
    gt_mask              : Optional ground-truth for evaluation.
    agent                : Agent instance; uses module-level singleton if None.
    confidence_threshold : Confidence gate passed to both sub-pipelines.

    Returns
    -------
    dict — superset of run_region_pipeline() keys, plus:
        selected_strategy  : str   — "global" | "region" | "original"
        strategy_reason    : str   — why this strategy was selected
        acceptance_reason  : str   — tier that accepted (or rejection explanation)
        spread             : dict  — issue spread analysis across regions
        global_result      : dict  — full result from global pipeline
        region_result      : dict  — full result from region pipeline
        score_original     : float — proxy score of original mask
        score_global       : float — proxy score of global pipeline output
        score_region       : float — proxy score of region pipeline output
    """
    if agent is None:
        agent = _DEFAULT_AGENT

    # ── Step 1: measure issue spread ─────────────────────────────────────────
    spread = _measure_issue_spread(mask, agent)

    # ── Step 2: run both pipelines (independent sessions) ────────────────────
    agent.reset_session()
    global_result = run_postprocessing_pipeline(
        mask, gt_mask=gt_mask, agent=agent,
        confidence_threshold=confidence_threshold,
    )

    agent.reset_session()
    region_result = run_region_pipeline(
        mask, gt_mask=gt_mask, agent=agent,
        confidence_threshold=confidence_threshold,
    )

    # ── Step 3: proxy scores ──────────────────────────────────────────────────
    dice_before: float | None = None
    if gt_mask is not None:
        dice_before = global_result["improvement_summary"].get("dice_before")

    score_original = _proxy_score(mask, mask,                        gt_mask, dice_before)
    score_global   = _proxy_score(mask, global_result["mask_after"], gt_mask, dice_before)
    score_region   = _proxy_score(mask, region_result["mask_after"], gt_mask, dice_before)

    # ── Step 4: route and select ──────────────────────────────────────────────
    global_issue        = global_result["decision"]["issue"]
    pref, pref_reason   = _route_strategy(global_issue, spread)

    selected_strategy = "original"
    acceptance_reason = "no candidate accepted"
    best_result       = _make_noop_result(mask, gt_mask, global_result)

    if pref == "global":
        accepted, acc_reason = _accept_candidate(
            mask, global_result["mask_after"],
            score_original, score_global, gt_mask, dice_before,
        )
        if accepted:
            selected_strategy = "global"
            acceptance_reason = acc_reason
            best_result       = global_result
        else:
            strategy_reason   = f"{pref_reason}; global not accepted ({acc_reason})"
            acceptance_reason = acc_reason

    elif pref == "region":
        accepted, acc_reason = _accept_candidate(
            mask, region_result["mask_after"],
            score_original, score_region, gt_mask, dice_before,
        )
        if accepted:
            selected_strategy = "region"
            acceptance_reason = acc_reason
            best_result       = region_result
        else:
            strategy_reason   = f"{pref_reason}; region not accepted ({acc_reason})"
            acceptance_reason = acc_reason

    else:
        # "score" — accept the best of global and region, then fall back to original
        g_ok, g_reason = _accept_candidate(
            mask, global_result["mask_after"],
            score_original, score_global, gt_mask, dice_before,
        )
        r_ok, r_reason = _accept_candidate(
            mask, region_result["mask_after"],
            score_original, score_region, gt_mask, dice_before,
        )

        if g_ok and r_ok:
            # Both accepted: pick higher proxy score as tiebreaker
            if score_global >= score_region:
                selected_strategy = "global"
                acceptance_reason = f"both accepted; global score ({score_global:.4f}) >= region ({score_region:.4f})"
                best_result       = global_result
            else:
                selected_strategy = "region"
                acceptance_reason = f"both accepted; region score ({score_region:.4f}) > global ({score_global:.4f})"
                best_result       = region_result
        elif g_ok:
            selected_strategy = "global"
            acceptance_reason = g_reason
            best_result       = global_result
        elif r_ok:
            selected_strategy = "region"
            acceptance_reason = r_reason
            best_result       = region_result
        else:
            acceptance_reason = f"neither accepted (global: {g_reason}; region: {r_reason})"

    # Build the full strategy_reason string for all branches
    if selected_strategy == "original":
        strategy_reason = (
            pref_reason + f"; kept original — {acceptance_reason}"
            if "strategy_reason" not in dir()
            else strategy_reason   # already set above
        )
    else:
        strategy_reason = f"{pref_reason}; {acceptance_reason}"

    # ── Step 5: final evaluation on selected mask ─────────────────────────────
    metrics_before = _evaluate(mask, gt_mask)
    metrics_after  = _evaluate(best_result["mask_after"], gt_mask)
    improvement    = _summarise(metrics_before, metrics_after, gt_mask is not None)

    # ── Build return dict ─────────────────────────────────────────────────────
    result = dict(best_result)
    result.update({
        "metrics_before":      metrics_before,
        "metrics_after":       metrics_after,
        "improvement_summary": improvement,
        "mask_before":         mask,
        "selected_strategy":   selected_strategy,
        "strategy_reason":     strategy_reason,
        "acceptance_reason":   acceptance_reason,
        "spread":              spread,
        "global_result":       global_result,
        "region_result":       region_result,
        "score_original":      score_original,
        "score_global":        score_global,
        "score_region":        score_region,
    })

    if "n_regions" not in result:
        result["n_regions"]        = region_result.get("n_regions", 0)
        result["regions_modified"] = region_result.get("regions_modified", 0)
        result["region_log"]       = region_result.get("region_log", [])
        result["global_log"]       = region_result.get("global_log", [])

    return result


# ── Helper: no-op result wrapping the original mask ──────────────────────────

def _make_noop_result(
    mask:      np.ndarray,
    gt_mask:   np.ndarray | None,
    reference: dict,
) -> dict:
    """Minimal result dict representing 'no action taken'."""
    metrics    = _evaluate(mask, gt_mask)
    improvement = _summarise(metrics, metrics, gt_mask is not None)
    result = dict(reference)
    result.update({
        "mask_after":           mask.copy(),
        "action_applied":       "no_action",
        "selected_checkpoint":  "original",
        "metrics_before":       metrics,
        "metrics_after":        metrics,
        "improvement_summary":  improvement,
        "step_log":             [],
        "n_regions":            reference.get("n_regions", 0),
        "regions_modified":     0,
        "region_log":           [],
        "global_log":           [],
    })
    return result
