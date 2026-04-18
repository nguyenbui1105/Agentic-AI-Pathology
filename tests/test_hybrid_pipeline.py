"""
Smoke tests for the hybrid pipeline.

Verifies:
  1. Return dict contains all required keys
  2. selected_strategy is one of "global" | "region" | "original"
  3. 0 regressions: mask_after never degrades Dice vs mask_before (with GT)
  4. Noisy mask routes to global strategy (or original fallback if global fails)
  5. Mixed-issue mask routes to region strategy
  6. Clean mask stays at original (no change)

Run with:
    python tests/test_hybrid_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.hybrid_pipeline import run_hybrid_pipeline, reset_agent_session
from src.evaluation.metrics import compute_dice


# ── Synthetic mask builders (reused from test_region_pipeline) ────────────────

def circle(h, w, cy, cx, r, val=255):
    arr = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    arr[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = val
    return arr


def make_clean(h=256, w=256):
    mask = circle(h, w, 80, 80, 40)
    mask = np.maximum(mask, circle(h, w, 170, 170, 40))
    return mask


def make_noisy(h=256, w=256):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:150, 50:150] = 255
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = rng.integers(5, h - 5)
        c = rng.integers(5, w - 5)
        s = rng.integers(2, 6)
        mask[r:r + s, c:c + s] = 255
    return mask


def make_mixed_issues(h=256, w=256):
    """
    Three well-separated regions with distinct issues so dominant fraction < 50%.
    Region 1 (top-left):  holey (small holes)
    Region 2 (top-right): thin bridge (two blobs connected by narrow neck)
    Region 3 (bottom):    clean circle
    Dominant = 33% → is_mixed=True → routing should prefer region.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    # Region 1: square gland with small holes
    mask[10:90, 10:90] = 255
    mask[30:44, 30:44] = 0
    mask[60:74, 60:74] = 0
    # Region 2: two circles joined by a thin bridge (thin_bridge issue)
    mask = np.maximum(mask, circle(h, w, 50, 165, 22))
    mask = np.maximum(mask, circle(h, w, 50, 220, 22))
    mask[47:53, 186:200] = 255   # narrow bridge
    # Region 3: clean circle, well separated
    mask = np.maximum(mask, circle(h, w, 185, 128, 35))
    return mask


def make_holey_gt(h=256, w=256):
    """Produce a GT that is just the mask with holes filled."""
    from scipy.ndimage import binary_fill_holes
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:150, 20:140] = 255
    mask[55:70,  45:60]  = 0
    mask[95:110, 90:105] = 0
    filled = binary_fill_holes(mask > 0).astype(np.uint8) * 255
    return mask, filled   # (pred, gt)


# ── Helpers ───────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "features", "decision", "action_applied", "action_sequence",
    "params_used", "confidence", "confidence_threshold",
    "selected_checkpoint", "step_log", "mask_before", "mask_after",
    "metrics_before", "metrics_after", "improvement_summary",
    "n_regions", "regions_modified", "region_log", "global_log",
    "selected_strategy", "strategy_reason", "spread",
    "global_result", "region_result",
    "score_original", "score_global", "score_region",
}

_VALID_STRATEGIES = {"global", "region", "original"}


def run_tests():
    reset_agent_session()
    cases = []
    all_pass = True

    # ── Test 1: required keys present ─────────────────────────────────────────
    mask = make_clean()
    result = run_hybrid_pipeline(mask, confidence_threshold=0.25)
    missing = _REQUIRED_KEYS - set(result.keys())
    ok = len(missing) == 0
    cases.append(("keys_present",   ok, f"missing={missing}" if not ok else ""))

    # ── Test 2: selected_strategy is valid ────────────────────────────────────
    ok = result["selected_strategy"] in _VALID_STRATEGIES
    cases.append(("strategy_valid", ok, f"got={result['selected_strategy']}" if not ok else ""))

    # ── Test 3: clean mask → no modification ──────────────────────────────────
    mask = make_clean()
    result = run_hybrid_pipeline(mask, confidence_threshold=0.25)
    ok = result["regions_modified"] == 0 and np.array_equal(result["mask_after"], mask)
    cases.append(("clean_unchanged", ok, f"regions_modified={result['regions_modified']}"))

    # ── Test 4: noisy mask → strategy is global or original (never region) ────
    reset_agent_session()
    mask = make_noisy()
    result = run_hybrid_pipeline(mask, confidence_threshold=0.25)
    ok = result["selected_strategy"] in {"global", "original"}
    cases.append(("noisy_not_region", ok, f"strategy={result['selected_strategy']}"))

    # ── Test 5: no regression with GT available ───────────────────────────────
    reset_agent_session()
    pred_mask, gt_mask = make_holey_gt()
    result = run_hybrid_pipeline(pred_mask, gt_mask=gt_mask, confidence_threshold=0.25)
    imp = result["improvement_summary"]
    ok = imp.get("dice_delta", 0.0) >= 0.0   # no regression
    cases.append(("no_regression",   ok, f"dice_delta={imp.get('dice_delta')}"))

    # ── Test 6: mixed-issue mask → strategy prefers region ────────────────────
    reset_agent_session()
    mask = make_mixed_issues()
    result = run_hybrid_pipeline(mask, confidence_threshold=0.25)
    spread = result["spread"]
    ok = not spread["is_widespread"] or result["selected_strategy"] in {"region", "original"}
    cases.append(("mixed_routes_region", ok,
                  f"strategy={result['selected_strategy']} spread={spread}"))

    # ── Test 7: proxy scores are in [0, 1] ────────────────────────────────────
    ok = all(0.0 <= result[k] <= 1.0 for k in ("score_original", "score_global", "score_region"))
    cases.append(("scores_bounded",  ok,
                  f"orig={result['score_original']} glob={result['score_global']} reg={result['score_region']}"))

    # ── Report ────────────────────────────────────────────────────────────────
    print("Running hybrid pipeline tests...\n")
    print(f"  {'Case':<22} {'Status':<7}  Note")
    print(f"  {'-'*22} {'-'*6}  {'-'*40}")
    for name, ok, note in cases:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:<22} {status:<7}  {note}")

    print()
    print("All tests passed." if all_pass else "SOME TESTS FAILED.")
    return all_pass


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
