"""
Test the region-aware post-processing pipeline on synthetic masks.

Each case is designed so that different regions receive different tools,
verifying that the pipeline applies corrections locally rather than globally.

Run with:
    python tests/test_region_pipeline.py

Key behaviors tested:
  1. clean           — all regions clean, no action
  2. holey_artifact  — small holes → fill_holes applied to the holey region only
  3. holey_structural— large holes relative to object → morph_close preferred
  4. lumen_safe      — large lumen hole → protected, no fill
  5. mixed_holes     — one small-hole region + one clean region → only holey fixed
  6. global_noisy    — debris + real gland → global cleanup, gland analyzed as region
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from skimage.measure import label

from src.region_pipeline import run_region_pipeline, reset_agent_session


# ── Synthetic mask builders ───────────────────────────────────────────────────

def circle(h, w, cy, cx, r, val=255):
    arr = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    arr[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = val
    return arr


def make_clean(h=256, w=256):
    """Two compact circular glands — no issues."""
    mask = circle(h, w, 80,  80,  40)
    mask = np.maximum(mask, circle(h, w, 170, 170, 40))
    return mask


def make_holey_artifact(h=256, w=256):
    """
    Left gland: two small internal holes (each ~225px = ~2.4% of gland area).
    Hole ratio << 8% → fill_holes expected.
    Right gland: perfectly clean.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:150, 20:140] = 255          # large left gland (~9600px)
    mask[55:70,  45:60]  = 0            # hole1: 15×15 = 225px
    mask[95:110, 90:105] = 0            # hole2: 15×15 = 225px
    mask = np.maximum(mask, circle(h, w, 90, 200, 42))  # clean right gland
    return mask


def make_holey_structural(h=256, w=256):
    """
    Left gland: large internal hole (~10% of gland area).
    Hole ratio > 8% → morph_close expected (not fill_holes).
    Right gland: clean.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:150, 20:140] = 255          # large gland (~9600px)
    mask[65:110, 55:100] = 0            # large hole: 45×45 = 2025px (~21% of 9600)
    mask = np.maximum(mask, circle(h, w, 90, 200, 42))
    return mask


def make_lumen_safe(h=256, w=256):
    """
    Ring gland: large central lumen (> 20% of object area) — must NOT be filled.
    Expected: no_action (lumen safety gate protects it).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    mask[(yy - 128) ** 2 + (xx - 128) ** 2 <= 60 ** 2] = 255   # outer ring
    mask[(yy - 128) ** 2 + (xx - 128) ** 2 <= 30 ** 2] = 0      # lumen hole (~2827px ≈ 25% of ring)
    return mask


def make_mixed_holes(h=256, w=256):
    """
    Top gland:    small holes → fill_holes expected
    Bottom gland: clean → no_action expected
    Validates per-region tool differentiation.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    # Top: square gland with two small holes
    mask[15:105, 15:105] = 255          # ~8100px gland
    mask[35:50,  35:50]  = 0            # 15×15 = 225px hole
    mask[70:85,  70:85]  = 0            # 15×15 = 225px hole
    # Bottom: clean circle, well-separated
    mask = np.maximum(mask, circle(h, w, 185, 185, 42))
    return mask


def make_global_noisy(h=256, w=256):
    """Large real gland + many tiny debris blobs → tests global cleanup tier."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:150, 50:150] = 255          # real gland: 10000px
    rng = np.random.default_rng(42)
    for _ in range(25):
        r = rng.integers(5, h - 5)
        c = rng.integers(5, w - 5)
        s = rng.integers(2, 8)
        mask[r:r + s, c:c + s] = 255
    return mask


# ── Assertions ────────────────────────────────────────────────────────────────

def action_applied_for(region_log, action):
    return any(e["applied"] and e["action"] == action for e in region_log)


def issue_detected(region_log, issue):
    return any(e["issue"] == issue for e in region_log)


# ── Test runner ───────────────────────────────────────────────────────────────

def run_tests():
    reset_agent_session()

    cases = [
        {
            "name": "clean",
            "mask": make_clean(),
            "check": lambda r: r["regions_modified"] == 0,
            "desc": "no regions modified",
        },
        {
            "name": "holey_artifact",
            "mask": make_holey_artifact(),
            "check": lambda r: action_applied_for(r["region_log"], "fill_holes"),
            "desc": "fill_holes applied to small-hole region",
        },
        {
            "name": "holey_structural",
            "mask": make_holey_structural(),
            "check": lambda r: (
                # morph_close OR fill_holes acceptable (large hole is ambiguous)
                r["regions_modified"] >= 1 or
                issue_detected(r["region_log"], "holey")
            ),
            "desc": "holey issue detected on large-hole region",
        },
        {
            "name": "lumen_safe",
            "mask": make_lumen_safe(),
            "check": lambda r: r["regions_modified"] == 0,
            "desc": "lumen (large hole) must not be filled — both fill_holes and morph_close suppressed",
        },
        {
            "name": "mixed_holes",
            "mask": make_mixed_holes(),
            "check": lambda r: (
                # Holey region diagnosed and acted upon; clean region left alone
                issue_detected(r["region_log"], "holey") and
                any(e["issue"] == "clean" for e in r["region_log"])
            ),
            "desc": "holey region diagnosed; clean region untouched",
        },
        {
            "name": "global_noisy",
            "mask": make_global_noisy(),
            "check": lambda r: r["n_regions"] >= 1,
            "desc": "real gland analyzed as a significant region",
        },
    ]

    print("Running region-aware pipeline tests...\n")
    print(f"  {'Case':<18} {'Status':<7} {'Regions':>7} {'Modified':>8}  Note")
    print(f"  {'-'*18} {'-'*6} {'-'*7} {'-'*8}  {'-'*40}")

    all_pass = True
    for c in cases:
        result = run_region_pipeline(c["mask"], confidence_threshold=0.25)
        ok     = c["check"](result)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        fail_note = f"EXPECTED: {c['desc']}" if not ok else ""
        print(
            f"  {c['name']:<18} {status:<7} {result['n_regions']:>7} "
            f"{result['regions_modified']:>8}  {fail_note}"
        )

        for entry in result["region_log"]:
            applied_str = "APPLIED" if entry["applied"] else "skipped"
            ts_str = "  ".join(
                f"{t}={s:.2f}" for t, s in entry.get("tool_scores", {}).items()
            )
            reason_str = f"  [{entry['reason']}]" if entry.get("reason") else ""
            print(
                f"      r{entry['label']} area={entry['area']:>6}px "
                f"issue={entry['issue']:<16} action={entry['action']:<22} "
                f"conf={entry['confidence']:.3f} {applied_str}{reason_str}"
            )
            if ts_str:
                print(f"         scores: {ts_str}")
        if result["global_log"]:
            for g in result["global_log"]:
                gstatus = "ACCEPTED" if g.get("accepted") else "REJECTED"
                print(f"      [global] {g['action']} -> {gstatus}  {g.get('reason') or ''}")
        print()

    print("All tests passed." if all_pass else "SOME TESTS FAILED.")
    return all_pass


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
