"""
Test PathologyReasoningAgent across all 8 issue types.

Run with:
    python tests/test_agent.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.reasoning import PathologyReasoningAgent

agent = PathologyReasoningAgent()

# ── Mock feature dictionaries ─────────────────────────────────────────────────
# Each set is designed to clearly represent one failure mode.

FEATURES = {
    # 1. Noisy: one large gland + many tiny debris fragments
    #    small_object_ratio=0.85, size_cv=2.5 → bimodal size distribution
    "noisy": {
        "num_objects": 20, "avg_object_size": 350.0, "max_object_size": 15000.0,
        "object_size_std": 875.0, "size_cv": 2.5, "small_object_ratio": 0.85,
        "boundary_roughness": 0.95, "holes_count": 0, "avg_hole_size": 0.0,
        "total_mask_area": 7000.0, "mask_density": 0.11, "compactness": 1.60,
    },
    # 2. Holey: single gland with several small internal holes (lumen artifacts)
    #    holes_count=4, avg_hole_size=280 → small discrete holes → fill_holes
    "holey": {
        "num_objects": 1, "avg_object_size": 24000.0, "max_object_size": 24000.0,
        "object_size_std": 0.0, "size_cv": 0.0, "small_object_ratio": 0.0,
        "boundary_roughness": 1.05, "holes_count": 4, "avg_hole_size": 280.0,
        "total_mask_area": 24000.0, "mask_density": 0.37, "compactness": 0.90,
    },
    # 3. Merged: two glands fully fused — single large object, very low compactness
    #    compactness=0.35, smooth boundary (roughness=1.25) → watershed_split
    "merged": {
        "num_objects": 1, "avg_object_size": 18000.0, "max_object_size": 18000.0,
        "object_size_std": 0.0, "size_cv": 0.0, "small_object_ratio": 0.0,
        "boundary_roughness": 1.25, "holes_count": 0, "avg_hole_size": 0.0,
        "total_mask_area": 18000.0, "mask_density": 0.28, "compactness": 0.35,
    },
    # 4. Thin bridge: two glands joined by a narrow neck
    #    compactness=0.42, roughness=1.90 (concave neck raises roughness) → morph_open
    "thin_bridge": {
        "num_objects": 2, "avg_object_size": 9000.0, "max_object_size": 12000.0,
        "object_size_std": 1500.0, "size_cv": 0.33, "small_object_ratio": 0.0,
        "boundary_roughness": 1.90, "holes_count": 0, "avg_hole_size": 0.0,
        "total_mask_area": 18000.0, "mask_density": 0.28, "compactness": 0.42,
    },
    # 5. Under-segmented: mask is very sparse, gland boundary too tight
    #    mask_density=0.008 → well below 0.05 threshold → dilation
    "under_segmented": {
        "num_objects": 2, "avg_object_size": 300.0, "max_object_size": 450.0,
        "object_size_std": 75.0, "size_cv": 0.25, "small_object_ratio": 0.0,
        "boundary_roughness": 1.08, "holes_count": 0, "avg_hole_size": 0.0,
        "total_mask_area": 600.0, "mask_density": 0.008, "compactness": 0.88,
    },
    # 6. Over-segmented: many fragments of similar size (low cv, high count)
    #    num_objects=12, size_cv=0.3, small_ratio=0.0 → morph_close
    "over_segmented": {
        "num_objects": 12, "avg_object_size": 1800.0, "max_object_size": 2400.0,
        "object_size_std": 540.0, "size_cv": 0.30, "small_object_ratio": 0.0,
        "boundary_roughness": 1.10, "holes_count": 0, "avg_hole_size": 0.0,
        "total_mask_area": 21600.0, "mask_density": 0.33, "compactness": 0.85,
    },
    # 7. Rough boundary: single gland, correct shape, jagged edges
    #    compactness=0.78, roughness=1.75 → morph_close
    "rough_boundary": {
        "num_objects": 1, "avg_object_size": 14000.0, "max_object_size": 14000.0,
        "object_size_std": 0.0, "size_cv": 0.0, "small_object_ratio": 0.0,
        "boundary_roughness": 1.75, "holes_count": 0, "avg_hole_size": 0.0,
        "total_mask_area": 14000.0, "mask_density": 0.21, "compactness": 0.78,
    },
    # 8. Clean: single well-shaped gland, no issues
    "clean": {
        "num_objects": 1, "avg_object_size": 11000.0, "max_object_size": 11000.0,
        "object_size_std": 0.0, "size_cv": 0.0, "small_object_ratio": 0.0,
        "boundary_roughness": 1.05, "holes_count": 0, "avg_hole_size": 0.0,
        "total_mask_area": 11000.0, "mask_density": 0.17, "compactness": 0.91,
    },
}

EXPECTED = {
    "noisy":          ("noisy",          "remove_small_objects"),
    "holey":          ("holey",          "fill_holes"),
    "merged":         ("merged",         "watershed_split"),
    "thin_bridge":    ("thin_bridge",    "morph_open"),
    "under_segmented":("under_segmented","dilation"),
    "over_segmented": ("over_segmented", "morph_close"),
    "rough_boundary": ("rough_boundary", "morph_close"),
    "clean":          ("clean",          "no_action"),
}


def test_all():
    print("Running agent decision tests...\n")
    col = 16
    print(f"  {'Case':<{col}} {'Status':<7} {'Issue':<17} {'Action':<26} {'Conf':>6}")
    print(f"  {'-'*col} {'-'*6} {'-'*16} {'-'*25} {'-'*6}")

    all_pass = True
    for name, features in FEATURES.items():
        d = agent.decide(features)
        exp_issue, exp_action = EXPECTED[name]
        ok = d["issue"] == exp_issue and d["selected_action"] == exp_action
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(
            f"  {name:<{col}} {status:<7} {d['issue']:<17} "
            f"{d['selected_action']:<26} {d['confidence']:>6.3f}"
        )

    print()
    return all_pass


def print_decisions():
    print("=" * 65)
    print("DETAILED DECISIONS")
    print("=" * 65)
    for name, features in FEATURES.items():
        d = agent.decide(features)
        print(f"\n[{name.upper()}]")
        print(f"  issue           : {d['issue']}")
        print(f"  candidates      : {d['candidate_tools']}")
        print(f"  tool_scores     : {d['tool_scores']}")
        print(f"  selected_action : {d['selected_action']}")
        print(f"  confidence      : {d['confidence']}")
        print(f"  reason          : {d['reason']}")


if __name__ == "__main__":
    passed = test_all()
    print_decisions()
    print("\n" + ("All tests passed." if passed else "SOME TESTS FAILED."))
