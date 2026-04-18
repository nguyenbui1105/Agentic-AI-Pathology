"""
Deterministic reasoning agent for pathology gland segmentation post-processing.

Decision flow:
    features
      -> _compute_issue_scores()  # score all 7 issue types
      -> interpret_features()     # pick dominant issue + severity
      -> get_candidate_tools()    # map issue to relevant tools
      -> score_tools()            # rate each candidate against feature values
      -> select_action()          # pick highest score, compute confidence
      -> _get_action_sequence()   # optionally extend to 2-step if secondary issue
      -> decide()                 # return full structured decision dict

Issue taxonomy (8 types):
    noisy           — small debris fragments dominating the mask
    holey           — internal voids / unfilled lumen regions
    merged          — two or more glands fully fused into one blob
    thin_bridge     — glands connected by a narrow neck (partial merge)
    over_segmented  — gland split into too many similarly-sized fragments
    under_segmented — mask too sparse; gland boundaries predicted too tight
    rough_boundary  — jagged/irregular edges on otherwise well-shaped objects
    clean           — no significant issue detected

Multi-step actions:
    When a primary issue and a significant secondary issue co-exist,
    the agent may return a 2-step action_sequence, e.g.:
        remove_small_objects -> morph_close
        morph_open           -> watershed_split
"""

from __future__ import annotations


# ── Single-issue thresholds ───────────────────────────────────────────────────

_COMPACTNESS_MERGED  = 0.55   # below → likely full merge (smooth but non-circular)
_COMPACTNESS_BRIDGE  = 0.75   # below → possible thin-bridge or partial merge
_ROUGHNESS_BRIDGE    = 1.30   # roughness above this → consider bridge / rough boundary
_SIZE_MERGED         = 3000   # avg_object_size above this → candidate for merge
_NUM_MERGED_SOFT_CAP = 2      # n_obj above this starts reducing merged score (soft gate)
_DENSITY_UNDERSEG    = 0.05   # mask_density below this → possible under-segmentation

# ── Dilation safety thresholds ────────────────────────────────────────────────
_DILATION_MAX_AREA_PER_OBJ = 3000  # avg area/object above this → not under-segmented
_DILATION_MERGE_PROXIMITY  = 0.15  # density*(n_obj-1) above this → high merge risk

# ── fill_holes specialization thresholds ─────────────────────────────────────
_LUMEN_RATIO_THRESHOLD   = 0.20   # hole > 20% of object area → likely true lumen, skip
_SAFE_HOLE_RATIO         = 0.05   # hole < 5% of object area → definitely safe artifact
_FILL_HOLES_SMALL_RATIO  = 0.08   # hole < 8% → small artifact zone → fill_holes preferred
_MORPH_CLOSE_LARGE_RATIO = 0.10   # hole > 10% → structural gap zone → morph_close preferred

# morph_close competes with fill_holes: reduce morph_close when fill_holes is viable
_MORPH_CLOSE_PENALTY_VS_SPECIFIC = 0.88

# ── Risky tool scoring penalties ──────────────────────────────────────────────
_RISKY_PENALTY_EROSION  = 0.85   # 15% discount on erosion score
_RISKY_PENALTY_DILATION = 0.90   # 10% discount on dilation score

# ── Session-level diversity discount ─────────────────────────────────────────
# Each time a tool is applied, its score is discounted 5% (max 15%) in
# subsequent decisions to encourage exploration when alternatives score similarly.
_DIVERSITY_DISCOUNT_PER_USE = 0.05
_DIVERSITY_DISCOUNT_MAX     = 0.15

# ── Multi-step action rules ───────────────────────────────────────────────────
# (primary_issue, secondary_issue, min_secondary_score, [step1, step2])
_MULTI_STEP_RULES: list[tuple[str, str, float, list[str]]] = [
    ("noisy",           "holey",            0.15, ["remove_small_objects", "morph_close"]),
    ("noisy",           "over_segmented",   0.20, ["remove_small_objects", "morph_close"]),
    ("thin_bridge",     "merged",           0.12, ["morph_open", "watershed_split"]),
    ("merged",          "thin_bridge",      0.10, ["morph_open", "watershed_split"]),
    ("over_segmented",  "holey",            0.15, ["morph_close", "fill_holes"]),
    ("under_segmented", "rough_boundary",   0.15, ["dilation", "morph_close"]),
    ("rough_boundary",  "holey",            0.20, ["morph_close", "fill_holes"]),
]


class PathologyReasoningAgent:
    """
    Rule-based reasoning agent supporting the full 7-tool action space
    with optional 2-step sequential correction.

    Available actions
    -----------------
    remove_small_objects  — remove noise / debris fragments
    fill_holes            — fill small internal voids (targeted lumen fix)
    morph_close           — fill gaps and smooth rough boundaries
    morph_open            — remove thin protrusions / narrow bridges
    erosion               — shrink over-predicted boundaries inward
    dilation              — expand under-predicted boundaries outward
    watershed_split       — separate fully merged / fused glands
    no_action             — mask is already acceptable
    """

    _CANDIDATES: dict[str, list[str]] = {
        "noisy":           ["remove_small_objects"],
        "holey":           ["fill_holes", "morph_close"],
        "merged":          ["watershed_split", "erosion"],
        "thin_bridge":     ["morph_open", "erosion"],
        "over_segmented":  ["morph_close", "dilation"],
        "under_segmented": ["dilation"],
        "rough_boundary":  ["morph_close"],
        "fragmented":      ["connect_fragments"],
        "clean":           ["no_action"],
    }

    _PATHOLOGY: dict[str, str] = {
        "fragmented": (
            "The gland body is broken into a large main component with small nearby "
            "satellite pieces.  The model under-predicted boundary continuity along "
            "part of the gland edge, leaving short gaps between fragments and the "
            "main body.  connect_fragments bridges these gaps without enlarging the "
            "gland overall."
        ),
        "noisy": (
            "Mask contains small spurious objects (staining artifacts, tissue debris, "
            "or model false positives) mixed with real glands."
        ),
        "holey": (
            "Gland interiors have unfilled holes — lumen regions or staining voids "
            "incorrectly predicted as background."
        ),
        "merged": (
            "Two or more glands appear fully fused into one large object; "
            "the shared boundary was not detected by the model."
        ),
        "thin_bridge": (
            "Adjacent glands are connected by a narrow neck or bridge; "
            "they are partially merged rather than fully fused."
        ),
        "over_segmented": (
            "A single gland was fragmented into many similarly-sized pieces; "
            "the model over-detected internal boundaries."
        ),
        "under_segmented": (
            "The foreground mask is very sparse; gland boundaries were predicted "
            "too tight, leaving real gland tissue as background."
        ),
        "rough_boundary": (
            "Gland shapes are broadly correct but boundaries are jagged or irregular, "
            "likely due to model uncertainty along the gland edge."
        ),
        "clean": (
            "No significant segmentation artifacts detected. Mask appears acceptable."
        ),
    }

    def __init__(self) -> None:
        self._tool_usage: dict[str, int] = {}

    def reset_session(self) -> None:
        """Reset session-level tool usage counters (call before a fresh evaluation run)."""
        self._tool_usage.clear()

    def record_action(self, tool: str) -> None:
        """Increment usage count for a tool that was actually applied."""
        if tool and tool != "no_action":
            self._tool_usage[tool] = self._tool_usage.get(tool, 0) + 1

    def _diversity_discount(self, tool: str) -> float:
        """
        Returns a multiplicative factor in [1-max_discount, 1.0].
        Tools used frequently in this session score slightly lower to
        encourage selection of valid alternatives when scores are close.
        """
        uses = self._tool_usage.get(tool, 0)
        return max(1.0 - _DIVERSITY_DISCOUNT_MAX,
                   1.0 - uses * _DIVERSITY_DISCOUNT_PER_USE)

    # ── Stage 0: raw issue scoring ────────────────────────────────────────────

    def _compute_issue_scores(self, features: dict) -> dict[str, float]:
        """
        Compute raw severity scores for the 7 non-clean issue types.

        Separated from interpret_features() so that decide() can read the full
        score table when planning multi-step action sequences.
        """
        compact   = features["compactness"]
        roughness = features["boundary_roughness"]
        n_obj     = features["num_objects"]
        avg_size  = features["avg_object_size"]
        ratio     = features["small_object_ratio"]
        cv        = features.get("size_cv", 0.0)
        h_count   = features["holes_count"]
        h_size    = features["avg_hole_size"]
        density   = features["mask_density"]
        hole_pen  = 0.2 if h_count > 0 else 1.0

        near_frag = features.get("nearby_fragment_ratio", 0.0)
        scores: dict[str, float] = {}

        # 1. noisy — bimodal size distribution (large glands + tiny debris)
        scores["noisy"] = ratio * min(1.0, cv / 2.0)

        # 2. holey — internal holes present
        scores["holey"] = (
            min(1.0, h_count / 5.0) * 0.5 +
            min(1.0, h_size / 2000.0) * 0.5
        )

        # 3. merged — fully fused: low compactness, large area, smooth boundary
        # Soft few-object gate (was binary n<=3): allows partial credit for n=3-5
        size_sig    = min(1.0, avg_size / _SIZE_MERGED)
        compact_def = max(0.0, (_COMPACTNESS_MERGED - compact) / _COMPACTNESS_MERGED)
        few_obj     = max(0.0, 1.0 - max(0, n_obj - _NUM_MERGED_SOFT_CAP) / 4.0)
        not_bridge  = max(0.0, 1.0 - max(0.0, roughness - _ROUGHNESS_BRIDGE) / 0.7)
        scores["merged"] = compact_def * size_sig * few_obj * hole_pen * not_bridge

        # 4. thin_bridge — partial merge: moderate compactness loss + high roughness
        bridge_compact = max(0.0, (_COMPACTNESS_BRIDGE - compact) / _COMPACTNESS_BRIDGE)
        bridge_rough   = min(1.0, max(0.0, roughness - _ROUGHNESS_BRIDGE) / 0.7)
        few_obj_tb     = 1.0 if n_obj <= 4 else 0.2
        scores["thin_bridge"] = bridge_compact * bridge_rough * few_obj_tb * hole_pen

        # 5. over_segmented — many fragments of similar size (not noisy debris)
        many_obj  = min(1.0, max(0.0, (n_obj - 4) / 10.0))
        equal_sz  = max(0.0, 1.0 - min(cv, 1.0))
        low_noise = max(0.0, 1.0 - ratio)
        scores["over_segmented"] = many_obj * equal_sz * low_noise

        # 6. under_segmented — very sparse foreground area
        scores["under_segmented"] = max(0.0, (_DENSITY_UNDERSEG - density) / _DENSITY_UNDERSEG)

        # 7. rough_boundary — high roughness on well-shaped (not merged) objects
        compact_rb = max(0.0, (compact - 0.65) / 0.35)
        rough_rb   = min(1.0, max(0.0, roughness - _ROUGHNESS_BRIDGE) / 0.7)
        scores["rough_boundary"] = compact_rb * rough_rb

        # 8. fragmented — small objects spatially close to large components
        # Distinguishes "fragments of a broken gland" from random debris.
        # Requires both a high small-object ratio AND high proximity.
        # Score = ratio * proximity_signal, scaled so that ratio=0.4, near_frag=0.8
        # gives ~0.64 which comfortably beats noisy (ratio*cv) in typical cases.
        scores["fragmented"] = round(min(1.0, ratio * near_frag * 2.0), 4)

        return scores

    # ── Stage 1: interpret features ───────────────────────────────────────────

    def interpret_features(self, features: dict) -> tuple[str, float]:
        """
        Score all 8 issue types and return the dominant issue + its severity.

        Differentiation rules
        ---------------------
        merged vs thin_bridge  : roughness — full merge is smooth (low roughness);
            thin bridge creates a concave neck (high roughness).
        thin_bridge vs rough_boundary : compactness — bridge lowers compactness
            substantially (< 0.75); jagged edges reduce it only mildly (> 0.65).
        noisy vs over_segmented : size_cv — noise is bimodal (high cv);
            over-segmented glands are similarly sized (low cv).
        holey/artifact vs holey/structural : tool selection uses hole_ratio
            (avg_hole_size / avg_object_size) — small ratio → fill_holes;
            large ratio or fragmented mask → morph_close.
        """
        scores = self._compute_issue_scores(features)
        best_issue = max(scores, key=lambda k: scores[k])
        best_score = scores[best_issue]

        if best_score < 0.05:
            return "clean", 0.0

        return best_issue, round(best_score, 4)

    # ── Stage 2: candidate tools ──────────────────────────────────────────────

    def get_candidate_tools(self, issue: str) -> list[str]:
        return list(self._CANDIDATES.get(issue, ["no_action"]))

    # ── Stage 3: score tools ──────────────────────────────────────────────────

    def score_tools(self, candidates: list[str], features: dict) -> dict[str, float]:
        """
        Score each candidate tool using feature values.

        Key specializations
        -------------------
        fill_holes   : wins when hole_ratio < 8% of object area (artifact territory)
                       AND the gland is compact and has few fragments.
                       Suppressed for large lumen (hole > 20% of object area).

        morph_close  : wins for structural gaps (large holes > 10%), fragmented masks
                       (many objects), or rough boundaries. Does NOT win on small
                       artifact holes alone — fill_holes is more precise there.
                       Penalised when fill_holes scores above 0.30 (gives up 12%).

        watershed_split : scores higher with very low compactness AND large objects.
                       Few-object bonus amplifies the score for classic 2-gland merges.

        morph_open   : bridge topology signal: few objects + compactness drop + roughness
                       all compound. Wins clearly when all three signals are present.

        erosion      : conservative 15% penalty; fallback when bridge/merge signals
                       are insufficient for the dedicated split tools.

        dilation     : 3-gate: sparse mask + small per-object area + low merge risk.
                       10% risky penalty; suppressed near other objects.
        """
        compact    = features["compactness"]
        roughness  = features["boundary_roughness"]
        n_obj      = features["num_objects"]
        avg_size   = features["avg_object_size"]
        ratio      = features["small_object_ratio"]
        cv         = features.get("size_cv", 0.0)
        h_count    = features["holes_count"]
        h_size     = features["avg_hole_size"]
        density    = features["mask_density"]
        total_area = features.get("total_mask_area", avg_size * n_obj)

        hole_ratio = h_size / max(avg_size, 1.0) if h_count > 0 else 0.0

        scores: dict[str, float] = {}

        for tool in candidates:

            if tool == "remove_small_objects":
                count_sig = min(1.0, (n_obj - 1) / 30.0) if n_obj > 1 else 0.0
                scores[tool] = round(
                    ratio * min(1.0, cv / 2.0) * 0.7 + count_sig * 0.3, 4
                )

            elif tool == "fill_holes":
                # Hard lumen gate: hole > 20% of object → likely true lumen, skip.
                lumen_penalty = min(1.0, hole_ratio / _LUMEN_RATIO_THRESHOLD)
                safe_to_fill  = max(0.0, 1.0 - lumen_penalty)

                # Extra suppression: compact gland + large relative hole → tubular lumen.
                compact_lumen = max(0.0, compact - 0.70) / 0.30 * lumen_penalty
                safe_to_fill  = max(0.0, safe_to_fill - compact_lumen * 0.5)

                # Primary signal: clearly small artifact holes (< 8% of object).
                if hole_ratio < _FILL_HOLES_SMALL_RATIO:
                    small_hole_sig = 1.0 - hole_ratio / _FILL_HOLES_SMALL_RATIO
                else:
                    small_hole_sig = 0.0

                # Shape stability: compact + intact (few objects) = fill_holes is safe.
                compact_sig = max(0.0, (compact - 0.60) / 0.40)
                intact_sig  = max(0.0, 1.0 - max(0, n_obj - 4) / 6.0)

                count_sig = min(1.0, h_count / 3.0)

                scores[tool] = round(
                    (count_sig * 0.30 + small_hole_sig * 0.45 + compact_sig * intact_sig * 0.25)
                    * safe_to_fill, 4
                )

            elif tool == "morph_close":
                # Lumen safety: large hole (> 20% of object) → kernel that could
                # close the lumen would destroy tubular gland structure.
                if hole_ratio > _LUMEN_RATIO_THRESHOLD:
                    scores[tool] = 0.0
                    continue

                # Structural gap: large holes relative to object size.
                structural_gap = min(1.0, hole_ratio / _MORPH_CLOSE_LARGE_RATIO) if h_count > 0 else 0.0
                # Boundary roughness signal.
                rough_sig = min(1.0, max(0.0, roughness - 1.2) / 0.8)
                # Fragmentation signal: many objects (scale from n_obj=3).
                frag_sig  = min(1.0, max(0.0, (n_obj - 3) / 8.0))

                base = max(structural_gap, rough_sig, frag_sig * 0.65)
                scores[tool] = round(base, 4)

            elif tool == "watershed_split":
                # Primary signal: compactness well below merged threshold.
                compact_sig   = max(0.0, (_COMPACTNESS_MERGED - compact) / _COMPACTNESS_MERGED)
                # Amplifier: large objects are more likely merged glands.
                size_sig      = min(1.0, avg_size / _SIZE_MERGED)
                # Few-object bonus: classic 2-gland merge has n_obj = 1-2.
                few_obj_bonus = 1.0 if n_obj <= 2 else (0.75 if n_obj <= 3 else 0.50)
                scores[tool]  = round(
                    compact_sig * (0.55 + size_sig * few_obj_bonus * 0.45), 4
                )

            elif tool == "erosion":
                raw = max(0.0, (_COMPACTNESS_BRIDGE - compact) / _COMPACTNESS_BRIDGE) * 0.5
                scores[tool] = round(raw * _RISKY_PENALTY_EROSION, 4)

            elif tool == "morph_open":
                # Roughness signal alone (captures jagged bridges).
                rough_sig = min(1.0, max(0.0, roughness - 1.2) / 0.8)
                # Bridge topology: compactness drop + few objects + roughness together.
                bridge_compact   = max(0.0, (_COMPACTNESS_BRIDGE - compact) / _COMPACTNESS_BRIDGE)
                few_obj_topology = 1.0 if n_obj <= 4 else 0.30
                bridge_topology  = bridge_compact * few_obj_topology * rough_sig
                # Take the stronger signal; bridge_topology boosts score when all three agree.
                scores[tool] = round(max(rough_sig * 0.65, bridge_topology), 4)

            elif tool == "dilation":
                underseg   = max(0.0, (_DENSITY_UNDERSEG - density) / _DENSITY_UNDERSEG)
                area_per_obj = total_area / max(n_obj, 1)
                small_area   = max(0.0, 1.0 - area_per_obj / _DILATION_MAX_AREA_PER_OBJ)
                if n_obj > 1:
                    proximity  = min(1.0, density * (n_obj - 1) / _DILATION_MERGE_PROXIMITY)
                    merge_safe = max(0.0, 1.0 - proximity)
                else:
                    merge_safe = 1.0
                scores[tool] = round(underseg * small_area * merge_safe * _RISKY_PENALTY_DILATION, 4)

            elif tool == "connect_fragments":
                near_frag    = features.get("nearby_fragment_ratio", 0.0)
                proximity_sig = min(1.0, near_frag / 0.5)   # full score when 50%+ nearby
                size_sig      = min(1.0, ratio / 0.25)       # full score when 25%+ small objs
                scores[tool]  = round(proximity_sig * size_sig, 4)

            elif tool == "no_action":
                scores[tool] = 1.0

        # Post-score correction: reduce morph_close when fill_holes is a viable alternative.
        # This prevents morph_close from winning purely on fragmentation when the
        # primary signal (small holes in intact glands) clearly favours fill_holes.
        if "morph_close" in scores and "fill_holes" in scores:
            fh = scores["fill_holes"]
            mc = scores["morph_close"]
            if fh > 0.30 and mc > fh:
                scores["morph_close"] = round(mc * _MORPH_CLOSE_PENALTY_VS_SPECIFIC, 4)

        return scores

    # ── Stage 4: select action ────────────────────────────────────────────────

    def select_action(
        self,
        tool_scores: dict[str, float],
        issue_severity: float = 0.0,
    ) -> tuple[str, float]:
        """
        Pick the highest-scored tool and compute confidence.

        Confidence combines issue severity and tool score margin (clarity).
        """
        if not tool_scores:
            return "no_action", 0.0

        best_tool = max(tool_scores, key=tool_scores.__getitem__)
        sorted_ts = sorted(tool_scores.values(), reverse=True)

        if len(sorted_ts) == 1:
            tool_clarity = sorted_ts[0]
        else:
            margin = sorted_ts[0] - sorted_ts[1]
            tool_clarity = sorted_ts[0] * 0.6 + margin * 0.4

        confidence = (issue_severity + tool_clarity) / 2.0
        return best_tool, round(confidence, 4)

    # ── Stage 5: multi-step planning ─────────────────────────────────────────

    def _get_action_sequence(
        self,
        primary: str,
        all_scores: dict[str, float],
        single_action: str,
    ) -> list[str]:
        """
        Return a 1- or 2-step action sequence.

        Checks whether a secondary issue is significant enough to warrant a
        second corrective step. If no multi-step rule matches the
        (primary, secondary) pair, returns [single_action].
        """
        if single_action == "no_action":
            return ["no_action"]

        secondary: str | None = None
        secondary_score: float = 0.0
        for issue, score in sorted(all_scores.items(), key=lambda x: -x[1]):
            if issue != primary and score > 0:
                secondary = issue
                secondary_score = score
                break

        if secondary is not None:
            for p, s, min_score, sequence in _MULTI_STEP_RULES:
                if p == primary and s == secondary and secondary_score >= min_score:
                    return list(sequence)

        return [single_action]

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def decide(self, features: dict) -> dict:
        """
        Run all reasoning stages and return a structured decision dict.

        Returns
        -------
        dict with keys:
            issue, pathology_interpretation, candidate_tools,
            tool_scores, selected_action, action_sequence,
            reason, confidence, score_breakdown
        """
        issue, severity       = self.interpret_features(features)
        all_scores            = self._compute_issue_scores(features)
        candidates            = self.get_candidate_tools(issue)
        tool_scores           = self.score_tools(candidates, features)

        # Apply session-level diversity discount (encourages exploration).
        for tool in list(tool_scores.keys()):
            if tool != "no_action":
                tool_scores[tool] = round(
                    tool_scores[tool] * self._diversity_discount(tool), 4
                )

        selected_action, conf = self.select_action(tool_scores, issue_severity=severity)
        action_sequence       = self._get_action_sequence(issue, all_scores, selected_action)

        return {
            "issue":                    issue,
            "pathology_interpretation": self._PATHOLOGY[issue],
            "candidate_tools":          candidates,
            "tool_scores":              tool_scores,
            "selected_action":          action_sequence[0],
            "action_sequence":          action_sequence,
            "reason":                   _build_reason(issue, action_sequence, features, severity),
            "confidence":               conf,
            "score_breakdown":          _build_score_breakdown(tool_scores, action_sequence[0]),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_score_breakdown(tool_scores: dict[str, float], winner: str) -> dict:
    """Summarise winning margin and runner-up for logging."""
    sorted_scores = sorted(tool_scores.items(), key=lambda x: -x[1])
    winner_score  = tool_scores.get(winner, 0.0)
    if len(sorted_scores) >= 2:
        runner_up       = sorted_scores[1][0] if sorted_scores[0][0] == winner else sorted_scores[0][0]
        runner_up_score = tool_scores.get(runner_up, 0.0)
        margin          = winner_score - runner_up_score
    else:
        runner_up       = None
        runner_up_score = 0.0
        margin          = winner_score
    return {
        "winner":         winner,
        "winner_score":   round(winner_score, 4),
        "runner_up":      runner_up,
        "runner_up_score": round(runner_up_score, 4),
        "margin":         round(margin, 4),
        "all_scores":     dict(sorted_scores),
    }


def _build_reason(
    issue: str,
    action_sequence: list[str],
    features: dict,
    severity: float,
) -> str:
    """One-sentence human-readable justification for the decision."""
    action = " -> ".join(action_sequence) if len(action_sequence) > 1 else action_sequence[0]
    multi  = len(action_sequence) > 1

    if issue == "noisy":
        reason = (
            f"small_object_ratio={features['small_object_ratio']}, "
            f"size_cv={features.get('size_cv','?')} indicate debris mixed with real glands; "
            f"applying {action} (severity={severity:.2f})."
        )
        if multi:
            reason += " Second step closes any fragmentation left after debris removal."
        return reason

    if issue == "holey":
        h_size    = features["avg_hole_size"]
        avg_size  = features["avg_object_size"]
        hole_ratio = h_size / max(avg_size, 1.0)
        n_obj      = features["num_objects"]
        if action == "fill_holes" and hole_ratio < _FILL_HOLES_SMALL_RATIO and n_obj <= 4:
            sub_type = f"small artifact holes (hole_ratio={hole_ratio:.3f} < {_FILL_HOLES_SMALL_RATIO}, compact gland)"
        else:
            sub_type = f"structural gaps or fragmentation (hole_ratio={hole_ratio:.3f}, n_obj={n_obj})"
        return (
            f"holes_count={features['holes_count']}, avg_hole_size={h_size}px — {sub_type}; "
            f"applying {action} (severity={severity:.2f})."
        )

    if issue == "merged":
        reason = (
            f"compactness={features['compactness']} well below threshold, "
            f"avg_object_size={features['avg_object_size']}px — glands appear fully fused; "
            f"applying {action} (severity={severity:.2f})."
        )
        if multi:
            reason += " Bridge-opening precedes watershed to widen seed separation."
        return reason

    if issue == "thin_bridge":
        reason = (
            f"compactness={features['compactness']}, "
            f"boundary_roughness={features['boundary_roughness']} indicate a narrow neck connection; "
            f"applying {action} (severity={severity:.2f})."
        )
        if multi:
            reason += " Watershed follows open to cleanly split the separated blobs."
        return reason

    if issue == "over_segmented":
        reason = (
            f"num_objects={features['num_objects']} with low size_cv={features.get('size_cv','?')} "
            f"suggests a gland was fragmented into similarly-sized pieces; "
            f"applying {action} (severity={severity:.2f})."
        )
        if multi:
            reason += " Second step fills any voids exposed after closing."
        return reason

    if issue == "under_segmented":
        n = features["num_objects"]
        d = features["mask_density"]
        area = features.get("total_mask_area", features["avg_object_size"] * n)
        proximity = d * (n - 1) / _DILATION_MERGE_PROXIMITY if n > 1 else 0.0
        merge_note = (
            f"merge risk low (proximity={proximity:.2f})"
            if proximity < 0.5
            else f"merge risk elevated (proximity={proximity:.2f})"
        )
        reason = (
            f"mask_density={d} is very low, "
            f"avg_area/object={area/max(n,1):.0f}px — gland boundaries too tight; "
            f"{merge_note}; applying {action} (severity={severity:.2f})."
        )
        if multi:
            reason += " Boundary smoothing follows dilation to remove expansion artifacts."
        return reason

    if issue == "rough_boundary":
        reason = (
            f"boundary_roughness={features['boundary_roughness']} is high while "
            f"compactness={features['compactness']} is acceptable — boundary needs smoothing; "
            f"applying {action} (severity={severity:.2f})."
        )
        if multi:
            reason += " Hole-fill follows closing to repair any voids opened during smoothing."
        return reason

    return "All features within acceptable range; no post-processing needed."
