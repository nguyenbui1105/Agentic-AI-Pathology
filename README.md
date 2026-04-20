# Agentic Pathology Post-Processing

**Agentic AI system for safe, structure-aware post-processing in medical image segmentation.**  
Improving imperfect model predictions — without retraining, without an LLM, and without ever making things worse.

---

## Example Output

![Agent post-processing results across selected cases](outputs/visualizations/combined.png)

*Hybrid agent corrects fragmented and holey predictions using multi-step reasoning and safe, validated post-processing.*

---

## Key Insight

> Segmentation models make predictable, correctable mistakes.  
> Instead of retraining the model, we reason about its output and fix it — systematically.  
> This transforms segmentation from a prediction task into a decision-making problem.

---

## Overview

Deep learning models for histopathology segmentation regularly produce imperfect masks:
- Internal holes from staining artifacts
- Fragmented glands broken into pieces
- Merged boundaries where two glands touch
- Scattered noise blobs around real structures

Standard post-processing applies the same operation to every mask. That doesn't work — the right fix depends on what's actually wrong.

This system **classifies each failure mode** and applies the correct, validated correction. If a correction doesn't improve the mask, it is automatically rolled back.

---

## Pipeline

```
H&E Image  →  Prediction Mask
                    │
                    ▼
          Feature Extraction
          (roughness, holes, compactness,
           fragment ratio, object size...)
                    │
                    ▼
            Reasoning Agent
          (classify issue → score tools
               → select action)
                    │
                    ▼
         Hybrid Tool Execution
         ┌──────────────────────┐
         │  Global Pipeline     │  ← noise, fragments, widespread holes
         │  Region Pipeline     │  ← per-gland: holes, shape, bridges
         └──────────────────────┘
                    │
                    ▼
          Step Validation
          (area · compactness · Dice)
                    │
          ┌─────────┴─────────┐
        Accept             Rollback
     refined mask       original mask
```

---

## How It Works

**1. Feature Extraction**  
Each mask is described by a feature vector: object count, compactness, boundary roughness, hole count, nearby fragment ratio, and more.

**2. Issue Detection**  
A deterministic scoring agent classifies the dominant failure mode — `noisy`, `holey`, `fragmented`, `merged`, `thin_bridge`, or `clean`.

**3. Tool Selection**  
The agent scores candidate tools against the feature vector and selects the best match with adaptive parameters (kernel sizes, gap thresholds, hole limits computed from the mask).

**4. Hybrid Execution**  
Two pipelines run in parallel:
- **Global** — whole-mask corrections (remove debris, reconnect fragments)
- **Region** — per-component corrections (fill holes, smooth boundaries, split merges)

The orchestrator routes to the best strategy based on issue spread across glands, or scores both and picks the winner.

**5. Safety Validation**  
Every applied step is checked against proxy signals (area ratio, compactness delta, Dice improvement). Three-tier acceptance:
- **Hard veto** — reject catastrophic changes (area collapse, object count collapse)
- **Dice-first** — accept if GT Dice improves ≥ 0.002
- **Score-gate** — accept if proxy score stays within 4% of baseline

If a step fails all three, the mask rolls back to the previous state. No harmful action is ever committed.

---

## Tools

| Tool | Corrects |
|---|---|
| `remove_small_objects` | Noise, debris blobs |
| `fill_holes` | Small internal holes |
| `morph_close` | Gaps, internal voids |
| `morph_open` | Spiky or jagged boundaries |
| `erosion` | Thin bridges between merged objects |
| `dilation` | Under-segmented, too-tight boundaries |
| `watershed_split` | Merged / touching glands |
| `connect_fragments` | Broken gland continuity |

---

## Results

Evaluated on **80 samples** across two test sets using the hybrid pipeline.

| | |
|---|---|
| Samples improved (Dice Δ > +0.002) | **40%** |
| Max single-sample Dice gain | **+0.0167** |
| Harmful regressions committed | **0** |
| Actions automatically rolled back | as needed |

Average Dice delta: **+0.0015** across all 80 samples.

The system prioritises reliability over aggressive optimisation — it acts only when the evidence supports it, and always preserves the ability to roll back.

---

## Example Cases

**Fragmented gland**
- Signal: `nearby_fragment_ratio = 0.83`, multiple disconnected components
- Action: `connect_fragments` — bridges satellite pieces to main body via distance transform
- Result: object count 3 → 1, Dice +0.013

**Noisy mask**
- Signal: `small_object_ratio = 0.82`, low average object size
- Action: `remove_small_objects` → `morph_close` (two-step global correction)
- Result: debris removed, main gland preserved, Dice +0.008

**Safety case — no action**
- Signal: issue confidence below threshold, proxy score does not improve
- Action: rollback, original mask kept
- Result: Dice unchanged — system correctly declined to act

---

## Why Not Just Use an LLM?

LLMs add latency, cost, and unpredictability. This system is fully deterministic — every decision traces back to a feature value and a scoring rule. That makes it:
- **Auditable** — you can explain exactly why each tool was chosen
- **Fast** — no inference calls, runs in seconds per image
- **Safe** — bounded by hard validation rules, not probabilistic outputs

---

## How to Run

**Install**
```bash
pip install -r requirements.txt
```

**Demo (synthetic cases)**
```bash
python main.py
```

**Evaluate on a dataset**
```bash
# Dataset layout: data/samples/{image/, mask/, pred/}
python scripts/evaluate_samples.py
```
Outputs: `outputs/data_samples_eval/full_report.csv` + visualizations in `outputs/visualizations/`

**Run tool tests**
```bash
python tests/test_tools.py
```

---

## Project Structure

```
.
├── main.py                         # Synthetic demo
├── src/
│   ├── pipeline.py                 # Global pipeline
│   ├── region_pipeline.py          # Region-aware pipeline
│   ├── hybrid_pipeline.py          # Hybrid orchestrator
│   ├── agent/
│   │   ├── reasoning.py            # PathologyReasoningAgent
│   │   └── region_features.py      # Per-region feature extraction
│   ├── features/extractor.py       # Global feature extraction
│   ├── tools/postprocessing.py     # All 8 tools
│   └── evaluation/metrics.py       # Dice / IoU
├── scripts/
│   ├── evaluate_samples.py         # Batch evaluation
│   └── evaluate_glas.py            # GlaS-format evaluation
├── tests/
│   ├── test_tools.py
│   ├── test_agent.py
│   ├── test_features.py
│   ├── test_region_pipeline.py
│   └── test_hybrid_pipeline.py
└── outputs/visualizations/         # Result images
```

---

## Next Steps

- **Image-guided refinement** — incorporate H&E texture features into the agent's decision
- **Learnable correction modules** — replace hand-crafted scoring rules with a lightweight trained scorer
- **Topology-aware modeling** — use persistent homology to detect and correct structural errors
- **Broader evaluation** — full GlaS, CRAG, and DigestPath benchmarks
- **WSI integration** — tiled inference pipeline for whole-slide images

---

## Stack

```
Python 3.10+  |  numpy  |  scipy  |  scikit-image  |  opencv  |  matplotlib
```
