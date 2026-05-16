# PatchCore (“Towards Total Recall…”) vs. LLM/VLM one-shot approaches

This note answers: **why use a PatchCore-style memory-bank anomaly detector** (Roth et al., CVPR’22) instead of asking a large model to “just look at the image and decide”? It also lays out the major *cons* and when an LLM/VLM is the better tool.

Below, “LLM one-shot” should be read as **a general-purpose LLM/VLM** (e.g., GPT-4/4o-style vision, Gemini, Claude w/ vision) prompted to classify/describe defects directly, without a dedicated retrieval/memory-bank mechanism.

---

## 0) The core distinction

**PatchCore** is fundamentally:
- a **nonparametric** model of *normal* (nominal-only training)
- implemented as **nearest-neighbor distance** in a deep feature space
- with a **memory bank** that explicitly represents the support of “normal” at the *patch* level.

A typical **LLM/VLM one-shot** is:
- a **parametric** model trained broadly on internet-scale data
- which performs **semantic inference** (often classification-by-description)
- with internal representations you don’t directly control, inspect, or calibrate.

This matters because industrial anomaly detection is usually:
- **open-set** (unknown defect families)
- **extreme class imbalance** (defects rare)
- **high cost of false negatives** (missed defect)
- and often has **nuisance variation** (lighting, pose, texture, process drift).

---

## 1) Where PatchCore is genuinely better (pros)

### 1.1 Cold start with *no defect labels*
Industrial reality: you can usually gather “good” examples easily; defect labels are sparse and constantly changing.

PatchCore works well in exactly that regime:
- Train on nominal-only.
- Detect “distance from the manifold of normal.”

A one-shot VLM can *sometimes* do okay zero-shot, but it’s not reliably anchored to your factory’s definition of “normal.”

### 1.2 Open-set behavior is natural
PatchCore doesn’t need to “know” defect types. Anything sufficiently far from nominal memory is suspicious.

One-shot LLM/VLM tends to:
- force a semantic label even when it shouldn’t,
- or confidently describe a defect-like story when the issue is simply distribution shift.

### 1.3 Strong localization for “where is the defect?”
PatchCore produces **pixel/patch-level anomaly maps** derived from patch embeddings.

In production QA, localization is often more valuable than a label:
- it enables cropping for downstream review,
- supports operator trust (“show me the evidence”),
- and reduces time-to-triage.

Many VLM one-shot flows can describe an image, but localization is either absent, indirect, or not calibrated.

### 1.4 Explicit memory = controllable trade-offs
PatchCore’s “knobs” are simple and legible:
- memory size (coverage vs. RAM)
- coreset selection (speed vs. fidelity)
- kNN metric / PCA whitening (robustness vs. sensitivity)

You can relate these to operational constraints (latency, storage, recall).

LLM one-shot tuning is comparatively opaque:
- prompt tweaks are brittle,
- fine-tuning is expensive and slow,
- and you often don’t get predictable behavior changes.

### 1.5 Easier to calibrate to extreme recall targets
The “Towards Total Recall” framing is literally: *prioritize not missing defects.*

In practice you’ll pick thresholds on validation data to hit a recall operating point.

LLM/VLM one-shot outputs are typically:
- not well-calibrated probabilities,
- sensitive to prompt wording,
- and hard to threshold consistently across products/lighting shifts.

### 1.6 Privacy / on-prem friendliness
PatchCore can run fully local with:
- a fixed pretrained backbone,
- local embedding extraction,
- local retrieval.

LLM/VLM one-shot in many orgs implies:
- external API calls,
- potential policy friction,
- higher costs,
- and data governance complexity.

---

## 2) Where LLM/VLM one-shot can be better (pros)

### 2.1 Semantic reasoning + instructions + context
If the task needs **understanding specifications** or text-like reasoning:
- “Is this scratch within tolerance?”
- “Does this connector look seated?”
- “Explain likely failure mode given this defect pattern.”

A VLM can integrate:
- natural-language requirements,
- multi-step reasoning,
- and broader world knowledge.

PatchCore will only say “this is unusual here,” not *what it is*.

### 2.2 Better at heterogeneous scenes (sometimes)
PatchCore shines when images are consistent (same product, viewpoint, lighting envelope).

For highly variable scenes (different objects, clutter, backgrounds), a VLM may generalize better out of the box—though recall/precision can be unpredictable.

### 2.3 Low engineering friction for quick triage prototypes
If you need an immediate demo:
- VLM one-shot can be “prompt + a few examples,”
- no pipeline, no memory bank, no feature extraction harness.

PatchCore requires:
- data collection discipline,
- backbone choice,
- memory/coreset tuning,
- evaluation protocol work.

---

## 3) PatchCore limitations (cons)

### 3.1 Sensitive to nuisance variation / domain shift
PatchCore is a distance-to-nominal method. If the embedding distribution shifts due to:
- lighting,
- lens change,
- focus,
- pose differences,
- new batch texture,

…it may fire as “anomaly” even when parts are acceptable.

Mitigations exist (capture discipline, expanded nominal coverage, multi-bank routing, whitening), but the risk is real.

### 3.2 Memory + retrieval cost at inference
PatchCore scoring requires repeated kNN queries.
Costs:
- RAM/SSD footprint for the memory bank
- ANN index complexity (FAISS/IVF/etc) if you scale
- runtime cost on edge devices

### 3.3 Doesn’t tell you *what* the defect is
PatchCore gives a score + localization. It does not natively provide:
- defect taxonomy,
- causal explanation,
- recommended fix.

You typically need a second stage for interpretation.

### 3.4 Can underperform when defects are “global” semantics
If the defect is a higher-level concept (wrong part installed, subtle assembly sequence issue) rather than a local texture/geometry deviation, PatchCore may miss it or produce weak signals.

---

## 4) LLM/VLM one-shot limitations (cons)

### 4.1 Calibration and reliability
Industrial QA wants consistent thresholding under distribution shift.

LLM/VLM outputs often vary materially with:
- prompt phrasing,
- image resolution / compression,
- hidden model updates,
- and context length.

### 4.2 Hallucination risk
A one-shot model can confidently “see” artifacts or interpret shadows as damage.

In high-recall settings, you can’t tolerate a model that invents plausible defect narratives; you need stable evidence (localization, distance metrics, reproducible scoring).

### 4.3 Tooling and governance
- API costs can be high at line rates.
- Data cannot always leave the factory network.
- Debugging failures is harder because the model is a black box.

### 4.4 “One model for everything” hides operational coupling
A general model may be convenient, but it couples many requirements:
- defect detection,
- classification,
- explanation,
- policy compliance,
- latency,
- update cadence.

PatchCore splits the system into explicit, swappable parts.

---

## 5) The hybrid recommendation (usually best)

For many production-ish workflows, the best design is:

**Stage A (high-recall detector):** PatchCore-style nominal memory bank
- Purpose: *don’t miss* unfamiliar issues; localize candidate regions.

**Stage B (interpretation/triage):** LLM/VLM on cropped patches + context
- Purpose: propose defect type, generate operator-friendly explanation, suggest next checks.

This hybrid is attractive because:
- PatchCore anchors recall and provides evidence.
- The LLM adds semantics and decision support.
- You can gate expensive LLM calls (only on high-scoring anomalies).

---

## 6) Decision checklist (fast)

Choose **PatchCore-first** if:
- defects are rare/unknown (open set)
- you have lots of nominal images
- you need localization
- you must run on-prem and calibrate thresholds
- your KPI is “never miss defects” more than “label the defect.”

Choose **VLM-first** if:
- the task is mostly semantic/spec reasoning
- scenes are highly heterogeneous
- you need quick prototype value and can tolerate inconsistency
- you have strong human-in-the-loop review anyway.

---

## 7) Connection to “Total Recall” as a general pattern

PatchCore’s memory bank is a literal “total recall” mechanism: the model’s decision is grounded in **explicit retrieved neighbors** from nominal memory.

This mirrors the prompting argument for LLMs:
- recall/retrieve relevant items first
- then answer/decide

The key difference is **which memory** you trust:
- PatchCore: your curated nominal dataset (operational ground truth)
- LLM/VLM: the model’s learned priors + whatever context you provide.

Both patterns aim to avoid the failure mode of **one-shot guessing without grounding**.
