# Adding an LLM/VLM to the existing two-head PatchCore + known-failure memory-bank system

**Context:** This repo already implements a *two-head* architecture:
- **Head A:** nominal-only PatchCore anomaly detection + localization.
- **Head B:** known-failure retrieval/classification against a defect memory bank, with a reject option.

This note describes a pragmatic **Stage C (LLM/VLM triage head)** that you can bolt on without disturbing the core high-recall retrieval pipeline.

The goal is not “replace PatchCore with a VLM.” The goal is:
- keep PatchCore as the **high-recall, evidence-grounded trigger + localizer**
- use an LLM/VLM for **semantics, operator-facing explanation, and workflow routing**
- **gate** LLM use so cost/latency is bounded and failure modes are contained.

---

## 1) Where the LLM fits (system sketch)

### 1.1 Recommended decision graph

1) **Head A (nominal anomaly score)**
- Input: full image
- Output: `anom_score`, `anom_map`, candidate regions (top-k patches / boxes)
- Decision: if `anom_score < tau_nominal` → **status=nominal** (stop)

2) **Head B (known failure retrieval)**
- Trigger: only if anomalous by Head A
- Input: anomalous regions (crops) and/or whole image embedding
- Output: `known_mode_topk`, `known_confidence`, exemplar neighbors
- Decision:
  - if `known_confidence >= tau_known` → **status=known_failure** with label + evidence
  - else → **status=unknown_anomaly** with evidence

3) **Stage C (LLM/VLM triage)**
- Trigger: (a) unknown anomalies, or (b) known failures where you want a human-facing explanation / ticket
- Input: small set of evidence artifacts (not raw firehose):
  - the original image
  - 1–3 anomaly crops (from Head A)
  - optional: overlay image (heatmap / bounding boxes)
  - retrieved exemplars from Head B (top-1 or top-3)
  - minimal metadata (station/camera/product/rev)
- Output (structured):
  - human-readable description
  - tentative defect taxonomy (or “uncertain”)
  - recommended next checks
  - routing (which queue / who to ping)

**Key principle:** the LLM should be downstream of an evidence-producing detector.

---

## 2) What the LLM should and should NOT be asked to do

### 2.1 Good LLM tasks
- **Explain**: “Describe what seems wrong, referencing the highlighted regions.”
- **Compare**: “Does this resemble any of the retrieved known-failure exemplars? If yes, why?”
- **Suggest next steps**: imaging checks, teardown suggestions, process suspects.
- **Generate artifacts**: ticket text, summary for Slack/Jira, structured JSON for dashboards.
- **Spec application (carefully)**: if you supply tolerances explicitly.

### 2.2 Avoid as the primary decision maker
Do *not* rely on VLM one-shot for:
- the initial anomaly trigger (Head A)
- open-set “is it anomalous?” decisions at extreme recall targets
- calibrated thresholding

Reason: calibration, brittleness, and hidden updates.

---

## 3) Evidence packaging (what to send to the LLM)

Think of this as building a small “case file.”

### 3.1 Minimal case file (recommended)
- `image_full`: the original image (possibly downscaled)
- `image_overlay`: same image with heatmap/boxes + top anomaly score(s)
- `crops[]`: 1–3 crops around highest anomaly regions
- `neighbors_nominal[]` (optional): 1–2 nearest nominal neighbors for the top anomalous crop
- `neighbors_known[]` (optional): 1–3 retrieved known-failure exemplars + their labels
- `metadata`: product/station/camera, timestamp, any process flags

### 3.2 Why crops matter
Crops:
- reduce token+bandwidth cost
- focus attention
- reduce hallucinations from irrelevant background

### 3.3 Don’t drown the model
The LLM is not your database. Send:
- at most a handful of images
- a small, deterministic text summary of numeric scores

If you want rich retrieval, do it *outside* the LLM (PatchCore/FAISS/etc), then summarize.

---

## 4) Prompting: enforce humility + grounding

### 4.1 System prompt guidelines
You want something like:
- “You must ground claims in the highlighted regions or retrieved exemplars.”
- “If you are uncertain, say so.”
- “Do not invent measurements.”

### 4.2 Suggested output schema
Use a JSON schema to make downstream integration stable:

```json
{
  "summary": "...",
  "status": "known_failure" | "unknown_anomaly" | "needs_more_info",
  "tentative_failure_mode": "..." ,
  "confidence": 0.0,
  "evidence": ["..."],
  "recommended_actions": ["..."],
  "routing": {"queue": "...", "priority": "low|med|high"}
}
```

Treat this as *advisory*; the authoritative status remains Head A/B.

---

## 5) Gating + cost/latency control

### 5.1 When to call the LLM
Recommended triggers:
- **Unknown anomalies** above a severity threshold (e.g., top 0.1% of anomaly scores)
- **Clusters** of similar unknown anomalies (call LLM once per cluster exemplar)
- **Newly promoted failure mode candidates** (after unsupervised clustering)

Avoid calling the LLM for every anomalous image.

### 5.2 Deterministic caching
Cache by content hash of the case file:
- hash(original image bytes + crop coords + retrieved exemplar ids + prompt version)

This prevents repeated spend and improves reproducibility.

---

## 6) Failure modes and mitigations

### 6.1 Hallucinated defect descriptions
Mitigations:
- always include overlay/crops
- force citations (“point to crop #2, region upper-left”)
- allow “uncertain” output

### 6.2 Over-trusting retrieved known failures
Mitigations:
- keep the **reject option** upstream
- tell the LLM: “retrieval is suggestive, not definitive”
- include similarity scores so it can notice weak matches

### 6.3 Spec reasoning without actual measurements
Mitigations:
- if tolerances matter, provide explicit measurements from CV/geometry tools
- otherwise restrict the LLM to qualitative judgments

### 6.4 Data governance
If images are sensitive:
- run local models or on-prem endpoints
- strip metadata; avoid sending serials
- store only hashes / minimal logs

---

## 7) Evaluation plan for the LLM stage

You should not evaluate Stage C like a detector. Evaluate it like a **decision-support component**.

Metrics ideas:
- **Operator time-to-triage** reduction (A/B test)
- **Ticket quality**: completeness, correctness, actionability
- **Routing accuracy**: correct queue/priority
- **Novel mode discovery throughput**: how quickly unknown anomalies turn into stable labeled modes

Critically: Stage C should never reduce Head A recall.

---

## 8) Concrete implementation plan in this repo

### 8.1 New module boundaries (suggested)
- `src/triage/`:
  - `casefile.py`: build case file (crops, overlays, neighbor thumbnails)
  - `client.py`: LLM/VLM API wrapper (or local model wrapper)
  - `schema.py`: pydantic/dataclass for structured output
  - `cache.py`: content-hash cache

### 8.2 Add a CLI entrypoint
Example:
- `scripts/triage_anomalies.py --artifact <...> --images <dir> --out <jsonl>`

Flow:
1) run Head A/B scoring (existing)
2) for selected anomalies, build case file
3) call LLM
4) persist JSONL: one record per image with Head A/B + Stage C outputs

### 8.3 Integrate into the demo UI
If you want it in `hybrid_memory_demo/`:
- add a “Generate triage report” button
- show:
  - LLM summary
  - recommended actions
  - and *always* the retrieved exemplars + crops so humans can audit

---

## 9) Recommendation: keep responsibilities clean

A robust division of labor looks like:
- **PatchCore (Head A):** anomaly trigger + localization
- **Defect bank (Head B):** known-failure retrieval with reject
- **LLM (Stage C):** interpretation + communication + workflow

If you keep Stage C advisory and evidence-grounded, you get most of the benefits of an LLM with minimal risk to the core “total recall” objective.
