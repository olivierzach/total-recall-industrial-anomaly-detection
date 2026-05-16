# PDR rigor guidance (draft note)

Context: guidance for making the TTRIAD PDR “greenlight-grade” (top-lab / Starlink-style).

## What a greenlight-grade PDR must contain

A PDR that asks an org to greenlight building the system must read like:

1) we understand the failure modes
2) we have a credible plan to de-risk them with measurable gates
3) we know what resources we need
4) we’ll produce auditable evidence

The existing PDR skeleton is necessary but not sufficient; to be greenlight-grade it needs:

### A) “Decision ask” up front (1 page)
- what approval you want (time, compute, data access, headcount)
- what you will deliver by each milestone date
- what decisions those deliverables unblock

### B) Quantified acceptance gates (non-negotiable)
Not “high recall,” but explicit operating-point commitments:
- operating point definition: “<= X alerts / 1k units” (or “<= Y FPR”)
- primary KPI: “Recall >= R% at that operating point”
- secondary: latency <= L ms/image; memory <= M GB
- stability: metric variation <= Δ across (3 seeds × 2 splits)
- shift suite floors: e.g. brightness ±10%, small translation, blur → recall drop <= δ
- defect lookup head: top-1 accuracy >= A% on known defects at fixed reject rate + unknown rejection ROC

### C) Complete evaluation loop specification
- exact splits and leakage checks
- thresholding procedure
- artifact logging requirements
- confidence intervals / statistical tests (paired deltas)

### D) Risk register with mitigations + experiments
Create an explicit table:

| Risk | Why it matters | How we detect it | Mitigation | Kill switch criterion |
|---|---|---|---|---|
| Lighting drift causes FP | line stops / inspection overload | shift suite + drift monitoring | ROI, normalization, recalib, multi-bank | if FP doubles under ±10% brightness |
| Router misroutes product modes | wrong nominal bank → FP | routing confusion matrix | explicit product ID routing | if routing acc < 99% |
| Defect lookup over-claims | wrong label harms ops | unknown rejection curve | reject option + calibration | if mislabel rate > X% |

### E) Resource plan + feasibility
- data volume required (nominal images per mode; defect exemplars per class)
- compute budget per experiment
- timeline + what can be parallelized
- integration costs (UI, storage, monitoring)

### F) Evidence appendix
The PDR should point to:
- reproducible scripts generating tables/plots/panels
- initial benchmark evidence (proxy datasets ok)
- a plan to replicate on the real station dataset

## Inputs needed to finalize into a review-board doc
- inspection budget (alerts per hour / per 1k units / target FPR)
- failure cost asymmetry (FN vs FP seriousness)
- latency + hardware target
- number of product/station modes
- whether defect labels exist
- org’s MDR/PDR expectations/template
