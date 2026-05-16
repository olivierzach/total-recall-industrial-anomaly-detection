# Research Note: Hybrid "Negative" Memory Banks for Industrial Anomaly Detection

## 1) Executive answer

Short answer: **yes, the idea is feasible and justified**, but only if it is framed correctly.

The right framing is **not**:

- "replace nominal-only anomaly detection with a closed-set defect classifier"

The right framing is:

- **retain a nominal memory bank for open-set anomaly detection**
- **add a labeled failure memory bank for retrieval and recognition of recurring defect modes**
- **keep an explicit reject option for novel anomalies**

That hybrid framing is well supported by adjacent literature even though **"negative memory bank" is not a standard canonical term** in the anomaly-detection literature. The closest established research areas are:

1. **memory-bank anomaly detection** in industrial inspection
2. **open-world / open-set recognition**
3. **few-shot prototype and metric-based classification**
4. **few-shot visual anomaly detection with reference images**
5. **open-set supervised anomaly detection with a few labeled anomalies**

This note is therefore partly a literature review and partly a **research synthesis / inference** built from those strands.

## 2) Why this question matters

PatchCore and related cold-start industrial anomaly detectors solve an important problem:

- train from **normal data only**
- flag samples that are far from normal memory
- localize suspicious regions

But production inspection systems often need more than binary anomaly detection. Once a line has run for some time, operators usually accumulate a small but valuable archive of recurring failures:

- cracked housings
- contamination
- bent leads
- missing components
- weld defects
- surface scratches

At that point, a practical question appears:

- can the system do more than say "off nominal"?
- can it say "this looks like the failure family we saw last month"?
- can it still reject a truly new failure mode instead of over-forcing it into an existing class?

That is exactly where a **known-failure memory bank** becomes attractive.

## 3) What the literature already says

### 3.1 PatchCore already establishes the memory-bank side

PatchCore is the baseline anchor for this repo. Roth et al. show that a **memory bank of nominal patch features** plus nearest-neighbor scoring is extremely effective for cold-start industrial anomaly detection. The important implication for this note is structural:

- memory-bank methods are already legitimate, high-performing tools in industrial inspection
- nearest-neighbor distance in pretrained feature space is already a strong anomaly signal

Reference:

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022  
  https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html

### 3.2 Open-world recognition justifies "recognize known classes, reject unknowns"

Bendale and Boult formalized **open-world recognition**: a deployed system must both recognize known classes and reject unknown ones. Their Nearest Non-Outlier (NNO) result is especially relevant because it shows that **distance-based recognition with rejection** is not an ad hoc engineering trick; it has a principled role in open-world settings.

This is directly aligned with the hybrid defect-bank idea:

- known defect classes should be retrievable / classifiable
- unknown defects should remain rejectable

Reference:

- Bendale and Boult, *Towards Open World Recognition*, CVPR 2015  
  https://openaccess.thecvf.com/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html

### 3.3 Open-set deep recognition says a reject option is mandatory

The OpenMax work argues that deep closed-set classifiers are structurally bad at unknown rejection because they are forced to choose one known class. For a failure-memory system this matters a lot: if you add labeled defect classes without a reject mechanism, you create pressure to **mislabel novel failures as familiar ones**.

Therefore, the literature does **not** justify "just train a classifier on known failures." It justifies:

- classification or retrieval **with unknown rejection**

Reference:

- Bendale and Boult, *Towards Open Set Deep Networks*, CVPR 2016  
  https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html

### 3.4 Prototypical / metric few-shot learning justifies class memories from very few examples

Prototypical Networks are not anomaly-detection papers, but they are highly relevant. They show that for novel classes with few examples, **distance to class prototypes in embedding space** is a strong inductive bias.

That supports an important engineering conclusion:

- you do not necessarily need a large supervised classifier
- a small defect bank with prototypes or nearest neighbors is a credible first-class design

Reference:

- Snell et al., *Prototypical Networks for Few-shot Learning*, NeurIPS 2017  
  https://papers.nips.cc/paper/6996-prototypical-networks-for-fe

### 3.5 Deep nearest-neighbor OOD detection strengthens the non-parametric case

Sun et al. show that **non-parametric nearest-neighbor distance** is highly effective for out-of-distribution detection and can outperform strong parametric baselines. The relevance here is conceptual:

- if nearest-neighbor distance is already useful for OOD rejection
- and memory-bank distance is already useful for industrial anomaly detection
- then a hybrid nearest-neighbor architecture over both normal and labeled defect memory is technically plausible

Reference:

- Sun et al., *Out-of-Distribution Detection with Deep Nearest Neighbors*, ICML 2022  
  https://proceedings.mlr.press/v162/sun22d.html

### 3.6 Open-set supervised anomaly detection is the closest direct precedent

Ding et al. explicitly study the setting where **a few labeled anomalies are available**, but the system must still detect unseen anomalies. This is extremely close to the "known failure memory bank" problem.

Their framing is important because it rejects the false dichotomy between:

- purely one-class anomaly detection
- purely closed-set anomaly classification

Instead, they target both:

- seen anomalies
- unseen anomalies

That is the strongest direct literature-level justification for the hybrid idea.

Reference:

- Ding et al., *Catching Both Gray and Black Swans: Open-Set Supervised Anomaly Detection*, CVPR 2022  
  https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.html

### 3.7 Recent few-shot anomaly papers support reference-based inspection with margins and patch matching

Recent few-shot anomaly detection papers strengthen the case for **reference-based**, often **training-free or lightly-trained**, inference in industrial settings:

- **PromptAD** shows that even when real anomaly images are unavailable, constructing negative separation and enforcing an explicit margin is useful.
- **AnomalyDINO** shows that patch-level deep nearest-neighbor matching with strong pretrained visual features is enough to be a competitive few-shot anomaly detector.
- **UniVAD** pushes the few-shot reference idea toward a more general cross-domain setting.
- **InCTRL** argues that few-shot prompt images can support more general anomaly detection via residual comparison.

These papers are not all doing labeled defect retrieval, but collectively they support the broader claim that **small support sets + strong embeddings + distance-based scoring** are now a serious regime, not a toy one.

References:

- Li et al., *PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection*, CVPR 2024 / arXiv 2024  
  https://arxiv.org/abs/2404.05231
- Damm et al., *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2*, arXiv 2024  
  https://arxiv.org/abs/2405.14529
- Gu et al., *UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection*, arXiv 2024 / CVPR 2025  
  https://arxiv.org/abs/2412.03342
- Zhu and Pang, *Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts*, CVPR 2024  
  https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Toward_Generalist_Anomaly_Detection_via_In-context_Residual_Learning_with_Few-shot_CVPR_2024_paper.html

## 4) Research synthesis: what is actually justified?

The following claims are **inferences from the sources above**, not direct claims made by any single paper.

### 4.1 What is justified

The literature supports a system with three outputs:

1. `normal`
2. `known_failure(class_k)`
3. `unknown_anomaly`

That system is justified because:

- PatchCore-style memory banks justify nearest-neighbor anomaly scoring from normal data
- open-world / open-set recognition justifies distance-based rejection instead of forced classification
- few-shot metric/prototype learning justifies small labeled class memories
- open-set supervised anomaly detection justifies using some labeled anomalies while preserving sensitivity to unseen ones
- recent few-shot anomaly papers justify support-set-based inspection with strong pretrained features

### 4.2 What is **not** justified

The literature does **not** justify:

- collapsing anomaly detection into a pure closed-set defect classifier
- assuming every future anomaly belongs to a known failure class
- using raw nearest-neighbor class assignment without a reject threshold or margin test
- evaluating only top-1 class accuracy while ignoring novel-anomaly rejection

In other words, a negative memory bank is justified only as a **hybrid open-world inspection system**, not as a conventional multiclass classifier.

## 5) Recommended system formulation

### 5.1 Stage A: nominal gate

Keep the current PatchCore-style detector:

- input image -> patch embeddings
- compare against nominal patch bank
- produce:
  - image anomaly score
  - patch map

This stage answers:

- does the sample look normal enough to stop here?

### 5.2 Stage B: known-failure retrieval head

If Stage A says the image is anomalous, compute a second descriptor aimed at failure recognition:

- top anomalous patches
- pooled local descriptor
- optionally global image embedding
- optionally component-aware descriptor if the part has structure

Then search a labeled defect bank:

- nearest neighbors over individual support examples, or
- per-class prototypes, or
- both

Return:

- top-`k` support matches
- best class
- best distance
- second-best distance
- margin ratio

### 5.3 Stage C: reject decision

Only emit `known_failure(class_k)` when:

- anomaly score is above the anomaly gate
- best defect-bank distance is below a class-match threshold
- best-vs-second-best margin is large enough

Otherwise emit:

- `unknown_anomaly`

This is the critical open-set safeguard.

## 6) Why this is feasible in practice

### 6.1 Recurring failures are common in real lines

Many industrial defects recur in families rather than as one-off unique events. If the same failure family returns:

- scratch geometry
- contamination pattern
- missing component
- bent lead
- broken seal

then a memory bank should be able to retrieve similar historical cases.

### 6.2 Pretrained embeddings are already strong enough

The recent few-shot anomaly literature is increasingly showing that strong pretrained features plus matching can work with little or no extra training. This lowers the barrier to a negative-bank system:

- you can start non-parametrically
- you do not need a large defect-labeled dataset on day one

### 6.3 Retrieval is operationally attractive

Compared with a monolithic classifier, a memory bank has attractive operational properties:

- easy to add a new failure family
- easy to inspect the support evidence behind a prediction
- easy to retain provenance
- easier debugging when a class is confused

That last point matters in QA settings where human trust is important.

## 7) Why this is risky

### 7.1 Failure classes are often not clean classes

A "defect type" is not always visually compact. Some labels are process labels rather than appearance labels. Example:

- `contamination` may include oil, dust, residue, glue, and smearing

If a class is heterogeneous, nearest-neighbor retrieval may still be useful, but clean classification becomes harder.

### 7.2 Novelty never disappears

The open-set problem remains. Even with a rich defect bank, the system must still be allowed to say:

- anomalous, but unlike stored failures

### 7.3 Bank growth can hurt without curation

As the defect bank grows:

- near-duplicate supports can dominate retrieval
- mislabeled supports poison prototypes
- camera drift can make the bank inconsistent
- class imbalance can bias nearest-neighbor outcomes

This means the bank should be treated as a maintained artifact, not a raw folder dump forever.

### 7.4 Product and camera specificity still matter

The literature on industrial AD remains strongly consistent with:

- one product / camera / geometry / lighting setup is usually safer than over-sharing one model

So negative-bank systems should usually be versioned per deployment slice.

## 8) Strong recommendation: use "known-failure memory" rather than "negative memory"

In machine learning, the word **negative** is overloaded:

- negative examples in binary classification
- hard negatives in contrastive learning
- negative prompts in VLM work

For this repo and for production communication, the clearer terminology is:

- **nominal memory bank**
- **known-failure memory bank**
- **unknown anomaly**

That vocabulary aligns better with the actual operational semantics.

## 9) Research-grade experimental protocol

If this repo wants to claim a serious result on this direction, evaluation should separate four tasks.

### 9.1 Task 1: normal vs anomalous

Metrics:

- image AUROC
- AUPRC
- Recall@FPR
- calibration on held-out nominal days/shifts

This checks whether adding the defect bank damages the original anomaly detector.

### 9.2 Task 2: seen anomaly vs unseen anomaly

Metrics:

- AUROC or macro-F1 for `known_failure` vs `unknown_anomaly`
- false-known rate on truly unseen anomalies

This is the key open-set question.

### 9.3 Task 3: known-failure classification

Metrics:

- top-1 accuracy
- top-`k` accuracy
- macro-F1
- confusion matrix by defect family

This should be computed **only on anomalous samples belonging to known classes**.

### 9.4 Task 4: retrieval quality

Metrics:

- Recall@`k`
- mAP
- class-consistent nearest-neighbor rate

In practice, retrieval quality may matter more than hard classification quality.

## 10) Recommended ablations

If this becomes a serious research branch for the repo, the minimum ablation set should be:

1. **support size per defect class**
2. **individual-instance bank vs class prototypes**
3. **global image embedding vs anomaly-weighted local descriptor**
4. **nominal-only gate vs hybrid gate**
5. **with and without reject margin**
6. **with and without patch-level support pooling**
7. **backbone choice**
8. **domain shift across day/shift/camera**
9. **incremental bank growth over time**

## 11) Recommended initial implementation path

The most defensible implementation path is staged.

### Phase 1: non-parametric hybrid retrieval

- keep PatchCore nominal detector
- add labeled known-failure support bank
- build an anomaly-weighted descriptor from top anomalous patches
- classify with nearest-neighbor or class-prototype distance
- reject by threshold + margin

This is the lowest-risk path and is already the closest to the literature synthesis above.

### Phase 2: prototype refinement

Once you have enough labeled failures:

- learn class prototypes
- cluster each failure family to allow sub-modes
- calibrate per-class thresholds

This handles heterogeneous classes better than a single raw bank.

### Phase 3: metric learning or light adaptation

Only after enough labeled failures exist should you consider:

- contrastive learning
- defect-aware fine-tuning
- prototype learning
- open-set loss functions

At small data scale, non-parametric retrieval is usually the better inductive bias.

## 12) Bottom-line position

### My conclusion

The hybrid known-failure memory-bank idea is:

- **feasible**
- **scientifically justified**
- **practically valuable**
- **more defensible than trying to jump directly to a closed-set defect classifier**

However, it is justified only under a strong requirement:

- the system must remain **open-set**

So the scientifically sound formulation is:

> Use a nominal memory bank for anomaly detection, then use a labeled known-failure memory bank for retrieval and recognition of recurring defect families, while preserving explicit rejection for novel anomalies.

That is a credible research direction for this repo.

## 13) Source list

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022  
  https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html
- Bendale and Boult, *Towards Open World Recognition*, CVPR 2015  
  https://openaccess.thecvf.com/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html
- Bendale and Boult, *Towards Open Set Deep Networks*, CVPR 2016  
  https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html
- Snell et al., *Prototypical Networks for Few-shot Learning*, NeurIPS 2017  
  https://papers.nips.cc/paper/6996-prototypical-networks-for-fe
- Sun et al., *Out-of-Distribution Detection with Deep Nearest Neighbors*, ICML 2022  
  https://proceedings.mlr.press/v162/sun22d.html
- Ding et al., *Catching Both Gray and Black Swans: Open-Set Supervised Anomaly Detection*, CVPR 2022  
  https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.html
- Li et al., *PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection*, arXiv 2024  
  https://arxiv.org/abs/2404.05231
- Damm et al., *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2*, arXiv 2024  
  https://arxiv.org/abs/2405.14529
- Zhu and Pang, *Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts*, CVPR 2024  
  https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Toward_Generalist_Anomaly_Detection_via_In-context_Residual_Learning_with_Few-shot_CVPR_2024_paper.html
- Gu et al., *UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection*, arXiv 2024 / CVPR 2025  
  https://arxiv.org/abs/2412.03342
