# Coreset, Memory Bank Size, and Label-Regime Tradeoffs

## 1) Why this note exists

PatchCore can feel unintuitive at first for three reasons:

1. the training data are "only nominal images"
2. the memory bank still becomes very large
3. the method adds a coreset selection stage that looks like an extra approximation on top of another approximation

Then a practical deployment question makes things even less clean:

- what if the images I am calling "nominal" are not guaranteed to stay good?
- what if some parts fail later?
- what if negative / failure cases are actually easier to curate than truly trustworthy nominal coverage?

This note explains those issues in one place.

## 2) What the PatchCore coreset is doing

### 2.1 The key point

PatchCore does **not** store one vector per image.

It stores a **patch-level memory bank**.

That means each nominal image is converted into many local feature vectors:

- one feature vector per spatial patch location
- often from multiple feature layers
- often after resizing those layers to a shared grid

In this repo, that happens in:

- [embedding.py](src/patchcore/embedding.py)
- [extract.py](src/patchcore/extract.py)
- [patchcore.py](src/patchcore/patchcore.py)

So the memory bank size scales roughly like:

```text
number of training images × number of patch locations per image
```

not:

```text
number of training images
```

### 2.2 Why that is useful

PatchCore wants high recall. At inference, a test patch is scored by its distance to the nearest nominal patch in the bank. The larger and more representative the bank, the better the system can cover:

- normal texture variation
- illumination drift
- pose / alignment nuisance
- acceptable manufacturing tolerance
- recurring benign artifacts

This is why Roth et al. describe PatchCore as using a **maximally representative memory bank of nominal patch features** rather than a small parametric normal model.

Source:

- Roth et al., CVPR 2022  
  https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html

### 2.3 Why a coreset is needed

If you store every patch from every nominal image, the bank can become very large very quickly.

Example intuition:

- image size `256`
- layer-2 feature grid around `32 x 32`
- that is already about `1024` patch vectors per image
- `1000` nominal images becomes about `1,024,000` patch vectors

If each patch embedding is a few hundred dimensions, that is already a large bank for:

- RAM / disk
- nearest-neighbor latency
- downstream retrieval operations

PatchCore therefore uses a **coreset approximation**:

- keep a subset of nominal patch embeddings
- choose them to preserve geometric coverage of the full nominal cloud
- then do nearest-neighbor scoring against that subset

In this repo, that is implemented in:

- [coreset.py](src/patchcore/coreset.py)

### 2.4 What algorithm is being approximated

The common intuition is **k-center / covering-set selection**:

- choose representatives that are far from the already chosen points
- keep doing that until the subset covers the space well

That is the same family of geometry used in the k-center greedy coreset literature that PatchCore cites and inherits from.

Source:

- Sener and Savarese, *Active Learning for Convolutional Neural Networks: A Core-Set Approach*, ICLR 2018  
  https://openreview.net/forum?id=H1aIuk-RW

### 2.5 What coreset buys you

It trades:

- a small approximation error

for:

- much smaller memory
- much faster nearest-neighbor search
- lower deployment cost

The coreset is therefore not a random compression step. It is a **coverage-preserving memory reduction step**.

## 3) Why the memory bank is large even with only nominal images

### 3.1 Because anomaly detection is a coverage problem

This is the key conceptual answer.

For PatchCore, nominal training is not "learning one prototype for good parts." It is more like:

- sample the manifold of acceptable normal appearance

If your normal data cover only a thin slice of acceptable variation, then many perfectly good future parts will look anomalous.

So the memory bank is large because:

- normality in manufacturing is often broad and multi-modal
- the model is trying to remember local appearance support, not just global class identity

### 3.2 Why this differs from a classifier

A classifier compresses examples into decision boundaries between classes.

PatchCore instead behaves more like:

- a dense library of what acceptable local appearance looks like

That usually needs more memory than a compact classifier, especially if you want:

- localization
- low false positives
- few-shot cold start

### 3.3 The patch explosion is intentional

The bank is big partly because the model is local by design. If you instead store one global image vector per image, you reduce memory drastically, but you usually lose:

- fine-grained localization
- sensitivity to small defects
- nearest-match explanations at the patch level

So the large bank is not accidental. It is the price paid for:

- local matching
- high recall
- interpretability

## 4) Is there anything we can do "around" the coreset?

Yes, but each workaround changes the tradeoff.

### 4.1 Keep the full bank, use faster indexing

Instead of shrinking the bank, you can keep more of it and make search cheaper using:

- approximate nearest-neighbor indexes
- FAISS-like indexing

This keeps more nominal detail, but shifts the complexity into infrastructure.

### 4.2 Increase coreset aggressiveness

You can use a smaller coreset ratio.

This reduces:

- memory
- latency

But increases the risk of:

- false positives on legitimate but rare normal variation
- missing subtle coverage modes

### 4.3 Use prototypes or clusters instead of raw patch memory

You can cluster nominal patches and store:

- centroids
- medoids
- multiple prototypes per normal mode

This is a more structured compression than raw coreset subsampling.

Potential benefit:

- smaller bank
- smoother behavior

Potential cost:

- prototype averaging can blur rare but legitimate normal modes

### 4.4 Reduce the patch count directly

You can reduce bank size by:

- smaller input resolution
- later feature layers
- larger patch stride / coarser spatial grid
- ROI cropping to the relevant component region only

This is often the cleanest way to shrink memory, but it may reduce sensitivity to small defects.

### 4.5 Use a two-stage system

A strong practical design is:

1. coarse global screening
2. local patch-memory scoring only for candidates

This gives better latency at the cost of added system complexity.

### 4.6 If you only need image-level anomaly flags, not localization

Then PatchCore may be overkill.

If you do not need:

- anomaly maps
- patch retrieval
- fine local matching

then a more global representation may be enough and may avoid much of the memory-bank explosion.

## 5) What if my "nominal" images are not really trustworthy?

This is the most important practical question.

There are actually **two different cases**, and they should not be mixed.

### 5.1 Case A: the image already contains a visible defect, but nobody knows yet

This is the **contaminated training data** problem:

- the training pool is supposed to be normal
- some anomalies are already inside it

This directly violates one-class industrial AD assumptions.

The literature is clear that contamination hurts anomaly detection.

Sources:

- Guo et al., *Unsupervised Anomaly Detection and Segmentation on Dirty Datasets*, Future Internet 2022  
  https://www.mdpi.com/1999-5903/14/3/86
- Wu et al., *Understanding and Mitigating Data Contamination in Deep Anomaly Detection: A Kernel-based Approach*, IJCAI 2022  
  https://www.ijcai.org/proceedings/2022/322

The important takeaway is:

- if contaminated anomalies are stored in the "normal" memory bank, they become part of the nominal support
- that can directly suppress anomaly scores for the wrong patterns

### 5.2 Case B: the image looks visually normal now, but the part fails later

This is **not exactly the same problem**.

If the failure precursor is not visually present at capture time, then a vision anomaly detector cannot be expected to infer it from appearance alone. That becomes closer to:

- early failure prediction
- prognostics
- survival / reliability modeling
- multimodal quality prediction

In other words:

- if there is no visible precursor, adding more anomaly-detection machinery does not solve the problem
- you need richer labels, later outcomes, or non-visual signals

So before changing the model, ask:

- is the eventual failure already visible in the image?
- or is this really a downstream latent-failure problem?

That distinction matters a lot.

## 6) What do I do operationally when labels are delayed or uncertain?

### 6.1 Do not dump all unlabeled images into the nominal bank

If labels are delayed, the safest workflow is not:

- "all unlabeled = normal"

Instead, treat recent unlabeled data as:

- provisional
- reviewable
- potentially contaminated

### 6.2 Maintain separate pools

A good operational split is:

- `trusted_nominal_train`
- `nominal_calibration`
- `quarantine_unlabeled`
- `known_failures`
- `held_out_eval`

The quarantine pool can later be promoted into either:

- trusted nominal
- known failure classes

### 6.3 Start with the cleanest seed possible

If you can get even a modest seed of trusted nominal examples, that is often better than using a much larger but dirty pool.

Why:

- contamination in a non-parametric memory bank is very direct
- once anomalies enter the bank, they become false support for normality

### 6.4 If some labeled failures exist, use them

This is exactly where hybrid methods become useful.

If you have:

- abundant unlabeled data
- some trusted failures

then the literature suggests you should not stay purely one-class forever.

Positive-unlabeled anomaly formulations are relevant here.

Source:

- Ju et al., *PUMAD: PU Metric Learning for Anomaly Detection*, Information Sciences 2020  
  https://doi.org/10.1016/j.ins.2020.03.021

The implication is:

- when normals are uncertain but some failures are known, a hybrid / PU-style treatment is often more defensible than pretending the whole pool is clean normal data

## 7) Why providing negatives can feel more natural than providing normals

That feeling is legitimate.

PatchCore is strongest when:

- normals are abundant
- failures are rare
- failures are heterogeneous

But some real operations look more like:

- we trust only a subset of normals
- we can curate recurring failures more easily than perfectly clean normals

In that regime, the problem starts to move away from pure one-class anomaly detection and toward:

- hybrid anomaly detection
- positive-unlabeled learning
- open-set defect recognition

That is exactly why the known-failure memory-bank direction is attractive.

## 8) What does the benchmark snapshot say?

We now have a benchmark snapshot on public `MVTec bottle` data:

- protocol doc: [HYBRID_EXPERIMENT_PLAN.md](HYBRID_EXPERIMENT_PLAN.md)
- generated summary path: `outputs/hybrid_benchmark/mvtec_bottle/summary.md`

The most important result is not the absolute AUROC. It is the tradeoff:

- adding a known-failure bank improves recognition of recurring known defect families
- but the current implementation also increases the tendency to force unseen failures into known classes

For example, with `support/class = 8` on the recorded run:

- `known_failure_recall` rises strongly over the nominal-only baseline
- `known_label_accuracy` becomes meaningful
- but `novel_as_known_rate` remains high

That is the central tradeoff:

- more negative / failure memory helps classification of recurring defects
- but can hurt open-set rejection if the reject mechanism is weak

So your intuition is correct:

- just adding negatives is not automatically better
- it helps some tasks and harms others unless the reject stage is strong

## 9) Practical recommendation

### 9.1 If your normals are clean and abundant

Stay close to classic PatchCore:

- nominal bank
- coreset
- threshold calibration

Add a known-failure bank only if you actually need recurring-defect recognition.

### 9.2 If your normals are somewhat dirty, but you have some known failures

Do not treat everything as nominal.

Prefer:

- trusted nominal seed
- quarantine pool
- known-failure memory bank
- explicit open-set rejection

This is the regime where the hybrid approach is most justified.

### 9.3 If eventual failure is delayed and not visually present yet

Do not expect image anomaly detection alone to solve it.

That is a different modeling problem:

- early-failure prediction
- prognostics
- multimodal quality forecasting

### 9.4 If memory is the main concern

Try, in order:

1. ROI restriction
2. lower image resolution
3. stronger coreset compression
4. ANN indexing
5. prototype / clustered bank

Each step is usually less damaging than immediately abandoning patch-level memory.

## 10) Bottom line

### What the coreset is

- a geometry-based representative subset of the nominal patch bank

### Why the bank is large

- because PatchCore stores local normal support, not one vector per image

### Can we get around it?

- partially, yes
- but every workaround trades coverage, localization, latency, or engineering complexity

### What if labels are delayed?

- if anomalies are already present visually, it is a contaminated-training problem
- if future failure is not visually present yet, it is a prognostics problem

### Why negatives sometimes feel more natural

- because real operations are often not pure one-class settings
- once known failures exist, hybrid open-set retrieval is often more realistic than pretending you only have clean normals

## 11) Source list

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022  
  https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html
- Sener and Savarese, *Active Learning for Convolutional Neural Networks: A Core-Set Approach*, ICLR 2018  
  https://openreview.net/forum?id=H1aIuk-RW
- Guo et al., *Unsupervised Anomaly Detection and Segmentation on Dirty Datasets*, Future Internet 2022  
  https://www.mdpi.com/1999-5903/14/3/86
- Wu et al., *Understanding and Mitigating Data Contamination in Deep Anomaly Detection: A Kernel-based Approach*, IJCAI 2022  
  https://www.ijcai.org/proceedings/2022/322
- Ju et al., *PUMAD: PU Metric Learning for Anomaly Detection*, Information Sciences 2020  
  https://doi.org/10.1016/j.ins.2020.03.021
