# Setting Up Hybrid Known-Failure Memory

This is the concrete setup path for turning the prototype into a workflow where you can:

- train on **nominal** images
- add **known failure** folders
- keep detecting **unknown anomalies**
- later promote a newly discovered anomaly into a new known failure class

## 1) Folder layout

Use a layout like this:

```text
your_hybrid_dataset/
  nominal/
    train/
      0001.png
      0002.png
      ...
    calibration/
      1001.png
      1002.png
      ...
  failures/
    crack/
      crack_0001.png
      crack_0002.png
      ...
    scratch/
      scratch_0001.png
      scratch_0002.png
      ...
    corrosion/
      corrosion_0001.png
      ...
```

Meaning:

- `nominal/train`
  - good images used to build the nominal PatchCore memory bank
- `nominal/calibration`
  - separate good images used to set the anomaly threshold
- `failures/<label>`
  - labeled support examples for each known failure mode

Keep `calibration/` separate from `train/`. If you calibrate on the same images used to fit the nominal bank, your false-positive threshold will look better than it really is.

## 2) Fit the artifact

Run:

```bash
.venv/bin/python hybrid_memory_demo/fit_folder_hybrid_memory.py \
  --data-root /path/to/your_hybrid_dataset \
  --out outputs/hybrid_memory_demo/your_line_v1 \
  --device cpu
```

This writes:

- `config.json`
- `nominal_memory.npy`
- `failure_descriptors.npy`
- `support_records.json`
- `thresholds.json`
- `backbone_state.pt`
- `artifact_info.json`
- `fit_summary.json`

## 3) What the model does at inference

For each uploaded image:

1. score against the nominal memory bank
2. if score is below threshold -> `normal`
3. if score is above threshold:
   - compare to labeled failure supports
   - if close enough to one class -> `known_failure`
   - otherwise -> `unknown_anomaly`

That is the key product behavior:

- novel anomaly stays novel
- repeated anomaly can be promoted into a labeled memory bank and recognized later

## 4) Promote a new failure mode

When operators discover a recurring new anomaly:

1. create a new folder under `failures/`
2. place representative examples there
3. refit the artifact

Example:

```text
your_hybrid_dataset/
  failures/
    crack/
    scratch/
    corrosion/
    bent_lead/
```

Then rerun the same fit command. The next artifact now knows `bent_lead` as a valid known failure class.

## 5) Data standards

Use these rules if you want the hybrid bank to behave like an ML system rather than a demo:

- Keep one artifact per product/camera/lighting setup unless you have evidence they transfer.
- Use at least 3-5 images per known failure class for a serious pilot.
- Do not mix discovery images into calibration unless they are truly nominal.
- Version the dataset root and artifact output together.
- Keep a holdout evaluation set outside `failures/` if you want honest metrics.

## 6) Good first production split

Recommended structure:

```text
line_a_cam_3/
  nominal/
    train/
    calibration/
  failures/
    crack/
    chip/
    contamination/
  eval/
    nominal/
    crack/
    chip/
    contamination/
    unseen_failure/
```

The current fitting CLI uses `nominal/` and `failures/`. You should still maintain `eval/` for measurement and regression checks.

## 7) Browser UI

The browser UI works directly from the saved artifact:

```bash
.venv/bin/python hybrid_memory_demo/web.py \
  --artifact-dir outputs/hybrid_memory_demo/your_line_v1 \
  --device cpu
```

If you do not provide `--examples-json`, the app still works for uploads. If you do provide an examples file, it also renders a browsable gallery.

## 8) Current demo limitations

The included MVTec bottle demo is useful for the product concept, but it is not a final classifier:

- only one MVTec category is currently stored locally in this repo
- failure-class separation is still fairly simple
- known-class accuracy depends heavily on support diversity

That means the right next step for your use case is not “tune bottle forever”, but “fit artifacts on your own nominal + known-failure folders”.
