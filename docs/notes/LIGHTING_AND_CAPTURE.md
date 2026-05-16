# Lighting & capture for industrial anomaly detection (PatchCore-centric)

Industrial vision projects fail more often due to **capture variability** than due to model choice.
PatchCore is especially sensitive because it is essentially a **distance-to-nominal** method in embedding space.

This note is a practical guide to:
- the risks introduced by lighting/camera settings
- how to test those risks early (go/no-go)
- what to control in capture vs what to handle in modeling

---

## 0) The core failure mode

If the *embedding distribution* shifts due to lighting, PatchCore will interpret it as “out of distribution.”
That yields false positives (or unstable thresholds).

PatchCore is not unique here, but it makes the issue obvious:
- it scores anomalies by nearest-neighbor distance in feature space
- shifts that move features increase distances

---

## 1) Lighting risks (what changes, what it looks like)

### 1.1 Global illumination shift
Examples:
- different bulb intensity
- aging lights
- different exposure settings

Symptoms:
- score distribution on nominal images shifts upward
- heatmaps show broad, diffuse “everything is slightly anomalous”

### 1.2 White balance / color temperature drift
Examples:
- auto white balance
- different LED color temps

Symptoms:
- anomalies concentrate in color-sensitive textures
- thresholds do not transfer across shifts

### 1.3 Specular highlights and glare (the killer)
Examples:
- glossy plastics, solder joints, metal can caps
- small angle change → big highlight movement

Symptoms:
- heatmaps light up along edges/highlights
- false positives cluster in shiny regions

### 1.4 Shadows / vignetting
Examples:
- operator partially blocks light
- lens shading

Symptoms:
- anomalies look like blobs/gradients
- patch scores correlate with position

### 1.5 Motion blur
Examples:
- conveyor speed changes
- longer exposure due to dim lighting

Symptoms:
- edge regions pop as anomalies
- defects that depend on texture may become invisible

---

## 2) Camera settings that matter (in order)

### 2.1 Exposure control
If possible, **disable auto-exposure**.
Auto-exposure makes the input distribution non-stationary.

### 2.2 White balance
If possible, **disable auto white balance** or lock it to a fixed preset.

### 2.3 Focus / aperture / depth of field
- ensure the defect size is resolvable in pixels
- ensure depth of field covers expected pose variation

### 2.4 Polarization (often huge for glare)
A simple polarizer can reduce specular highlights.
If glare is a problem, ask about **cross-polarized lighting**.

### 2.5 Lens distortion / vignetting
These can be corrected, but the bigger win is consistent setup.

---

## 3) What to control with capture vs what to handle with ML

### 3.1 Capture control is the highest ROI
If you can fix lighting/camera settings, you reduce model complexity and improve reliability.

Good target: make the “good” distribution as stable as possible.

### 3.2 ML mitigations (when you can’t fully control)

#### Expand nominal coverage
Capture *nominal* images across the envelope of expected conditions:
- multiple shifts
- warm-up vs cold-start
- slightly different poses
- realistic variation in reflectance (dust, fingerprints)

PatchCore improves when its memory bank covers nuisance modes.

#### Routing / per-mode nominal banks
If you have multiple products/components or stations:
- maintain per-product or per-station nominal memory banks
- or use query-adaptive routing (IVF-style) to search within relevant nominal modes

#### Simple preprocessing
- brightness/contrast normalization (careful)
- color constancy / gray-world
- masking fixtures/background

#### Backbones and layers
Some backbones/layers are more sensitive to texture/illumination.
Ablate:
- different layer combinations
- ViT vs CNN backbones

---

## 4) Early go/no-go tests (do these in week 1)

You want to quickly answer: *Is lighting variability larger than defect signal?*

### 4.1 Nominal-only stress test (most important)
Collect:
- 20–50 nominal parts
- 3–5 capture conditions you expect on the line:
  - slightly brighter/dimmer
  - slightly different angle
  - warm lights vs cold lights
  - intentionally create glare if glossy

Protocol:
1) Train PatchCore on one condition.
2) Score nominal images from other conditions.
3) Plot score distributions per condition.
4) Set a threshold on the train condition (target FPR).
5) Measure FPR on the other conditions.

If FPR explodes under mild shift, you either:
- fix capture, or
- partition into modes (station/product) / routing, or
- increase nominal coverage.

### 4.2 Repeatability test
Capture the same part N times.
If scores vary widely, your capture is unstable.

### 4.3 “Nuisance map” test
Look at the top false positives and see if they cluster in:
- shiny edges
- screws/fixtures
- corners
- text/labels

If yes: mask those regions or fix lighting/polarization.

---

## 5) How to learn cameras/lighting without becoming a camera engineer

You don’t need a full optics background. You need a focused toolkit:

### 5.1 Learn the 10 knobs
- exposure (shutter)
- gain/ISO
- aperture
- white balance
- focus
- lighting geometry (diffuse vs directional)
- specular control (polarizers)
- repeatability/fixturing
- resolution and defect size in pixels
- motion blur

### 5.2 Work with manufacturing like a scientist
Treat capture as an experiment:
- change one knob at a time
- keep “golden parts” for repeatability checks
- log capture settings and date/time

### 5.3 Build a capture datasheet
For each station/product, record:
- camera model + lens
- fixed settings
- light type and position
- sample images and known nuisances

This will make your project feel rigorous.

---

## 6) Practical roadmap for your project

1) Stabilize capture where possible (disable auto-exposure/WB).
2) Collect nominal envelope data.
3) Run nominal stress test + repeatability test.
4) If multiple modes exist, adopt per-mode nominal banks or routing.
5) Only then optimize model details.

---

## Appendix: why PatchCore is sensitive

PatchCore uses mid-level pretrained embeddings. These embeddings are not invariant to illumination in the way you might wish.
Nearest-neighbor distances amplify distribution shifts.

So a stable input distribution is a first-class requirement, not an afterthought.
