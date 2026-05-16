# Image capture guide (industrial QA)

This is a practical guide for collecting nominal images that a PatchCore-style system can learn from.

## 1) Define the “unit of normal”
Before shooting anything, decide what “normal” means:
- acceptable cosmetic variation
- acceptable alignment/fit
- acceptable lighting artifacts (glare, specular highlights)

If humans disagree on whether something is acceptable, the model will be unstable.

## 2) Control the acquisition geometry
PatchCore is sensitive to nuisance variation. Start by locking down:

- **Camera**: same sensor/lens if possible
- **Distance**: fixed standoff; use a jig
- **Pose**: fixed orientation; use a fixture that constrains rotation
- **FOV**: ensure the part occupies a consistent portion of the frame
- **Background**: matte, non-textured, stable color (gray/black)

If you can’t control pose, treat each pose as a separate model or add pose normalization.

## 3) Lighting (most important)
Aim for stable, repeatable illumination:

- **Diffuse lighting** (light tent / diffusion panel) to reduce hard specular glare
- Avoid direct point lights; they create highlights that look like anomalies
- Fix color temperature (avoid auto-white-balance drift if possible)

If the part is shiny, consider:
- cross-polarization (polarizer on lights + lens) to suppress reflections
- multiple light angles (but then lock them down)

## 4) Camera settings
Prefer manual settings:
- fixed exposure
- fixed ISO
- fixed white balance
- fixed focus

Auto exposure can cause nominal images to shift brightness and create false positives.

## 5) Resolution and cropping
- Shoot high-res if possible; downsample later.
- For PatchCore, what matters is consistent framing and enough resolution to see the defect scale.

Recommendation:
- store originals
- create a standardized preprocessed version (e.g., 256×256 or 512×512)

## 6) How many images and what diversity?
Collect **diversity across nuisance factors**:
- multiple days/shifts
- different operators if that changes placement
- natural manufacturing tolerances
- common benign artifacts (dust, minor smudges)

Split into:
- `nominal_train` (memory bank)
- `nominal_calib` (threshold calibration)
- `nominal_monitor` (drift checks)

## 7) Metadata to record (cheap, high value)
Store alongside each image:
- timestamp
- camera id
- line/station id
- product id / variant
- operator id (if relevant)
- exposure/lighting configuration version

This helps debug drift and false positives.

## 8) Quick sanity checks
After collecting the first 200–500 nominal images:
- fit PatchCore
- score a held-out nominal set
- inspect top 20 highest-scoring “nominal” images

If they’re all benign artifacts, add those artifacts to the nominal set or adjust capture.
