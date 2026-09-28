# ai-prepress

A standalone tool for the print/retouching side of an AI image pipeline: color matching, masking, cropping, face retouching, dust removal, background replacement. Built after a production incident where files kept reaching retouching with inconsistent profiles and colors drifting warmer across repeated regenerations of the same image ("dérives chromatiques").

![Inspect view showing a loaded image next to its real dimensions, bit depth, and ICC profile fields](assets/screenshot.png)

Early stage. What's here right now: a shared image I/O layer, an acceptance-check utility, an image inspector (real dimensions, bit depth, ICC profile, EXIF, raw tags), Match Look, Face Regions + Retouch Faces (Dark Circles/Even Skin/Contouring, running locally via MediaPipe landmarks), a local FastAPI service, and a dark-themed HTML/JS front end. Masking, AI crop, Blemish removal, dust removal, background replacement and snap-to-eye aren't built yet.

## How it works

Two pieces everything else builds on:

- **I/O** (`ai_prepress/io.py`) loads and saves images without touching bit depth or dropping the ICC profile. Every feature routes through this instead of writing its own load/save path. It exists because a common failure mode - silently dropping the ICC profile and clamping to 8-bit right at the point pixel data crosses a plain numpy array (`Image.fromarray()` returns a fresh object with an empty `.info`) - is exactly the bug that caused the incident above. Checked directly, not assumed: round-tripping a 16-bit wide-gamut TIFF through Pillow's own array path collapses it; going through `tifffile` with the profile written as an explicit extratag does not. TIFF is the only format that keeps full bit depth here; PNG/JPEG/WebP are accepted too but are 8-bit only (WebP is written lossless - Pillow defaults it to lossy, and unlike JPEG it has a real lossless mode, so there's no reason to take the lossy path).
- **Acceptance checks** (`ai_prepress/checks.py`) run on a before/after pair and report Delta-E 2000 drift, a bit-depth sanity check (count of unique pixel values per channel - 256 or fewer on a file that's supposed to be 16-bit means something clamped it upstream), and whether the ICC profile survived. The Delta-E math goes through `colour-science`'s own RGB→XYZ→Lab conversion in full float rather than Pillow's ICC transform, specifically because Pillow can't hold 16-bit RGB - routing through it would quantize to 8-bit before measuring, making the check blind to exactly the sub-8-bit drift it exists to catch.

## Inspect

Drop an image in and get the real picture before running anything on it: dimensions, dtype, bit depth, file size, per-channel unique-value counts (and a flag if a file claiming to be 16-bit only actually has ≤256 distinct values per channel - a common sign something upstream already collapsed it to 8-bit), DPI, TIFF compression, and EXIF.

For color, two separate things are shown, not one:
- **The real embedded ICC profile**, read directly off the file when there is one - description, color space, connection space, device class, rendering intent, ICC version, copyright, creation date, white point temperature. All of this comes straight from the profile's own header, nothing inferred.
- **A colour-space guess** (`sRGB` or `Adobe RGB (1998)`), shown alongside it. This is a heuristic - a substring match on the profile's description, falling back to `sRGB` if there's no profile or nothing matches - and it's what Match Look's math actually buckets colors into internally (see the note below). It stays visible even when the real profile is known, since it isn't always the same thing and Match Look's own assumption is worth seeing plainly.

TIFF files also get a full raw tag dump (every tag on the page, the ICC profile's own bytes excluded since they're already shown properly above) for whenever the curated fields aren't enough.

## Match Look

First real feature, built first on purpose: no model, no GPU, plain numpy. Applies one reference image's color statistics to a target - MKL (matches full covariance) or Reinhard (matches per-channel mean/std in a log-LMS space). Reimplemented directly rather than depending on the `color-matcher` package: it's GPL-3.0, this repo is MIT, and its own example code clamps output to 8-bit before saving anyway, which is the thing this whole project exists to stop doing. The math itself (Reinhard et al. 2001, Monge-Kantorovich linearization) is published and unencumbered.

One function covers two workflows: point it at a flat, neutral reference before retouching to normalize a batch without baking in a look, or at a graded reference after retouching for delivery consistency.

> [!NOTE]
> The color-transfer math itself works in one of two named spaces (sRGB or Adobe RGB (1998)), decided by the same heuristic guess Inspect shows - a substring match on the profile description, not a real ICC parser. Covers the two spaces actually in use here, nothing more general yet.

## Training data (dust removal)

Dust removal needs a fine-tuned detector before anything else, and that needs a dataset before the fine-tuning does - so this came first, on its own, ahead of the feature itself.

`training/dust_removal/synthesize.py` takes a directory of clean (dust-free) images and adds synthetic sensor-dust spots, writing the augmented images plus a COCO-style `annotations.json` next to them:

```bash
uv run python -m training.dust_removal.synthesize path/to/clean_images/ path/to/output/
```

Dust is modeled as multiplicative attenuation in linear light (soft-edged, semi-transparent, roughly circular) rather than painted onto the gamma-encoded pixels directly - matches how DxO describes building their own detector's training data, and it's the only way the falloff shape comes out right. Goes through the same `cctf_decoding`/`cctf_encoding` pair Match Look uses, not a hardcoded sRGB gamma.

No RF-DETR fine-tuning yet, and no real "clean" source photos lined up to run this against at scale - both still open.

## Installation

```bash
uv sync
```

Needs Python 3.12 - pinned deliberately rather than using whatever's newest on the machine, since some of the model-facing dependencies landing once masking gets built are unlikely to have wheels for a Python release that's still this new.

## Usage

Run the API and UI - no environment variables needed, every feature currently in the UI runs locally:

```bash
uv run uvicorn ai_prepress.api.main:app --reload
```

Open `http://127.0.0.1:8000` - it opens straight into Inspect. Drag an image onto the drop zone (or click to browse; TIFF, PNG, JPEG, WebP) and it fills in place with a preview, a filename/size chip, and replace/remove controls - the same import control is reused wherever the app needs an image in, including both slots in Match Look. Previews shown in the browser are downsized (1024px on the long edge) and JPEG-encoded before being sent over - on a 48-megapixel 16-bit TIFF that cut preview generation from a few seconds and a 129MB PNG down to well under 50ms and a few hundred KB. The stored result itself, reachable from the download link, keeps full bit depth and resolution.

The dark theme (`tokens.css`, `components.css`) is dropped in as-is from an internal design handoff (POP R&D), per that handoff's own instructions for a CSS-based stack - not modified, just wired up.

Or call the core directly:

```python
from ai_prepress.io import load, save
from ai_prepress.features.match_look import match_look

target = load("target.tiff")
reference = load("reference.tiff")
result = match_look(target, reference, method="mkl")
save(result, "output.tiff")
```

Ran it on a synthetic 6000x4000 16-bit file to get a real number instead of guessing: about 6 seconds for MKL, CPU only, on an M-series Mac.

Or read a file's real metadata without touching pixels:

```python
from ai_prepress.metadata import describe

info = describe("scan.tiff")
info.icc_profile       # real fields read from the embedded profile, {} if there isn't one
info.colourspace_guess  # the sRGB / Adobe RGB (1998) bucket Match Look's math uses internally
```

## Face regions (local, MediaPipe landmarks)

Region detection for Retouch Faces (under-eye, cheek, forehead, skin) runs locally via `ai_prepress.face_landmarks` - no Modal deployment, no GPU, MediaPipe's own CPU path is fast enough on a single image.

This replaced an earlier version built on `jonathandinu/face-parsing` (SegFormer-B5 fine-tuned on CelebAMask-HQ), deployed to Modal and non-commercial-licensed. That approach got the plumbing proven but turned out to be the wrong foundation: CelebAMask-HQ's 19-class taxonomy has no under-eye, cheek, or forehead class, and Retouch Faces' actual sub-tools (Dark Circles, Contouring) need exactly those regions - no amount of model quality fixes a class that doesn't exist in the training data. MediaPipe's 478-point face mesh defines these regions as geometry instead, and is Apache 2.0.

```python
from ai_prepress.io import load
from ai_prepress.face_landmarks import detect_landmarks, under_eye_band, skin_region

image = load("portrait.tiff")
landmarks = detect_landmarks(image.array)  # None if no face found
right_under_eye = under_eye_band(landmarks, "right")  # polygon, (x, y) pixel coords
skin_oval, cutouts = skin_region(landmarks)  # face oval + eye/lip cutouts to subtract from it
```

Region index groups (`FACE_OVAL`, eye/lip loops, eyebrows) are walked from mediapipe's own `face_mesh_connections.py` edge lists, not guessed - see the module's docstring for sourcing and for which regions are geometrically precise (eyes, lips) versus approximate (cheek, forehead - no natural landmark boundary exists for those the way an eyelid margin does for the eye).

**Face Regions tab in the UI**: upload a portrait, hit Find regions. Coordinates come back already scaled to match the preview image, so the browser draws each region as a canvas path directly - no per-pixel compositing needed the way the old label-map viewer required. Checkbox legend toggles regions, hover pops one region and dims the rest.

## Retouch Faces

The masks above feeding into an actual edit: `ai_prepress.features.retouch_faces` does classical frequency separation - split the image into a low-frequency (LF, tone/shading) layer and a high-frequency (HF, texture/pores) layer, edit LF only inside a region mask, recombine with the HF layer. HF stays untouched for Dark Circles and Contouring (what makes "waxy skin" impossible by construction, not something hoped for from a generative model); Even Skin's Texture control is a deliberate exception - see below. Blemish removal isn't here - it needs actual inpainting, a different kind of model this project doesn't have wired up yet. Built against a research pass across Capture One's own documentation, Retouch4me, Lightroom Classic's masking architecture, and the manual Photoshop frequency-separation workflow professional retouchers already use - see the "face-parsing-vs-landmarks-pivot" project note for the full findings.

```python
from ai_prepress.io import load
from ai_prepress.features.retouch_faces import retouch_faces, RetouchStrengths

image = load("portrait.tiff")
result = retouch_faces(image, RetouchStrengths(dark_circles=0.5, even_skin=0.4, contouring=0.3))
# None if no face detected. A plain RetouchStrengths applies to every detected face; pass
# {0: RetouchStrengths(...), 1: RetouchStrengths(...)} for independent per-face control.
```

**Controls**, confirmed against Capture One's own Skin Tools panel rather than guessed:
- **Dark circles / Even skin / Contouring** - 0-1 strength, as before.
- **Even skin - Texture** (`even_skin_texture`, -1..1) - Capture One's Even Skin panel has both Amount and a separate signed Texture slider; negative pushes extra flattening beyond Amount alone (their own example, "80 Amount / -70 Texture", is described as an editorial-polish look), positive mildly restores detail. This scales the HF layer within the even_skin mask specifically - the one place HF is ever touched.
- **Mask feather / edge** (`feather_amount`, `edge_amount`) - Lightroom Classic's own mask "Reshape" step (Feather + Edge) is the cheap fix for "the AI-derived region is a bit off" before a full paint brush, which this project doesn't have yet. `edge_amount` erodes (negative) or dilates (positive) a region's boundary via an exact separable min/max filter, capped at half an eye-width so an extreme value shrinks a region without a one-click "mask disappeared" trap.
- **Multi-face** - every competitor researched treats this as core. `detect_landmarks(rgb, max_faces=N)` returns every detected face, sorted left-to-right for stable indexing.

No scipy, and PIL's `GaussianBlur` flatly refuses float-mode images (confirmed by hand, not assumed) - routing 16-bit data through it would mean quantizing to 8-bit first, so the LF/HF split runs on a from-scratch separable box-blur approximation operating directly on the same float64 arrays the rest of the pipeline uses. Mask edges are feathered (a hard polygon edge on a brightened region looks like a sticker) using the same blur; erode/dilate uses a separate exact min/max filter (a rectangle is the Minkowski sum of a horizontal and vertical segment, so one pass per axis is exact, not an approximation).

**Retouch Faces tab in the UI**: upload a portrait - every detected face is listed (with a cropped thumbnail) alongside an "All faces" row. Select a row to edit that face's (or everyone's) sliders independently; once any individual face is customized, "All faces" greys out rather than silently dropping whatever was set there, since the API sends either one broadcast strengths object or an explicit per-face map, never both at once. Shows original next to retouched side by side (same before/after pattern as Match Look), plus the same acceptance-check stats (Delta-E, bit depth, ICC profile) Match Look surfaces, and a full-precision TIFF download.

## Roadmap

- [x] Shared I/O + acceptance checks
- [x] Image inspection (dimensions, bit depth, real ICC profile + guess, EXIF, raw tags)
- [x] Match Look
- [ ] pywebview desktop shell around the current FastAPI + HTML UI
- [ ] Shared segmentation backend (SAM3 + BiRefNet), feeds masking, AI crop, and background cutout
- [ ] AI Crop
- [x] Face regions (`ai_prepress.face_landmarks`, local MediaPipe) - supersedes the earlier Modal-deployed semantic parser, see above. `deploy/face_parsing.py` and `ai_prepress.face_parsing` are still in the repo (real, tested, still deployed) but no longer wired into the UI.
- [x] Retouch Faces - Dark Circles/Even Skin/Contouring via LF/HF split on the face-region masks, see above. Blemish removal still needs Inpaint-Anything, not built yet.
- [x] Dust removal training-data synthesizer (`training/dust_removal/`)
- [ ] Dust Removal (RF-DETR fine-tune on the synthetic data, then the fill step)
- [ ] Background Replacement
- [ ] Snap to Eye - blocked on a scope call, Capture One's actual feature is a coarse focus-check aid, not a precision alignment tool, and it's not clear yet which one is wanted here

Standalone desktop tool, not a Photoshop plugin, but the core is a plain package behind a local HTTP API specifically so a plugin (or a different deployment shape, later) can become just another client instead of a rewrite. The original plan routed every model-tier step through RunPod/Modal for v1 simplicity - in practice, MediaPipe's CPU path turned out fast enough that Face Regions and Retouch Faces both run entirely locally instead, no deployment needed. `deploy/face_parsing.py` (the retired Modal-hosted semantic parser, see above) is the one remaining example of the originally-planned remote-model pattern, kept as reference.
