# adaf

Automatic Detection of Archaeological Features in LiDAR terrain models. ZRC SAZU
+ Bias Variance Labs + The Discovery Programme. **Apache-2.0 code,
CC-BY-SA-4.0 weights** — the split matters, see below.

HRNet semantic segmentation and Faster R-CNN object detection over relief
visualizations of a bare-earth DTM. Four classes, each with its own pair of
models: `barrow`, `ringfort`, `enclosure`, and `AO` (all archaeology).

Field-tested on a 197 km² Irish infrastructure corridor: 84% recall against
known sites, plus 116 candidates that manual inspection had missed.

## Share-alike attaches to anything you fine-tune

The code is Apache-2.0. The **weights are CC-BY-SA-4.0**
([Zenodo](https://doi.org/10.5281/zenodo.15848663)), and that is a share-alike
licence.

| What you do | Obligation |
|---|---|
| Run inference, publish detections | None. Results are not a derivative |
| Fine-tune, keep in-house | None |
| Fine-tune, **distribute the weights** | Must release under CC-BY-SA-4.0 |
| Redistribute weights unmodified | Attribution + notice. This image does both |

If a downstream ever needs permissively-licensed weights, do not start from
these — train from ImageNet initialisation inside the same Apache-2.0 ADAF and
AiTLAS code. That forfeits the archaeological pretraining and buys licence
freedom. It is a real trade.

Notice ships at `/opt/licenses/LICENSE.weights.md` inside the image.

## Feed it SLRM at 512 px, or the scores mean nothing

The weight filenames state the training regime outright:
`barrow_HRNet_SLRM_512px_pretrained_...`. **SLRM** is the Simple Local Relief
Model, produced by `rvt-py` — already a dependency, via AiTLAS.

Give these models a hillshade, a raw DTM, or a different tile size and they will
still return detections. Just worse ones, for reasons that have nothing to do
with your terrain. This is the most likely way to get a misleadingly poor result
from this image.

### Resolution is an open question, not a solved one

The models were trained on national ALS at roughly **0.5 m**. Drone LiDAR at
50–80 pts/m² supports a DTM around **0.25 m**, which puts 4× more pixels on the
same earthwork — a barrow that filled a 512 px tile at 0.5 m fills a sixteenth
of one at 0.25 m.

Whether to resample down to match the receptive field or run native is
**unanswered for any new survey**. Run both and compare before trusting either.
Resampling to 0.5 m throws away resolution you paid for; running native may put
features at a scale the network never saw.

## Calling it on Linux

Upstream hardcodes its model paths as Windows raw strings:

```python
"barrow": r".\ml_models\OD_barrow.tar"
```

On Linux a backslash is an ordinary filename character, so that resolves to a
single nonexistent file and every stock code path fails. **This is in the
source, not just the install docs.** This image supports two ways around it,
neither of which patches upstream.

### The clean route — `custom_model`, a documented parameter

Both entry points take `custom_model`, and `models["custom"]` is that value.
Because `Path(a) / Path(b)` returns `b` when `b` is absolute, an absolute POSIX
path passes through exactly as intended:

```python
from adaf.adaf_inference import run_aitlas_segmentation, run_aitlas_object_detection

W = "/opt/adaf-weights"
seg = run_aitlas_segmentation(
    ["custom"], tiles_dir,
    custom_model=f"{W}/barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
)
det = run_aitlas_object_detection(
    ["custom"], tiles_dir, custom_model=f"{W}/OD_barrow.tar",
)
```

`custom` is a single slot, so loop one class at a time and keep the outputs in
separate directories — every run labels its output `custom`.

Use this route. It is a supported parameter, so it survives upstream updates.

### The compatibility route — for the stock notebooks

`main_routine()` and the shipped notebooks index the model dict by label and
never touch `custom_model`. So the build also creates symlinks whose filenames
literally contain backslashes, letting that code find the weights unmodified.
Both spellings are wired, and the build asserts all eight resolve either way.

When upstream fixes the separators, the backslash links become inert rather than
breaking, and the POSIX-named copies in `adaf/ml_models/` are what it will find.

## Python 3.9 is not an oversight

AiTLAS pins `h5py<3.2.1`, `imagecodecs==2023.3.16` and `ipykernel==6.15.0`.
h5py 3.2.0 predates Python 3.10 and ships no cp310+ wheel, so a newer
interpreter turns that pin into a source build that fails. Upstream's "newer
Python should also work" refers to GDAL, not to these.

GDAL, rasterio and fiona come from conda-forge, resolved as a set rather than
pinned to exact versions. Exact pins are what broke the first build: conda found
a combination satisfying all three against a `libsqlite` too old for the
`libgdal` it chose, and the image failed with `undefined symbol:
sqlite3_total_changes64`. What matters is that they resolve consistently, so the
solver gets the constraint that actually matters (`libsqlite>=3.46`) and is left
free on the rest.

Upstream installs these from cgohlke's `win_amd64` wheels, which have no Linux
equivalent; that, not anything architectural, is why the published instructions
are Windows-only.

AiTLAS itself installs from a wheel **vendored inside the ADAF repo**, pinned at
0.0.1, not from PyPI. Upstream of it is
[biasvariancelabs/aitlas](https://github.com/biasvariancelabs/aitlas)
(Apache-2.0). Installing the vendored copy is what upstream instructs;
substituting the public package would be a change of dependency, not a
packaging detail.

## GPU

Not a CUDA base image, so it sets `NVIDIA_VISIBLE_DEVICES` explicitly — without
that, enroot injects no driver and torch reports no GPU inside a job that was
allocated one. The cost is that **CPU-partition runs need an override**:

```bash
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
```

Inference is small by 2026 standards; an H100 is oversized even for fine-tuning
at this model scale. CPU works, slowly.

## What is baked

- ADAF at commit `a44acdd678773666458ea50f544b5d2292bcef51` → `/opt/adaf`
- Eight weight TARs, 5.52 GB → `/opt/adaf-weights` (**not** extracted; AiTLAS
  loads the `.tar` directly and upstream is explicit that extracting breaks it)
- CC-BY-SA notice → `/opt/licenses/LICENSE.weights.md`

## Status

**Experimental. Plumbing verified on real data; detection quality unmeasured.**

Run on Compute2 2026-09-01, job 2957140, c2-gpu-016 (H100 80GB), against the
Tisch Park drone-LiDAR DTM (0.25 m, 1017 x 686 m, EPSG:32615). Whole chain in
36 seconds:

| Stage | Result |
|---|---|
| torch / GPU | 2.5.1+cu121, H100 visible |
| SLRM visualisations | 12 tiles, 0.9 s |
| Segmentation via `custom_model` | 11.6 s |
| Vectorisation (library defaults) | GPKG written, 1 polygon |
| Stock `labels=["barrow"]` route | works — the backslash symlinks resolve |

Both calling routes work, so the Linux path shim does what it claims.

**This says nothing about detection quality.** Tisch Park is an urban campus
park with no archaeology in it, so the single polygon (159 m2, roundness 0.961)
is a false positive by construction — almost certainly a landscaping mound or
tree ring. One detection over 0.7 km2 is at least not a noise storm, but one
number on a site with no ground truth is not a specificity measurement.

Measuring recall needs a site with a published earthwork inventory.

## Provenance

- Upstream: <https://github.com/EarthObservation/adaf>
- Weights: <https://doi.org/10.5281/zenodo.15848663>
- Paper: Čož, Corns, Curran, Kocev & Kokalj (2026), *Journal of Archaeological
  Science: Reports* 71:105733, <https://doi.org/10.1016/j.jasrep.2026.105733>
