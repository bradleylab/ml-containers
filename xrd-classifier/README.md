# xrd-classifier

Automated phase identification from powder X-ray diffraction patterns
via autoXRD (Szymanski et al. 2021, *Chem. Mater.*). 1D CNN trained on
simulated patterns from a user-supplied CIF library, with
physics-informed augmentation (preferred orientation,
crystallite-size broadening, strain). Multi-phase by design (default
cap of 3 phases per pattern).

CPU-only by design. autoXRD inference is ~10 s/pattern on CPU; the
Compute2 use case (catalog-scale phase ID across many measurements)
parallelises across CPU job arrays better than it scales up a single
GPU. A CUDA variant could be added later if a training-side workload
warrants it.

## Image tag

`ghcr.io/bradleylab/xrd-classifier:v2` (also `:latest`,
`:autoxrd-tf2.16-cpu`). `:v1` remains pullable for rollback.

**AMD64-only.** The upstream's prediction pipeline calls into BGMN
(Rietveld refinement engine) at every spectrum, and BGMN ships only
as a Linux x86_64 binary. There is no arm64 path. Apple-Silicon Mac
users must run with `docker pull --platform linux/amd64` and qemu
emulation, which works (~2-3× slower than native x86_64).

## Changes since v1

- **Single source of truth for `autoXRD`.** v1 had two installs (pip
  0.0.7 + cloned-repo 0.0.6), and which one Python found depended on
  `cwd`, with the cloned 0.0.6 matching the bundled `Model.h5` weights
  and the pip 0.0.7 not. v2 installs editable from the cloned repo
  only.
- **BGMN pre-baked.** v1's first `run_CNN.py` invocation crashed
  because upstream's `BGMNWorker` does a lazy network download of the
  Linux x86_64 BGMN binary, races under multiprocessing, and leaves
  no usable zip behind. v2 calls `download_bgmn()` once at build time
  so the binary is already in the image and first inference works
  offline.

## Stack

- Base: `python:3.11-slim`
- TensorFlow >= 2.16 (CPU)
- `autoXRD >= 0.0.6` (PyPI)
- pymatgen, pyxtal, scipy, scikit-image (transitive)

## Bundled demo model

The container bakes in the upstream Li-Mn-Ti-O-F demo:

- **Path:** `/opt/xrd-autoanalyzer/Example/Model.h5` (~73 MB)
- **Chemistry:** Li, Mn, Ti, O, F (Szymanski's lithium battery paper system)
- **Reference CIFs:** `/opt/xrd-autoanalyzer/Example/References/`
- **Sample spectra:** `/opt/xrd-autoanalyzer/Example/Spectra/`

Run the demo end-to-end:

```bash
docker run --rm -it ghcr.io/bradleylab/xrd-classifier:v1
# Inside the container:
cd Example
python run_CNN.py
```

This reads `Spectra/*.xy` (two-column 2θ + intensity) and prints
phase predictions with probabilities.

## Inference on your own data

Drop two-column ASCII patterns (`*.xy`, `*.txt`, or csv) into a
`Spectra/` directory and bind-mount it:

```bash
docker run --rm -it \
  -v "$PWD/my-spectra:/opt/xrd-autoanalyzer/Example/Spectra" \
  -v "$PWD/my-output:/opt/xrd-autoanalyzer/Example/Output" \
  ghcr.io/bradleylab/xrd-classifier:v1 \
  bash -c "cd Example && python run_CNN.py"
```

The Li-Mn-Ti-O-F demo model is **only valid for that chemical
system**. For arbitrary minerals or other chemistries, retrain
(see below).

## Retraining for a new chemical system

The upstream `Novel-Space/` directory is the template. From inside
the container:

```bash
cd /opt/xrd-autoanalyzer/Novel-Space
# 1. Drop your CIFs into All_CIFs/  (e.g. quartz, calcite, dolomite,
#    K-feldspar, plagioclase for sed-pet teaching).
cp /your/host/cifs/*.cif All_CIFs/
# 2. Generate simulated reference + augmented training set.
python generate_References.py
python generate_XRD.py
# 3. Train the CNN.
python train_CNN.py
# 4. The trained Model.h5 lands in Novel-Space/. Copy or bind-mount
#    it into Example/ and run run_CNN.py against your spectra.
```

Training is CPU-tractable for small (<50 phase) systems but a CPU job
on Compute2 `general-cpu` is the cleanest path for larger ones. See
the upstream
[README](https://github.com/njszym/XRD-AutoAnalyzer) for full
documentation of the training pipeline and hyperparameters.

## Inputs

- Two-column ASCII pattern: 2θ (degrees) + intensity, no header. The
  upstream pipeline expects the standard 5–90° range; trim or
  resample if your instrument geometry differs.
- For retraining: a directory of CIFs covering every phase in the
  chemical space of interest. The
  [Crystallography Open Database](https://www.crystallography.net/cod/)
  is the canonical open source.

## Run on Compute2

Catalog-scale phase ID on `general-cpu`:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=4 \
       --mem=8G \
       --time=04:00:00 \
       --array=0-99 \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+xrd-classifier+v1.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley/spectra:/opt/xrd-autoanalyzer/Example/Spectra,/scratch2/fs1/alexander.s.bradley/xrd-output:/opt/xrd-autoanalyzer/Example/Output \
         --container-workdir=/opt/xrd-autoanalyzer/Example \
         bash -lc "export PYTHONNOUSERSITE=1; python run_CNN.py"'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — see
`~/.claude/rules/research-infrastructure.md`.

## Limitations

- **The demo model is chemistry-specific.** Li-Mn-Ti-O-F is a battery
  cathode system, not a general mineralogy classifier. Predictions
  on minerals outside that elemental set are meaningless. Plan to
  retrain.
- **Multi-phase cap is 3 by default.** Configurable in `run_CNN.py`,
  but increasing it costs accuracy because the search space scales
  combinatorially in phase count.
- **Preferred orientation augmentation assumes powder geometry.**
  Single-crystal or strongly textured samples are out of distribution.
- **No HF Hub presence for the upstream tool.** Weights distribution
  is via in-tree `Example/Model.h5`. If/when the Bradley Lab
  contributes a mineralogy-trained checkpoint, push it to HF Hub
  under MIT/Apache to fix the field-level gap.
- **Adjacent tool — Dara** (CederGroupHub/dara, *automated Rietveld*)
  fits the same workflow slot but is a *classical* tree-search +
  refinement pipeline, not an ML model. Not in this image; ship
  separately if needed (fails the `ml-containers` boundary test).
