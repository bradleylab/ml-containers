# n2n4m

[rob-platt/N2N4M](https://github.com/rob-platt/n2n4m) — Noise2Noise
denoising for CRISM Mars hyperspectral SWIR data.

CRISM's L (SWIR) detector has degraded steadily since 2006, and a large
share of the later archive is treated as unusable: spike and stripe noise
swamps the absorption features that mineralogy is read from. N2N4M is a 1D
convolutional U-Net (1,092,945 parameters) trained Noise2Noise-style —
noisy target, noisy input, no clean reference — that denoises CRISM
spectra pixel by pixel. It operates on the 350 channels of the
`PLEBANI_WAVELENGTHS` subset and passes the remaining 88 of CRISM's 438
L-sensor bands through untouched.

The same package ships three things that come with it in practice: CoTCAT
(Bultel et al. 2015) as the benchmark denoiser to compare against, image
ratioing against the Plebani et al. (2022) bland-pixel model, and the
standard CRISM summary parameters.

**Relation to `momo` in this repo.** MOMO is a foundation model over Mars
orbital *imagery* — HiRISE / CTX / THEMIS tiles, panchromatic-scale
geomorphology. n2n4m works on CRISM *hyperspectral* cubes and answers a
different question: it is the step that makes the degraded part of the
CRISM archive analysable at all, rather than a model you fine-tune onto a
downstream task.

## Image tag

`ghcr.io/bradleylab/n2n4m:v1` (also `:latest`, `:torch2.9-cu128`)

## Stack

- Base: `python:3.11-slim`
- Python 3.11 (matches upstream's `environment.yml`; `setup.py` allows
  `>=3.10,<3.14`)
- PyTorch 2.9.1, cu128 wheels
- `n2n4m` at pinned commit `7e5c58a` (2025-09-29)
- `crism_ml` ([Banus/crism_ml](https://github.com/Banus/crism_ml), the
  Plebani toolkit) at pinned commit `028025e` (2025-11-11)
- numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.7.2,
  spectral 0.25, mat73 0.65, Bottleneck 1.6.0, joblib 1.5.3,
  matplotlib 3.10.7, pyarrow 22.0.0, ipywidgets 8.1.9, ipykernel 7.3.0,
  pytest 8.4.2

`MPLBACKEND=Agg` is set so `n2n4m.plot` imports in a headless container.

### Why this base and not a CUDA or NGC image

The model is ~1.1 M parameters and runs perfectly well on CPU; upstream
picks the device opportunistically (`check_available_device()` returns
`cuda` only if torch sees one). A `nvidia/cuda:*-runtime` base would add
CUDA runtime libraries that the PyPI cu128 torch wheel already bundles as
its own `nvidia-*` dependencies, and an NGC PyTorch base would add its
whole Megatron / apex / transformer-engine surface for a package that
imports none of it.

So: slim Debian base, CUDA capability carried entirely by the torch wheel.
One image runs unchanged on Compute2's `general-cpu` and on `general-gpu`.

CPU-only torch wheels were the obvious alternative and would have made
the image substantially smaller — the bundled CUDA libraries are most of
its bulk. Rejected because that makes the GPU path *impossible* rather
than optional, which is not what "GPU is optional" should mean, and the
alternative (two image variants) is worse for a 4 MB model.

## Weights — baked, deliberately

**Decision: the committed weights count as baked, and this image is
documented as a baked-weights image.**

Upstream commits both artifacts into its source tree and declares them as
package data (`package_data={"n2n4m": ["data/*"]}`), so a plain
`pip install` places them in site-packages:

| Artifact | Path in image | Size |
|---|---|---|
| Trained N2N4M weights | `.../site-packages/n2n4m/data/trained_model_weights.pt` | 4.45 MB |
| Fitted feature scaler (sklearn `RobustScaler`, 350 features) | `.../site-packages/n2n4m/data/n2n4m_feature_scaler.joblib` | 13.7 KB |

Resolve them with `importlib.resources`, or just let
`instantiate_default_model()` and `load_scaler()` find them — those are
the upstream defaults.

Why call this baked rather than "arrives with the source checkout":

- Every consequence the label exists to communicate holds. No network at
  inference time, no HF token, no cache bind-mount, deterministic content
  per image digest.
- It is a *stronger* pin than the fetch-at-runtime images in this repo.
  For `momo`, the tag `:v1` tells you nothing about which
  `Mirali33/MOMO` checkpoint a given job actually loaded. Here one commit
  SHA pins code and weights together, and the image digest pins that.
- The usual reason to avoid baking — multi-GB layers that bloat every
  pull — does not apply at 4.45 MB.
- Licensing does not fork. MIT covers the whole upstream repo including
  `data/`, so there is no code-MIT / weights-CC-BY split to track the way
  `momo` has.

The dependency this creates is on upstream continuing to ship the files as
package data. The build-time smoke test asserts both are present *and*
loads the checkpoint and runs a forward pass, so if that ever changes the
build fails rather than publishing a weightless image.

## External data — Plebani bland-pixel dataset (NOT baked)

Denoising needs nothing external. **Ratioing does.** `ratio_image()` /
`ratio_denoised_image()` fit the Plebani GMM bland-pixel model, which
needs `CRISM_bland_unratioed.mat` from Zenodo record
[10.5281/zenodo.13338091](https://zenodo.org/records/13338091) ("CRISM ML
dataset", CC-BY-4.0):

| File | Size | Needed for |
|---|---|---|
| `CRISM_bland_unratioed.mat` | 0.47 GB | image ratioing |
| `CRISM_labeled_pixels_ratioed.mat` | 1.52 GB | retraining only |

Not baked: half a gigabyte of CC-BY data with a different license and its
own citation, used by one code path. Fetch once to a persistent host
directory and bind-mount it:

```bash
mkdir -p crism-ml-data
curl -L -o crism-ml-data/CRISM_bland_unratioed.mat \
  https://zenodo.org/records/13338091/files/CRISM_bland_unratioed.mat
```

The directory you pass as `train_data_dir` must be the one *containing*
the file, and the filename must be exactly `CRISM_bland_unratioed.mat` —
`crism_ml` looks it up by name.

## Run

```bash
docker run --rm -it \
  -v "$PWD/crism-ml-data:/data/CRISM_ML" \
  -v "$PWD/scenes:/data/scenes" \
  ghcr.io/bradleylab/n2n4m:v1
```

Add `--gpus all` if a GPU is present. Nothing changes in the code — torch
finds it or it doesn't.

## Minimal denoising example

Whole-image denoising, no external data needed:

```python
from n2n4m.n2n4m_denoise import denoise_image

# CRISM L-sensor TRR3 .img (the matching .hdr or .lbl must sit beside it)
denoised = denoise_image("/data/scenes/3561F/ATU0003561F_01_IF168L_TRR3.img")
print(denoised.shape)  # (rows, cols, 438) — 350 bands denoised, 88 passed through
```

The object-oriented path, which is what the tutorial notebooks use and
what you want if you are also ratioing or computing summary parameters:

```python
from n2n4m.crism_image import CRISMImageN2N4M

img = CRISMImageN2N4M("/data/scenes/3561F/ATU0003561F_01_IF168L_TRR3.img")
img.load_n2n4m_scaler()          # the baked RobustScaler
img.load_n2n4m_model()           # the baked weights
img.n2n4m_denoise(batch_size=1000)

# Ratioing — this is the step that needs the Zenodo file
img.ratio_denoised_image(train_data_dir="/data/CRISM_ML")
img.calculate_summary_parameter("OLINDEX3")
```

Write the result back out as ENVI so it can be map-projected in CAT:

```python
from n2n4m.io import write_image
write_image("/data/scenes/3561F/denoised.hdr", img.denoised_image, img.SPy)
```

The upstream tutorial notebooks (`notebooks/tutorials/`, notably
`CRISMImageN2N4M_nb.ipynb`) are not copied into the image — clone the repo
at the pinned SHA if you want them.

## Inputs

- CRISM **L sensor**, **TRR3**-processed `.img` cubes, BIL interleave,
  438 bands. `n2n4m.io.load_image` rejects anything that is not a BIL
  file. If only a `.lbl` is present, `crism_ml` generates the `.hdr`.
- If you are using MarsSI `_CAT_corr.img` products, rename them to the
  original `.img` filenames — upstream's convention, and the ENVI header
  lookup depends on it.
- Bad values (65535) are imputed before denoising; upstream's threshold
  parameter controls this and is reported, not silent.

## Resources

Honest version: **no GPU required, and no H100 required.** This is a
1.09 M-parameter model. The GPU path is a throughput convenience for
processing many scenes, not a requirement.

| | |
|---|---|
| GPU | Optional. Any CUDA device torch can see. sm_90 (H100) is supported but is not the design point. |
| CPU | Fine for single scenes. The ratioing step (Plebani GMM fit) is CPU/numpy-only and is **not** GPU-accelerated at all, so a GPU does nothing for it. |
| RAM | Driven by cube size, not by the model. The pipeline holds several `n_pixels × 438` float64 copies simultaneously (imputed, scaled, recombined) — budget ~3.5 KB per pixel per live copy and assume ~4 are live. Ratioing additionally holds the 0.47 GB `.mat` plus the fitted GMM. |
| Batch size | `batch_size=1000` spectra per forward pass is the upstream default; lower it if a GPU runs out of memory. |
| Runtime | Not benchmarked in-house. Measure one scene before sizing a batch job — do not extrapolate from this table. |

## Run on Compute2

Denoising only, no GPU, `general-cpu`:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=8 \
       --mem=32G \
       --time=04:00:00 \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+n2n4m+v1.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley/crism-ml-data:/data/CRISM_ML,/scratch2/fs1/alexander.s.bradley/crism-scenes:/data/scenes \
         bash -lc "export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/n2n4m_denoise_scene.py"'
```

Many scenes, GPU worth having, `general-gpu`:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-gpu \
       --gpus=1 \
       --cpus-per-task=8 \
       --mem=64G \
       --time=08:00:00 \
       --array=0-49 \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+n2n4m+v1.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley/crism-ml-data:/data/CRISM_ML,/scratch2/fs1/alexander.s.bradley/crism-scenes:/data/scenes \
         bash -lc "export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/n2n4m_denoise_scene.py"'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
into the container, so any stray `pip install --user` on the login node
would otherwise shadow the container's site-packages. See
`~/.claude/rules/research-infrastructure.md`.

Import the `.sqsh` cache with:

```bash
enroot import -o /storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+n2n4m+v1.sqsh \
  'docker://ghcr.io#bradleylab/n2n4m:v1'
```

See `n2n4m/SMOKE.md` for the smallest Compute2 job that proves the image
works.

## Licensing and citation

- **Code and weights: MIT** (single license, whole upstream repo including
  `n2n4m/data/`). Confirmed against the GitHub API, 2026-08-18.
- **`crism_ml`**: separate upstream project (Banus/crism_ml), pulled in as
  a dependency for ratioing.
- **Plebani bland-pixel dataset**: CC-BY-4.0, Zenodo
  10.5281/zenodo.13338091. Cite it if you publish ratioed products.

Cite the Plebani toolkit paper for the ratioing and blandness model —
Plebani E, Ehlmann BL, Leask EK, Fox VK, Dundar MM. *A machine learning
toolkit for CRISM image analysis.* Icarus. 2022 Apr;376:114849 — and
CoTCAT if you use it as a benchmark — Bultel B, Quantin C, Lozac'h L.
*Description of CoTCAT (Complement to CRISM Analysis Toolkit).* IEEE
JSTARS. 2015 Jun;8(6):3039–49.

For N2N4M itself: Platt RW, Arcucci R, John CM. *Noise2Noise Denoising of
CRISM Hyperspectral Data.* arXiv:2403.17757, 26 Mar 2024.
[doi:10.48550/arXiv.2403.17757](https://doi.org/10.48550/arXiv.2403.17757).
Verified via OpenAlex 2026-08-18; indexed there as a preprint only, so
check for a journal version before citing in a manuscript.

## Limitations and caveats

- **The scaler was pickled under scikit-learn 1.2.2.** Loading it under
  the pinned 1.7.2 emits an `InconsistentVersionWarning`. Verified that it
  reads correctly: `RobustScaler`'s fitted state is `center_` / `scale_`
  and `transform` is `(X - center_) / scale_`, arithmetically identical
  across these versions. The warning is noise, not a defect — but do not
  suppress it globally without knowing that.
- **torch 2.6+ `weights_only` — checked, not affected.** torch 2.6 flipped
  `torch.load`'s `weights_only` default to `True`, which breaks
  checkpoints that pickle arbitrary objects, and upstream calls
  `torch.load` with no `weights_only` argument. Inspecting the checkpoint
  shows it is a plain `OrderedDict` of tensors referencing only
  `torch._utils._rebuild_tensor_v2` and `torch.{Float,Long}Storage`, all
  on the `weights_only` allowlist. Verified loading under torch 2.9.1.
  The build-time smoke test asserts this so a future torch bump cannot
  break it silently.
- **`crism_ml` was pinned by us, not by upstream.** n2n4m's `setup.py`
  declares it as `git+...@master`, so upstream's own install instructions
  are not reproducible — a build today and a build next month can get
  different Plebani code. `CRISM_ML_REF` in the Dockerfile is our pin.
- **Scope: L sensor, TRR3, only.** Not the S (VNIR) detector, not TRR2,
  not map-projected MTRDR products.
- **`n2n4m` exposes no `__version__`** and its `setup.py` version has sat
  at `0.0.2`. The commit SHA is the only meaningful version; that is why
  the smoke test asserts it from pip's `direct_url.json` rather than
  trusting the version string.
- **Stale `requires.txt` in the repo.** The committed `n2n4m.egg-info/`
  still lists `ray>=2`; the pinned commit removed that requirement. Read
  `setup.py`, not the egg-info. Ray is needed only for hyperparameter
  tuning and is not installed here.
- **Retraining is out of scope for this image.** It needs both Zenodo
  `.mat` files plus the raw MarsSI imagery, and `ray` for hyperparameter
  tuning. The image ships the inference path.
- **Small upstream project.** 11 stars, last pushed 2025-09-29, a single
  PhD author. It is well-structured and MIT-licensed, but expect to be
  the one who finds the sharp edges.
