# corrdiff — CorrDiff COSMO/ERA5 km-scale generative downscaling

Takes coarse global ERA5 reanalysis (0.25°) and produces kilometre-scale
regional fields over Europe, reproducing the COSMO-REA reanalyses that would
otherwise require a limited-area numerical weather prediction run. One package
holds two target resolutions, each with a deterministic and a generative mode.

Despite the CorrDiff name, this is a diffusion **transformer** (DiT) with axial
2D rotary position embeddings and localized neighborhood attention, not the
U-Net of the original CorrDiff, and its generative model predicts the target
field directly rather than a residual on top of a regression prediction.

- Framework: https://github.com/NVIDIA/earth2studio (Apache-2.0)
- Networks: https://github.com/NVIDIA/physicsnemo (Apache-2.0)
- Weights: https://huggingface.co/nvidia/corrdiff-cosmo-era5 —
  **OpenMDW-1.1**, ungated (see Licensing; this is not Apache-2.0)
- Papers: https://arxiv.org/abs/2309.15214 (CorrDiff),
  https://arxiv.org/abs/2206.00364 (EDM preconditioning)

> **Status: experimental.** The domain is Europe, which is not where the lab
> works; the value here is the downscaling method, not the region. No runtime
> or peak-memory figure below has been measured on Compute2 — `SMOKE.md` is the
> job that would produce the first ones.

## Image tag

`ghcr.io/bradleylab/corrdiff:latest` (also `:v1`, `:e2s0.17-cu128`)

## Contents

- `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`, Python 3.12, plus
  torch 2.11.0 + torchvision 0.26.0 from the PyPI cu128 wheels.
- `earth2studio==0.17.0`.
- `nvidia-physicsnemo` built from git at the revision earth2studio pins — **not**
  a PyPI release, for a specific reason spelled out under Caveats.
- `natten==0.21.7` compiled from source for sm_90, with the Hopper
  fused-neighborhood-attention path enabled.
- No wrapper CLI. Earth-2 Studio is a Python API; run inference from a script.

## Checkpoints and GPU requirements

Nothing is baked into the image. On the first `load_model` call earth2studio
downloads the package from Hugging Face into `$EARTH2STUDIO_CACHE`.

| Item | Value | Source |
|---|---|---|
| HF repo | `nvidia/corrdiff-cosmo-era5` | ungated, OpenMDW-1.1 |
| Total download | 2.33 GB, 19 files | HF API |
| REA6 checkpoints | `rea6/diffusion.mdlus` 0.70 GB, `rea6/regression.mdlus` 0.39 GB | HF API |
| REA2 checkpoints | `rea2/diffusion.mdlus` 0.70 GB, `rea2/regression.mdlus` 0.39 GB | HF API |
| Static invariants | `rea2/invariants_rea2_ext.nc` 0.12 GB, `rea6/invariants_rea6_ext.nc` 0.02 GB | HF API |
| REA6 grid / outputs | 824 × 848 at 6 km, 45 variables | model card |
| REA2 grid / outputs | 780 × 724 at 2.2 km, 22 variables | model card |
| Parameters | 9.8 × 10⁷ (regression mode), 1.7 × 10⁸ (diffusion mode) | model card |
| Diffusion sampler | 18-step deterministic (Heun) | model card |
| GPU | 1 × H100 80 GB | see caveat |

**On the GPU row.** The networks are small — under 200M parameters — but the
activations are not: the DiT attends over a 824 × 848 field, and diffusion mode
runs 18 sampler steps per sample. Peak allocation has not been measured here
and scales with the number of ensemble samples. `--mem=64G` in the job script
is host RAM. Both numbers are starting points to be replaced by measurements.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node, pointing enroot's scratch directories off the
50 GB home:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+corrdiff+v1.sqsh \
    'docker://ghcr.io#bradleylab/corrdiff:v1'
```

Pre-stage the package on the login node, which has network access:

```bash
EARTH2STUDIO_CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache \
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+corrdiff+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; python -c "
from earth2studio.models.dx import CorrDiffCosmoEra5
p = CorrDiffCosmoEra5.load_default_package()
for f in (\"config.json\", \"rea6/metadata.json\", \"rea6/stats.json\",
          \"rea6/grids.nc\", \"rea6/invariants_norm_stats.json\",
          \"rea6/invariants_rea6_ext.nc\", \"rea6/regression.mdlus\",
          \"rea6/diffusion.mdlus\"):
    p.resolve(f)
"'
```

That stages REA6 only (about 1.2 GB of the 2.33 GB). Repeat with the `rea2/`
paths if the 2.2 km model is wanted.

Then submit a single-H100 job:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J corrdiff

SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+corrdiff+v1.sqsh

srun --container-image=$SQSH \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc '
       export PYTHONNOUSERSITE=1
       export EARTH2STUDIO_CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache
       export EARTH2STUDIO_DATA_CACHE=/scratch2/fs1/alexander.s.bradley/earth2studio-data
       python /storage3/fs1/alexander.s.bradley/Active/scripts/corrdiff_downscale.py
     '
```

Three lines in that script are lab-standard and easy to drop:

- `PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
  into the container, so a stray `pip install --user` on the login node would
  otherwise shadow the container's site-packages.
- `EARTH2STUDIO_CACHE` is where the model package lands. Storage3 keeps the
  download across jobs.
- `EARTH2STUDIO_DATA_CACHE` splits the ERA5 input cache off onto scratch, since
  those pulls are large and regenerable.

No `#SBATCH --exclude=` line: every Compute2 GPU is an H100, so there is
nothing to exclude.

## Minimal inference (Python API — no CLI)

Downscale one ERA5 timestep to the COSMO-REA6 6 km grid, deterministic mode:

```python
import numpy as np

from earth2studio.data import ARCO, fetch_data
from earth2studio.models.dx import CorrDiffCosmoEra5

package = CorrDiffCosmoEra5.load_default_package()
model = CorrDiffCosmoEra5.load_model(package, mode="mean", resolution="rea6")
model = model.to("cuda")

# ERA5, anonymous read from Google Cloud. interp_to regrids the global fields
# onto the footprint the model was trained on.
x, coords = fetch_data(
    source=ARCO(),
    time=np.array([np.datetime64("2020-02-10T12:00")]),
    variable=model.input_coords()["variable"],
    device="cuda",
    interp_to=model.input_coords(),
)

out, out_coords = model(x, coords)   # (time, variable, rea6 grid)
```

Switch `mode="diffusion"` for the generative ensemble mode, which samples
stochastic realizations with the 18-step sampler and gives sharper fields plus
a handle on uncertainty. `resolution="rea2"` selects the 2.2 km checkpoint.

Downscaling a *forecast* rather than a reanalysis timestep is the
`earth2studio.run.diagnostic` workflow, which couples a prognostic model to
this diagnostic one — that needs a global forecast model in the same
environment, which this image deliberately does not carry (one model per
container). Run the prognostic step in `fourcastnet3` and hand its output here.

For a sub-region rather than the full European domain, `model.set_domain(lat_min,
lat_max, lon_min, lon_max)` returns a **new** model sharing the same weights but
with the grid and static invariants sliced to that box. The DiT is crop-size
agnostic at fixed resolution, so any sub-region runs in a single forward. Boxes
reaching into the extended invariant margin work but emit a one-time
out-of-distribution warning; boxes beyond that margin raise.

## Licensing

**This is the one model in this batch whose weights are not Apache-2.0.** Two
licences apply to different files in the same repo.

### OpenMDW-1.1 (the model weights)

The Linux Foundation's OpenMDW License Agreement version 1.1, a permissive
licence written specifically for model weights rather than for source code. It
governs the "Model Materials": the model architecture and parameters plus all
related artifacts distributed with them.

What it grants: permission to deal in the Model Materials **without
restriction** — commercial and non-commercial use, modification, distribution —
under copyright, patent, database, **and trade secret** rights. That last one
has no Apache-2.0 equivalent.

What it requires, and this is the whole of it:

1. **If you redistribute any portion of the Model Materials, ship a copy of the
   OpenMDW agreement with it, and retain the copyright and origin notices that
   came with the files you are redistributing.** This is why the weights are
   fetched at runtime rather than baked into the image: publishing a container
   layer containing the checkpoints would be redistribution, and would put this
   obligation on every `docker pull`.
2. **A patent/copyright retaliation clause.** Filing, maintaining, or
   voluntarily joining a lawsuit alleging that the Model Materials infringe a
   patent or copyright terminates all rights granted to you, unless that suit
   is a response to one brought against you first.

What it explicitly does **not** do, and where it differs from what people
assume of a model licence: it imposes **no restrictions or obligations on
outputs**. Fields produced by running this model carry no downstream licence
condition — no attribution requirement, no share-alike, no field-of-use limit.
There is also no acceptable-use policy and no acceptance step; the repo is
ungated.

Practical differences from Apache-2.0, since that is the obvious comparison:
OpenMDW additionally grants trade-secret rights, defines its subject matter as
model materials rather than source, has **no NOTICE-file mechanism** and no
modified-files marking requirement, and its retaliation trigger covers
copyright claims, not just patent claims.

### CC BY 4.0 (the static invariants)

`rea6/invariants_rea6_ext.nc` and `rea2/invariants_rea2_ext.nc` are carved out
of the OpenMDW grant and licensed **CC BY 4.0** instead, with attribution
obligations listed in the repo's `ATTRIBUTION.md`. They are a derived product:
NVIDIA resampled public datasets onto the target grids and computed land
fraction, surface roughness, continentality, distance-to-coast and terrain
slopes. Per-channel sources are GLOBE v1.0 (NOAA NGDC, public domain) for
elevation and slopes, and ESA WorldCover 2021 v200 (CC BY 4.0) for land
fraction, roughness, continentality and distance-to-coast.

CC BY 4.0 attaches to **redistribution of those files or of adaptations of
them**, not to model outputs. If a paper reproduces or redistributes the
invariant fields, cite ESA WorldCover (Zanaga et al. 2022,
doi:10.5281/zenodo.7254221) and GLOBE v1.0 (doi:10.7289/V52R3PMS) and reproduce
`ATTRIBUTION.md`'s notices.

### Code

`earth2studio`, `physicsnemo` — Apache-2.0. `natten` — check the `LICENSE` and
`NOTICE` files in the installed package before redistributing anything built on
it. This image ships only code; weights are fetched at runtime and are never
part of a published layer.

## Caveats

- **Europe only, and validated only in-domain.** The model was trained on the
  COSMO-REA6 and REA2 domains. NVIDIA's card states validation is in-domain on
  the native grids and that use over the extended margin or other regions is
  unvalidated. It cannot be pointed at North America.
- **Input data is free.** The model needs 47 ERA5 surface and pressure-level
  variables over the domain. `ARCO()` reads the Analysis-Ready
  Cloud-Optimized ERA5 mirror on Google Cloud anonymously — no account, no key,
  no cost. `CDS()` reaches the same reanalysis through Copernicus and does
  require a free account plus an API key; there is no reason to prefer it. The
  static invariants and grids ship inside the model package, so nothing else
  needs sourcing.
- **The cosmo extra cannot be installed with plain pip, at all.** `pip install
  earth2studio[cosmo]` fails: the extra requires `nvidia-physicsnemo>=2.2.0a0`,
  a version that has never been published to PyPI. Upstream resolves it through
  a `[tool.uv.sources]` git entry that pip does not read. The reason is
  specific — the DiT here uses `attention_backend="natten2d_rope"`, which
  exists on physicsnemo main and is absent from the v2.1.1 tag (verified by
  diffing `physicsnemo/models/dit/dit.py` at both refs). The Dockerfile pins
  upstream's own SHA, `ced75d93d014f70bb691372788eee2d201171c12`, and the smoke
  test reads it back from pip's `direct_url.json` so a drifted pin fails the
  build rather than shipping.
- **When PhysicsNeMo 2.2.0 reaches PyPI, revisit the pin.** Upstream's own
  `TODO(cosmo)` says the git pin goes away then. Moving to a release also
  removes the conflict earth2studio declares between `cosmo` and the `all` /
  `da-healda` extras, which cap physicsnemo at 2.1.1.
- **NATTEN is the build-cost problem in this image.** It publishes an sdist
  only — no wheels on PyPI at any version — so every build compiles CUTLASS
  kernels from source. The layer is limited to sm_90 with 4 workers, which is
  what a standard GitHub-hosted runner tolerates, but this is by far the
  longest step and the most likely thing to hit a CI time limit. If it does,
  the fix is a larger runner or a prebuilt base layer, not widening the arch
  list.
- **NATTEN can also install with no CUDA kernels and not tell you.** With no
  GPU visible and `NATTEN_CUDA_ARCH` unset, its `setup.py` prints "Building
  WITHOUT libnatten" and *succeeds*. The result imports fine and fails only
  when a DiT block runs on a GPU, hours into a job. The Dockerfile sets the
  variable and the smoke test asserts `natten.HAS_LIBNATTEN`.
- **Kernels are compiled for sm_90 only.** Matches Compute2's H100s. This image
  will not run on an older GPU without a rebuild.
- **physicsnemo sets the torch floor.** It declares `torch>=2.10.0`; the image
  pins torch 2.11.0 and holds it with a pip constraint file so no later resolve
  can silently replace the GPU stack — the failure mode the cuequivariance
  0.11.1 incident produced.
