# fourcastnet3 — FourCastNet 3 global AI weather forecast

NVIDIA's third-generation global weather model, run through the Earth-2 Studio
framework. Takes a single atmospheric state on the 0.25° lat/lon grid and rolls
it forward in 6-hour steps, predicting 72 surface and pressure-level variables
at each step. The architecture is a spherical neural operator with a stochastic
core, so it is built for ensemble forecasting rather than a single trajectory.

- Framework: https://github.com/NVIDIA/earth2studio (Apache-2.0)
- Network: https://github.com/NVIDIA/makani (Apache-2.0)
- Weights: https://huggingface.co/nvidia/fourcastnet3 — **Apache-2.0**, ungated
- Paper: https://arxiv.org/abs/2507.12144

> **Status: experimental.** Not yet run against lab data or validated on any
> lab-relevant question. Nothing below reports a measured runtime or a measured
> peak GPU memory on Compute2 — see `SMOKE.md` for the job that would produce
> the first numbers.

## Image tag

`ghcr.io/bradleylab/fourcastnet3:latest` (also `:v1`, `:e2s0.17-cu128`)

## Contents

- `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`, Python 3.12, plus
  torch 2.11.0 + torchvision 0.26.0 from the PyPI cu128 wheels.
- `earth2studio==0.17.0`.
- `torch_harmonics` built from NVIDIA's git at the revision earth2studio pins,
  **with the DISCO CUDA kernels compiled for sm_90**.
- `makani` from NVIDIA's git at the revision earth2studio pins — this is the
  package that actually holds the FCN3 network.
- `nvidia-physicsnemo==2.1.1`, pinned because makani asks for it without an
  upper bound.
- No wrapper CLI. Earth-2 Studio is a Python API; run inference from a script.

## Checkpoints and GPU requirements

Nothing is baked into the image. On the first `load_model` call earth2studio
downloads the package from Hugging Face into `$EARTH2STUDIO_CACHE`.

| Item | Value | Source |
|---|---|---|
| HF repo | `nvidia/fourcastnet3` | ungated, Apache-2.0 |
| Revision earth2studio pins | `76ef0c60237e458b33196ba027134e27f3fc4538` | `earth2studio/models/px/fcn3.py` |
| Total download | 2.85 GB, 11 files | HF API |
| Largest file | `training_checkpoints/best_ckpt_mp0.tar`, 2.84 GB | HF API |
| Other files | `orography.nc`, `land_mask.nc`, `global_means.npy`, `global_stds.npy`, `mins.npy`, `maxs.npy`, `config.json`, `metadata.json` | HF API |
| Grid | 721 × 1440 (0.25°, global) | `FCN3.input_coords()` |
| Variables | 72 | `FCN3.VARIABLES` |
| Step | 6 h | `FCN3.output_coords()` |
| GPU | 1 × H100 80 GB | see caveat |

**On the GPU row.** One H100 is what these jobs are sized for and a 2.84 GB
checkpoint on a 721 × 1440 × 72 state leaves ample headroom, but peak
allocation has not been measured here, and it grows with `batch` and with the
ensemble size. The `--mem=64G` in the job script below is host RAM, not GPU
memory; treat both as starting points to be replaced by measurements from the
first successful run.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node, pointing enroot's scratch directories off the
50 GB home:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+fourcastnet3+v1.sqsh \
    'docker://ghcr.io#bradleylab/fourcastnet3:v1'
```

Pre-stage the weights on the login node, which has network access — compute
nodes may not, and a job that discovers this after allocation wastes the
allocation:

```bash
EARTH2STUDIO_CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache \
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+fourcastnet3+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; python -c "
from earth2studio.models.px import FCN3
FCN3.load_default_package().resolve(\"training_checkpoints/best_ckpt_mp0.tar\")
"'
```

Then submit a single-H100 job:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J fcn3

SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+fourcastnet3+v1.sqsh

srun --container-image=$SQSH \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc '
       export PYTHONNOUSERSITE=1
       export EARTH2STUDIO_CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache
       export EARTH2STUDIO_DATA_CACHE=/scratch2/fs1/alexander.s.bradley/earth2studio-data
       python /storage3/fs1/alexander.s.bradley/Active/scripts/fcn3_forecast.py
     '
```

Three lines in that script are lab-standard and easy to drop:

- `PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
  into the container, so a stray `pip install --user` on the login node would
  otherwise shadow the container's site-packages.
- `EARTH2STUDIO_CACHE` is where the model package lands. Storage3 keeps the
  2.85 GB download across jobs.
- `EARTH2STUDIO_DATA_CACHE` splits the input-data cache off onto scratch. ERA5
  and GFS pulls are large and regenerable; they should not accumulate beside
  the weights.

No `#SBATCH --exclude=` line: every Compute2 GPU is an H100, so there is
nothing to exclude.

## Minimal inference (Python API — no CLI)

A 4-day forecast from ERA5 initial conditions, written to a local Zarr store:

```python
from datetime import datetime

from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN3
from earth2studio.run import deterministic

model = FCN3.load_model(FCN3.load_default_package())

deterministic(
    time=[datetime(2024, 7, 1)],
    nsteps=16,                       # 16 x 6 h = 96 h
    prognostic=model,
    data=ARCO(),                     # ERA5, anonymous read from Google Cloud
    io=ZarrBackend(file_name="fcn3_2024070100.zarr"),
)
```

`ARCO` is the Analysis-Ready Cloud-Optimized ERA5 mirror on Google Cloud;
earth2studio reads it anonymously, so the job needs no credentials. Swap it for
`GFS()` to initialize from an operational analysis instead.

FCN3's stochastic core is the point of the model, so a single deterministic
trajectory is a diagnostic, not the intended product. Ensembles run through
`earth2studio.run.ensemble` with a perturbation method. That path needs
earth2studio's `perturbation` extra, whose two requirements — `scipy` and
`torch-harmonics` at the same git revision the `fcn3` extra uses — are both
already satisfied in this image, so no extra install is needed.

## Licensing

- **Weights** (`nvidia/fourcastnet3`): Apache-2.0, ungated. No acceptance step,
  no token, no use restriction beyond Apache's notice and patent terms.
- **Framework** (`earth2studio`, `makani`, `physicsnemo`): Apache-2.0.
- **torch-harmonics**: BSD-3-Clause.
- This image ships only code. Weights are fetched at runtime and are never part
  of a published layer.

## Caveats

- **Input data is free but not local.** FCN3 needs one complete atmospheric
  state on the 0.25° grid: 72 variables including geopotential, temperature,
  humidity and winds on 13 pressure levels. Two free, no-account paths are
  wired into earth2studio and read anonymously — `ARCO()` (ERA5 on Google
  Cloud, reanalysis, ~5-day latency) and `GFS()` (NOAA on AWS, operational
  analysis, near real time). `CDS()` reaches ERA5 through Copernicus instead
  and does require a free account plus an API key; there is no reason to use it
  when ARCO covers the same data.
- **Fetching a single initial condition is not fast.** One timestep of 72
  variables is assembled from many chunked reads. Pre-stage initial conditions
  the same way as the weights when a job is on a tight walltime.
- **The fcn3 extra cannot be installed with plain pip.** `pip install
  earth2studio[fcn3]` fails: the extra lists `makani`, which has never been
  published to PyPI, and earth2studio resolves it through a `[tool.uv.sources]`
  git entry that pip does not read. The Dockerfile restates upstream's own git
  SHAs. When bumping earth2studio, re-read that table rather than assuming the
  pins carried over.
- **makani drags in physicsnemo, and that is what sets the torch floor.**
  makani declares `nvidia-physicsnemo>=0.5.0a0` with no ceiling; the current
  physicsnemo declares `torch>=2.10.0`. Left unpinned, a routine rebuild would
  quietly move both. This is the same class of failure as the cuequivariance
  0.11.1 incident, arriving one dependency level deeper. `physicsnemo==2.1.1`
  and a pip constraint file holding `torch==2.11.0` are the two guards.
- **torch-harmonics can build without its CUDA kernels and not tell you.** The
  build only compiles the DISCO kernels when it sees a GPU or when
  `FORCE_CUDA_EXTENSION=1` is set; CI runners have no GPU. Without the flag the
  install succeeds and FCN3 falls back to a slow path at inference. The
  Dockerfile sets the flag and the smoke test asserts
  `cuda_kernels_is_available()`.
- **Kernels are compiled for sm_90 only.** `TORCH_CUDA_ARCH_LIST=9.0` matches
  Compute2's H100s and keeps the build short. This image will not run on an
  older GPU without a rebuild.
- **Image size.** The `-devel` CUDA base plus a compiled torch-harmonics is
  large. The GitHub Actions workflow frees runner disk before building for this
  reason.
