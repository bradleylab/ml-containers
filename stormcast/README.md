# stormcast — StormCast v1 convection-allowing regional nowcast

NVIDIA's kilometre-scale regional model, run through the Earth-2 Studio
framework. It steps the atmosphere forward one hour at a time on the HRRR 3 km
CONUS grid — fine enough to resolve convection rather than parameterize it —
while a coarse global forecast supplies the large-scale conditioning at each
step, keeping the regional solution anchored to the synoptic flow.

Each step runs twice: a deterministic U-Net predicts the next state, then an
EDM-preconditioned diffusion model refines it over 18 sampler steps.

- Framework: https://github.com/NVIDIA/earth2studio (Apache-2.0)
- Networks: https://github.com/NVIDIA/physicsnemo (Apache-2.0)
- Weights: https://huggingface.co/nvidia/stormcast-v1-era5-hrrr —
  **Apache-2.0**, ungated
- Paper: https://arxiv.org/abs/2408.10958

> **Status: experimental.** Not yet run against lab data. No runtime or
> peak-memory figure below has been measured on Compute2 — `SMOKE.md` is the
> job that would produce the first ones.

## Image tag

`ghcr.io/bradleylab/stormcast:latest` (also `:v1`, `:e2s0.17-cu128`)

## Contents

- `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`, Python 3.12, plus
  torch 2.11.0 + torchvision 0.26.0 from the PyPI cu128 wheels.
- `earth2studio==0.17.0` with everything its `stormcast` extra declares.
- `nvidia-physicsnemo==2.1.1` — pinned; earth2studio asks only for `>=2.0`.
- No wrapper CLI. Earth-2 Studio is a Python API; run inference from a script.

This is the only one of the three Earth-2 images that compiles nothing from
source, so it is also the fastest to build.

## Checkpoints and GPU requirements

Nothing is baked into the image. On the first `load_model` call earth2studio
downloads the package from Hugging Face into `$EARTH2STUDIO_CACHE`.

| Item | Value | Source |
|---|---|---|
| HF repo | `nvidia/stormcast-v1-era5-hrrr` | ungated, Apache-2.0 |
| Revision earth2studio pins | `6c89a0877a0d6b231033d3b0d8b9828a6f833ed8` | `earth2studio/models/px/stormcast.py` |
| Total download | 0.80 GB, 7 files | HF API |
| Diffusion net | `EDMPrecond.0.0.mdlus`, 0.48 GB | HF API |
| Regression net | `StormCastUNet.0.0.mdlus`, 0.31 GB | HF API |
| Normalization / grid | `metadata.zarr.zip`, `model.yaml`, `config.json` | HF API |
| Grid | HRRR CONUS, 3 km | model card |
| State variables | 99 in, 99 out | model card |
| Conditioning variables | 26, from a global forecast | model card |
| Step | 1 h | `StormCast.output_coords()` |
| Diffusion sampler | 18 steps (`sampler_steps` argument, default 18) | `StormCast.load_model()` |
| GPU | 1 × H100 80 GB | see caveat |

**On the GPU row.** The two networks together are under 0.8 GB on disk, but
they run on a full CONUS field at 3 km with 99 channels, and every forecast
hour costs 18 diffusion passes rather than one. Runtime scales linearly with
lead time in a way the global models do not. Peak GPU allocation has not been
measured here; `--mem=64G` in the job script is host RAM. Treat both as
starting points and replace them with numbers from the first successful run.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node, pointing enroot's scratch directories off the
50 GB home:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+stormcast+v1.sqsh \
    'docker://ghcr.io#bradleylab/stormcast:v1'
```

Pre-stage the weights on the login node, which has network access:

```bash
EARTH2STUDIO_CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache \
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+stormcast+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; python -c "
from earth2studio.models.px import StormCast
p = StormCast.load_default_package()
for f in (\"model.yaml\", \"config.json\", \"metadata.zarr.zip\",
          \"StormCastUNet.0.0.mdlus\", \"EDMPrecond.0.0.mdlus\"):
    p.resolve(f)
"'
```

Then submit a single-H100 job:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -J stormcast

SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+stormcast+v1.sqsh

srun --container-image=$SQSH \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc '
       export PYTHONNOUSERSITE=1
       export EARTH2STUDIO_CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache
       export EARTH2STUDIO_DATA_CACHE=/scratch2/fs1/alexander.s.bradley/earth2studio-data
       python /storage3/fs1/alexander.s.bradley/Active/scripts/stormcast_nowcast.py
     '
```

Three lines in that script are lab-standard and easy to drop:

- `PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
  into the container, so a stray `pip install --user` on the login node would
  otherwise shadow the container's site-packages.
- `EARTH2STUDIO_CACHE` is where the model package lands. Storage3 keeps the
  0.80 GB download across jobs.
- `EARTH2STUDIO_DATA_CACHE` splits the input-data cache off onto scratch. This
  matters more here than for the global models: StormCast pulls both HRRR and
  GFS fields for every forecast hour, so the data cache grows much faster than
  the weights ever will.

Note the walltime is longer than the global-model jobs. Each forecast hour is
18 diffusion passes over a CONUS-sized field, and there is no measured
per-step timing yet to size this from.

No `#SBATCH --exclude=` line: every Compute2 GPU is an H100, so there is
nothing to exclude.

## Minimal inference (Python API — no CLI)

A 12-hour nowcast from an HRRR analysis, conditioned on the GFS forecast,
written to a local Zarr store:

```python
from datetime import datetime

from earth2studio.data import GFS_FX, HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast
from earth2studio.run import deterministic

package = StormCast.load_default_package()
model = StormCast.load_model(package, conditioning_data_source=GFS_FX())

deterministic(
    time=[datetime(2024, 5, 21, 0)],
    nsteps=12,                       # 12 x 1 h
    prognostic=model,
    data=HRRR(),                     # initial state, anonymous read from AWS
    io=ZarrBackend(file_name="stormcast_2024052100.zarr"),
)
```

Two data sources, not one, and this is the thing that trips people up.
`data=HRRR()` supplies the initial atmospheric state; `conditioning_data_source`
supplies the coarse global forecast that constrains every subsequent step.
`GFS_FX()` is the default and reads NOAA's forecast archive on AWS
anonymously. Both need to cover the forecast window, or the run fails partway
through rather than at the start.

`sampler_steps` on `load_model` trades sharpness for time; it defaults to 18.
Lowering it is the first knob to reach for when a run does not fit the
walltime, and it is an analytical change, not a performance tweak — record it.

## Licensing

- **Weights** (`nvidia/stormcast-v1-era5-hrrr`): Apache-2.0, ungated. No
  acceptance step, no token, no use restriction beyond Apache's notice and
  patent terms.
- **Framework** (`earth2studio`, `physicsnemo`): Apache-2.0.
- This image ships only code. Weights are fetched at runtime and are never part
  of a published layer.

## Caveats

- **CONUS only.** The model lives on the HRRR grid. There is no knob that moves
  it to another domain; a different region means a different model.
- **Input data is free, from two sources.** HRRR analyses and GFS forecasts are
  both public NOAA products on AWS, read anonymously by earth2studio — no
  account, no key, no egress charge. Neither is a Copernicus-style gated
  dataset.
- **HRRR's archive is shallow and its variables are not the usual ones.**
  StormCast needs 99 state variables including hybrid/model-level winds,
  temperature, humidity, geopotential and pressure (`*_hl*`) plus composite
  reflectivity. These come from HRRR's native levels, not from pressure-level
  products, so a substitute reanalysis will not stand in. Check availability
  for the target date before committing a job.
- **Cost grows with lead time in a way the global models' does not.** Every
  forecast hour is one regression pass plus 18 diffusion passes, and every hour
  also pulls a fresh conditioning field. A 48 h run is not four times a 12 h
  run in wall-clock terms once data fetching is included.
- **physicsnemo is pinned, deliberately.** earth2studio asks for
  `nvidia-physicsnemo>=2.0` with no ceiling; left alone, two builds a month
  apart would ship different networks. 2.1.1 is the newest release and was
  checked to carry the two modules StormCast imports
  (`physicsnemo.diffusion.preconditioners.legacy.EDMPrecond` and
  `physicsnemo.models.diffusion_unets.StormCastUNet`) before pinning. The smoke
  test asserts that earth2studio recorded no optional-dependency failure for
  the StormCast module, which is what a future physicsnemo reshuffle would
  break.
- **physicsnemo also sets the torch floor.** It declares `torch>=2.10.0`. The
  image pins torch 2.11.0 and holds it with a pip constraint file so no later
  resolve can silently replace the GPU stack — the failure the cuequivariance
  0.11.1 incident produced.
- **`tensordict` is a soft spot.** physicsnemo 2.1.1 requires
  `tensordict>=0.11.0,<0.12`, and tensordict ships small compiled wheels built
  against a particular torch. Its own metadata declares an unpinned `torch`, so
  pip will not warn if the pairing is off. The smoke test prints the resolved
  tensordict version; if a future torch bump produces odd C-extension errors,
  look here first.
