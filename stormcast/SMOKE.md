# stormcast — Compute2 smoke test

The smallest job that proves the image works end to end on an H100: both
networks load, and one 1-hour step produces a finite field on the HRRR CONUS
grid.

StormCast differs from its two sibling Earth-2 images in one way that shapes
this test. Its `__call__` reaches out to the conditioning data source on every
step — `fetch_data(self.conditioning_data_source, ...)` is inside the forward
path, not something the caller does beforehand. **A forward pass therefore
needs outbound network from wherever it runs.** Step 1 below assumes Compute2
compute nodes have egress; Step 1-alt is the weaker test to run if they do not.

## Step 0 — pre-stage the weights (login node, no allocation)

```bash
SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+stormcast+v1.sqsh
CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache

srun --container-image=$SQSH \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc "
       export PYTHONNOUSERSITE=1
       export EARTH2STUDIO_CACHE=$CACHE
       python -c '
from earth2studio.models.px import StormCast
p = StormCast.load_default_package()
for f in (\"model.yaml\", \"config.json\", \"metadata.zarr.zip\",
          \"StormCastUNet.0.0.mdlus\", \"EDMPrecond.0.0.mdlus\"):
    print(p.resolve(f))
'
     "
```

**Success:** five absolute paths under `$CACHE/stormcast/`, and `du -sh` on that
directory reporting roughly 0.8 GB.

## Step 1 — one 1-hour forecast step on an H100

Save as `stormcast_smoke.sbatch` under
`/storage3/fs1/alexander.s.bradley/Active/c2_jobs/`:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -J stormcast-smoke
#SBATCH -o stormcast-smoke-%j.out

SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+stormcast+v1.sqsh

srun --container-image=$SQSH \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc '
export PYTHONNOUSERSITE=1
export EARTH2STUDIO_CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache
export EARTH2STUDIO_DATA_CACHE=/scratch2/fs1/alexander.s.bradley/earth2studio-data
python - <<"PY"
from datetime import datetime
import numpy as np, torch
from earth2studio.data import GFS_FX, HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast
from earth2studio.run import deterministic

assert torch.cuda.is_available(), "no CUDA device visible"
print("gpu:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)

package = StormCast.load_default_package()
model = StormCast.load_model(package, conditioning_data_source=GFS_FX())
print("state variables:", len(model.variables),
      "| conditioning variables:", len(model.conditioning_variables))

io = deterministic(
    time=[datetime(2024, 5, 21, 0)],
    nsteps=1,                     # one 1 h step
    prognostic=model,
    data=HRRR(),
    io=ZarrBackend(),             # in-memory, nothing written to disk
)

arr = io["t2m"]
print("t2m shape:", tuple(arr.shape))
print("finite:", bool(np.isfinite(np.asarray(arr)).all()))
print("t2m range (K): %.1f %.1f" % (float(np.nanmin(arr)), float(np.nanmax(arr))))
print("peak GPU alloc (GiB): %.2f" % (torch.cuda.max_memory_allocated() / 2**30))
PY
'
```

Submit with `sbatch stormcast_smoke.sbatch`.

Unlike the fourcastnet3 and corrdiff smoke tests, this one runs on **real**
data, because a synthetic input would not exercise the conditioning fetch that
is the interesting part. That makes the printed `t2m range` weakly meaningful:
a plausible near-surface temperature range over CONUS in late May is a sign the
normalization buffers loaded correctly. It is a sanity check, not a validation
— do not quote it anywhere.

## Success criteria

The job exits 0 and `stormcast-smoke-<jobid>.out` contains:

| Line | Expected |
|---|---|
| `gpu:` | an H100 |
| `state variables:` | `99` |
| `conditioning variables:` | `26` |
| `t2m shape:` | two lead times (0 h and 1 h) over the HRRR CONUS grid |
| `finite:` | `True` |
| `t2m range (K):` | roughly 250–320 K. Anything far outside that means the normalization buffers in `metadata.zarr.zip` were not applied |
| `peak GPU alloc (GiB):` | any number — **record it**, this is the first real measurement of what a step costs |

Failure modes worth distinguishing:

- `RuntimeError: StormCast has been called without initializing the model's
  conditioning_data_source` — `load_model` was called without
  `conditioning_data_source`, or it was cleared.
- A network timeout or an empty-fetch error naming `noaa-hrrr` or `noaa-gfs` —
  either the compute node has no egress (use Step 1-alt) or the requested date
  falls outside what those archives hold. HRRR's public archive is shallow;
  check the date before blaming the image.
- An `OptionalDependencyError` naming the `stormcast` group — the pinned
  physicsnemo did not provide a symbol the model imports. This should have been
  caught at build time, so it means the image on GHCR is not the image the
  Dockerfile describes.

## Step 1-alt — no compute-node egress

If Step 1 fails on network rather than on compute, fall back to a load-only
check. It proves the checkpoints deserialize and reach the GPU; it does **not**
exercise a forward pass, so it is a weaker test and should be followed by a
Step 1 run somewhere with egress before the image is trusted.

Replace the Python heredoc in the job script with:

```python
import torch
from earth2studio.models.px import StormCast

assert torch.cuda.is_available(), "no CUDA device visible"
model = StormCast.load_model(StormCast.load_default_package()).to("cuda")

print("gpu:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)
print("state variables:", len(model.variables),
      "| conditioning variables:", len(model.conditioning_variables))
print("regression params:", sum(p.numel() for p in model.regression_model.parameters()))
print("diffusion params:", sum(p.numel() for p in model.diffusion_model.parameters()))
print("means finite:", bool(torch.isfinite(model.means).all()))
print("weights on:", next(model.regression_model.parameters()).device)
```

`load_model` without a conditioning source emits a warning by design; that is
expected here and is not a failure.

## Expected runtime

**Not yet measured.** The 00:30:00 request is a bound, not an estimate, and it
is longer than the sibling images' because most of the time will go to fetching
HRRR and GFS fields rather than to compute. After the first successful run,
replace this paragraph with the `sacct -j <jobid> --format=Elapsed,MaxRSS`
result, the peak GPU allocation printed above, and — separately — how much of
the elapsed time was data fetching, since that is what sets the walltime for
real jobs.

## What this test does not cover

Multi-hour rollout, where autoregressive drift and the growing conditioning
fetch both matter; `sampler_steps` other than the default 18; and any
comparison against observed HRRR fields. The first genuine job should run at
least 12 steps and verify against the HRRR analysis at the same valid times.
