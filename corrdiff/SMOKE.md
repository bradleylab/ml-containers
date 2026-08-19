# corrdiff — Compute2 smoke test

The smallest job that proves the image works end to end on an H100: the REA6
regression checkpoint loads, the NATTEN kernels execute on the GPU, and one
downscaling pass produces a finite field on the 6 km COSMO grid.

**The numbers this job produces are meaningless.** It downscales a field of
zeros, not an ERA5 state. It answers "does the stack run", not "does the model
downscale". Do not read anything into the output values.

The regression (`mode="mean"`) path is deliberately the one tested: it is a
single forward pass, where diffusion mode is 18 sampler steps. A separate
diffusion run is worth doing once, but not as the first thing.

## Step 0 — pre-stage the REA6 files (login node, no allocation)

Compute nodes may have no outbound network, and the full package is 2.33 GB
across both resolutions. Staging REA6 only is about 1.2 GB.

```bash
SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+corrdiff+v1.sqsh
CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache

srun --container-image=$SQSH \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc "
       export PYTHONNOUSERSITE=1
       export EARTH2STUDIO_CACHE=$CACHE
       python -c '
from earth2studio.models.dx import CorrDiffCosmoEra5
p = CorrDiffCosmoEra5.load_default_package()
for f in (\"config.json\", \"rea6/metadata.json\", \"rea6/stats.json\",
          \"rea6/grids.nc\", \"rea6/invariants_norm_stats.json\",
          \"rea6/invariants_rea6_ext.nc\", \"rea6/regression.mdlus\"):
    print(p.resolve(f))
'
     "
```

**Success:** seven absolute paths under `$CACHE/corrdiff_cosmo_era5/`, and
`du -sh` on that directory reporting roughly 0.5 GB (regression checkpoint plus
invariants and grids; the 0.70 GB diffusion checkpoint is not staged here).

## Step 1 — one downscaling pass on an H100

Save as `corrdiff_smoke.sbatch` under
`/storage3/fs1/alexander.s.bradley/Active/c2_jobs/`:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -J corrdiff-smoke
#SBATCH -o corrdiff-smoke-%j.out

SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+corrdiff+v1.sqsh

srun --container-image=$SQSH \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc '
export PYTHONNOUSERSITE=1
export EARTH2STUDIO_CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache
python - <<"PY"
from collections import OrderedDict
import numpy as np, torch
from natten import HAS_LIBNATTEN
from earth2studio.models.dx import CorrDiffCosmoEra5

assert torch.cuda.is_available(), "no CUDA device visible"
assert HAS_LIBNATTEN, "natten has no CUDA kernels — rebuild the image"
print("gpu:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__, "| libnatten:", HAS_LIBNATTEN)

package = CorrDiffCosmoEra5.load_default_package()
model = CorrDiffCosmoEra5.load_model(package, mode="mean", resolution="rea6").to("cuda")

# Coordinates come from the model, not hardcoded, so this cannot quietly pass
# against a checkpoint with a different grid or variable set. "batch" is a
# placeholder in input_coords(); earth2studio handles its absence.
coords = OrderedDict(
    (k, v) for k, v in model.input_coords().items() if k != "batch"
)
coords["time"] = np.array([np.datetime64("2020-02-10T12:00")])
x = torch.zeros([len(v) for v in coords.values()], device="cuda")
print("in shape:", tuple(x.shape), "| era5 variables:", len(coords["variable"]))

out, out_coords = model(x, coords)
print("out shape:", tuple(out.shape))
print("out variables:", len(out_coords["variable"]))
print("finite:", bool(torch.isfinite(out).all()))
print("peak GPU alloc (GiB): %.2f" % (torch.cuda.max_memory_allocated() / 2**30))
PY
'
```

Submit with `sbatch corrdiff_smoke.sbatch`.

## Success criteria

The job exits 0 and `corrdiff-smoke-<jobid>.out` contains:

| Line | Expected |
|---|---|
| `gpu:` | an H100 |
| `libnatten:` | `True` — if `False` the image was built without CUDA kernels and must be rebuilt |
| `in shape:` | leading `(1, 47, ...)` for time and the 47 ERA5 input variables |
| `out shape:` | a 6 km COSMO-REA6 field, 824 × 848 in its last two dimensions |
| `out variables:` | `45` |
| `finite:` | `True` |
| `peak GPU alloc (GiB):` | any number — **record it**, this is the first real measurement of regression-mode cost |

Any Python traceback is a failure. Two are worth distinguishing:

- A CUDA error from inside a NATTEN kernel means the kernels were compiled for
  the wrong architecture. The image targets sm_90 only; nothing older will run
  it.
- An `OptionalDependencyError` naming the `cosmo` group means the git-pinned
  physicsnemo did not resolve. This should have been impossible — the
  build-time smoke test reads the commit back from pip's `direct_url.json` — so
  seeing it here means the image on GHCR is not the image the Dockerfile
  describes.

## Expected runtime

**Not yet measured.** The 00:20:00 request is a bound chosen to be comfortably
larger than a checkpoint load plus one regression forward pass, not an
estimate. After the first successful run, replace this paragraph with the
`sacct -j <jobid> --format=Elapsed,MaxRSS` result and the peak GPU allocation
printed above.

Diffusion mode will be roughly an order of magnitude slower per field — 18
sampler steps instead of one — so size that job from the measurement here
rather than from this request.

## What this test does not cover

Real ERA5 input from ARCO, diffusion mode and its sampler, the REA2 2.2 km
checkpoint, and `set_domain` sub-region slicing. The first genuine downscaling
job should pull one real ERA5 timestep through `fetch_data(source=ARCO(), ...,
interp_to=model.input_coords())` and compare the regression output against the
published COSMO-REA6 field for the same hour — that is the only check that
tells you the pipeline is correct rather than merely running.
