# fourcastnet3 — Compute2 smoke test

The smallest job that proves the image works end to end on an H100: the
checkpoint loads, the network reaches the GPU, and one 6-hour step produces a
finite field of the right shape.

**The numbers this job produces are meaningless.** It steps a field of zeros,
not an atmospheric state. It answers "does the stack run", not "does the model
forecast". Do not read anything into the output values.

## Step 0 — pre-stage the weights (login node, no allocation)

Compute nodes may have no outbound network. Fetch the 2.85 GB package first, so
the GPU job never touches the network and never burns an allocation discovering
that it cannot.

```bash
SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+fourcastnet3+v1.sqsh
CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache

srun --container-image=$SQSH \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc "
       export PYTHONNOUSERSITE=1
       export EARTH2STUDIO_CACHE=$CACHE
       python -c '
from earth2studio.models.px import FCN3
p = FCN3.load_default_package()
print(p.resolve(\"training_checkpoints/best_ckpt_mp0.tar\"))
print(p.resolve(\"orography.nc\"))
'
     "
```

**Success:** two absolute paths under `$CACHE/fcn3/`, and
`du -sh $CACHE/fcn3` reporting roughly 2.9 GB.

## Step 1 — one forward step on an H100

Save as `fcn3_smoke.sbatch` under
`/storage3/fs1/alexander.s.bradley/Active/c2_jobs/`:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -J fcn3-smoke
#SBATCH -o fcn3-smoke-%j.out

SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+fourcastnet3+v1.sqsh

srun --container-image=$SQSH \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc '
export PYTHONNOUSERSITE=1
export EARTH2STUDIO_CACHE=/storage3/fs1/alexander.s.bradley/Active/earth2studio-cache
python - <<"PY"
from collections import OrderedDict
import numpy as np, torch
from torch_harmonics.disco import cuda_kernels_is_available
from earth2studio.models.px import FCN3

assert torch.cuda.is_available(), "no CUDA device visible"
print("gpu:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__, "| disco cuda kernels:", cuda_kernels_is_available())

model = FCN3.load_model(FCN3.load_default_package()).to("cuda")

# Coordinates are taken from the model, not hardcoded, so this test cannot
# quietly pass against a checkpoint with a different grid or variable set.
# "batch" is dropped: it is a placeholder in input_coords() and earth2studio's
# batch decorator handles its absence, matching what fetch_data() returns.
coords = OrderedDict(
    (k, v) for k, v in model.input_coords().items() if k != "batch"
)
coords["time"] = np.array([np.datetime64("2024-07-01T00:00")])
x = torch.zeros([len(v) for v in coords.values()], device="cuda")
print("in shape:", tuple(x.shape), "| variables:", len(coords["variable"]))

out, out_coords = model(x, coords)
print("out shape:", tuple(out.shape))
print("lead_time:", out_coords["lead_time"])
print("finite:", bool(torch.isfinite(out).all()))
print("peak GPU alloc (GiB): %.2f" % (torch.cuda.max_memory_allocated() / 2**30))
PY
'
```

Submit with `sbatch fcn3_smoke.sbatch`.

## Success criteria

The job exits 0 and `fcn3-smoke-<jobid>.out` contains:

| Line | Expected |
|---|---|
| `gpu:` | an H100 |
| `disco cuda kernels:` | `True` — if `False`, torch-harmonics shipped without its CUDA kernels and the image needs a rebuild |
| `in shape:` | `(1, 1, 72, 721, 1440)` — time, lead_time, variable, lat, lon |
| `out shape:` | identical to `in shape:` (FCN3 maps a state to a state on the same grid) |
| `lead_time:` | a single 6-hour timedelta |
| `finite:` | `True` |
| `peak GPU alloc (GiB):` | any number — **record it**, this is the first real measurement of what the model costs |

Any Python traceback is a failure, including a
`ModuleNotFoundError` for `makani` (the fcn3 extra did not resolve) or an
`OptionalDependencyError` (earth2studio found the extra incomplete at call
time). Both should have been caught by the build-time smoke test; seeing one
here means the image on GHCR is not the image the Dockerfile describes.

## Expected runtime

**Not yet measured.** The 00:20:00 request is a bound chosen to be comfortably
larger than a single forward pass plus checkpoint load, not an estimate. After
the first successful run, replace this paragraph with the `sacct -j <jobid>
--format=Elapsed,MaxRSS` result and the peak GPU allocation printed above.

## What this test does not cover

Fetching real initial conditions from ARCO or GFS, autoregressive rollout
stability over many steps, and ensemble generation. The first genuine forecast
job should start from `ARCO()` and run at least four steps, so that rollout
drift would show.
