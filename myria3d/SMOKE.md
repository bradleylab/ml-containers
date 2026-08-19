# myria3d — Compute2 smoke test

The smallest run that proves the image works on an H100: no real data, no
manual staging beyond the 13 MB checkpoint, and no dependence on any
particular lidar tile being available. Two stages, ~5 minutes of wall time
in total.

The build-time smoke tests in the Dockerfile already cover imports, the
model class, the PDAL/GDAL stack, and hydra config composition — all on
CPU, offline. What they cannot cover is the two things that only fail on
real hardware: **an sm_90 forward pass** and **loading the FRACTAL
checkpoint**. That is what this is for.

## Prerequisites

```bash
# once, on a login node
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+myria3d+v1.sqsh \
    'docker://ghcr.io#bradleylab/myria3d:v1'

# once, the checkpoint (13 MB) — see README "Weights" for the full form
mkdir -p /storage3/fs1/alexander.s.bradley/Active/weights/myria3d
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+myria3d+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley,/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc '
        export HF_HOME=/scratch2/fs1/alexander.s.bradley/hf-cache
        /opt/env/bin/hf download IGNF/FRACTAL-LidarHD_7cl_randlanet \
          FRACTAL-LidarHD_7cl_randlanet.ckpt \
          --revision 6ef0c46e11ab7fa9d20f3d9e39986c46dbd3814e \
          --local-dir /storage3/fs1/alexander.s.bradley/Active/weights/myria3d
     '
```

Verify the checkpoint before trusting anything downstream:

```bash
sha256sum /storage3/fs1/alexander.s.bradley/Active/weights/myria3d/FRACTAL-LidarHD_7cl_randlanet.ckpt
# expect 58baca3fbc00af2fa4af2a26cea345c08decbbba0215d21d6640c412a42e8cd1
```

## The job

Save as `myria3d_smoke.sbatch` and `sbatch` it.

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -J myria3d-smoke
#SBATCH -o myria3d-smoke-%j.log

set -euo pipefail

SCRATCH=/scratch2/fs1/alexander.s.bradley
STORAGE3=/storage3/fs1/alexander.s.bradley
WORK=$SCRATCH/myria3d_smoke
CKPT=$STORAGE3/Active/weights/myria3d/FRACTAL-LidarHD_7cl_randlanet.ckpt
mkdir -p "$WORK/in" "$WORK/out"

srun --container-image=$STORAGE3/Active/c2_jobs/bradleylab+myria3d+v1.sqsh \
     --container-mounts=$SCRATCH:$SCRATCH,$STORAGE3:$STORAGE3 \
     bash -lc "
set -euo pipefail
export PYTHONNOUSERSITE=1
cd $WORK

# ---- stage 0: GPU visible, checkpoint loads -------------------------------
/opt/env/bin/python - <<'PY'
import torch
from myria3d.models.model import Model

assert torch.cuda.is_available(), 'no CUDA device visible to the container'
print('GPU:', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))

model = Model.load_from_checkpoint('$CKPT', map_location='cpu').eval().to('cuda')
print('checkpoint loaded; parameters:', sum(p.numel() for p in model.parameters()))
PY

# ---- build a 50 m synthetic tile (no real data needed) --------------------
/opt/env/bin/python - <<'PY'
import laspy, numpy as np

SEED = 42
N = 100_000            # 50 x 50 m at 40 pts/m², the FRACTAL patch geometry
rng = np.random.default_rng(SEED)

hdr = laspy.LasHeader(version='1.4', point_format=6)   # format 6 has no RGB
hdr.offsets = np.array([700000.0, 4270000.0, 100.0])
hdr.scales  = np.array([0.01, 0.01, 0.01])
las = laspy.LasData(hdr)
las.x = 700000.0 + rng.uniform(0, 50, N)
las.y = 4270000.0 + rng.uniform(0, 50, N)
las.z = 100.0 + rng.uniform(0, 20, N)
las.intensity        = rng.integers(0, 2000, N).astype(np.uint16)
las.return_number    = np.ones(N, dtype=np.uint8)
las.number_of_returns= np.ones(N, dtype=np.uint8)
las.classification   = np.ones(N, dtype=np.uint8)      # 1 = unclassified
las.write('$WORK/in/smoke_tile.laz')
print('wrote synthetic tile:', N, 'points over 50 x 50 m')
PY

# ---- stage 1: end-to-end prediction --------------------------------------
/opt/env/bin/python /opt/myria3d/run.py \
  task.task_name=predict \
  predict.src_las=$WORK/in/smoke_tile.laz \
  predict.output_dir=$WORK/out \
  predict.ckpt_path=$CKPT \
  predict.gpus=1 \
  datamodule.batch_size=10 \
  datamodule.tile_width=50 \
  datamodule.epsg=6344

# ---- check the output ----------------------------------------------------
/opt/env/bin/python - <<'PY'
import laspy, numpy as np

las = laspy.read('$WORK/out/smoke_tile.laz')
dims = set(las.point_format.dimension_names)
for d in ('PredictedClassification', 'entropy'):
    assert d in dims, f'missing output dimension {d}: {sorted(dims)}'

pc = np.asarray(las['PredictedClassification'])
ent = np.asarray(las['entropy'])
codes, counts = np.unique(pc, return_counts=True)

print('points          :', len(pc))
print('predicted codes :', dict(zip(codes.tolist(), counts.tolist())))
print('entropy min/max :', float(ent.min()), float(ent.max()))
print('extra dims      :', sorted(d for d in dims if d in
      ('PredictedClassification', 'entropy', 'building', 'ground')))

assert set(codes.tolist()) <= {1, 2, 5, 6, 9, 17, 64}, codes
assert ent.max() > 0, 'entropy is uniformly zero — nothing was predicted'
print('SMOKE OK')
PY
"
```

## What proves success

| Check | Expected |
|---|---|
| Stage 0 GPU line | `capability (9, 0)` — an H100. Anything else means the allocation is wrong, not the image. |
| Stage 0 checkpoint | loads without a `torch.load` / `weights_only` error and prints a parameter count |
| Prediction log | includes `Color channel Red not found. Creating fake Red filled with 0.` — this is the no-colour path being exercised on purpose, not a failure |
| Output file | `$WORK/out/smoke_tile.laz` exists |
| Output dimensions | `PredictedClassification` and `entropy` present, plus `building` and `ground` probability dimensions (what the shipped FRACTAL config writes) |
| Predicted codes | a subset of `{1, 2, 5, 6, 9, 17, 64}` |
| Entropy | not uniformly zero |
| Final line | `SMOKE OK` |

**What this does not prove.** The input is uniform random noise, so the
*labels are meaningless* — a run where every point comes out `2` (ground)
is still a pass. This checks plumbing, GPU kernels and checkpoint loading,
nothing about accuracy. Accuracy on non-French data is unmeasured; see the
caveat at the top of the README.

## Expected runtime

Container start plus imports dominate. Stage 0 is a few tens of seconds
once the `.sqsh` is on local scratch; stage 1 on a 100k-point 50 m tile is
a single forward pass over one receptive field and should finish in
seconds of GPU time. **If the job has not printed `SMOKE OK` within ten
minutes, something is wrong** — most likely enroot pulling the image over
the network, or `$HOME` site-packages shadowing the container's (check
that `PYTHONNOUSERSITE=1` survived into the shell).

These are bounds, not measurements. Whoever runs this first: record the
real wall time and peak GPU memory here, and fill in the *not measured*
rows in the README's resource table.

## If it fails

| Symptom | Likely cause |
|---|---|
| `no CUDA device visible` | job did not get a GPU, or the container runtime's GPU hook did not fire — check `--gpus=1` and `nvidia-smi` inside the same `srun` |
| `ModuleNotFoundError` for a package the build installed | `$HOME` site-packages shadowing; confirm `PYTHONNOUSERSITE=1` is exported inside the container shell |
| `_pickle.UnpicklingError` / `weights_only` on checkpoint load | torch was bumped past 2.5 — see the README build notes; this is the failure that pin exists to prevent |
| `No EPSG provided` | drop `datamodule.epsg=` at your peril; the synthetic tile carries no CRS by design |
| Permission denied writing `outputs/` | the job did not `cd` into a writable directory before calling `run.py`; hydra writes its run directory under the current working directory |
| `RuntimeError` from `pdal info` | the input path is wrong or the file is not readable by PDAL — test with `pdal info --metadata <file>` inside the container |
