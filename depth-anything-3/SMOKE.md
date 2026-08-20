# depth-anything-3 — Compute2 smoke test

The smallest run that proves the image works on an H100: **two synthetic
images through one forward pass**, checking that all four outputs come back
with the right shapes. Under a minute of GPU time, no data staging, no
network — the checkpoint is baked.

The build-time smoke test already covers imports and that the checkpoint
loads. What it cannot cover is what only fails on real hardware: an sm_90
forward pass, and whether the multi-view path actually returns poses rather
than only depth.

## 0. One-time: import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+depth-anything-3+v1.sqsh \
    'docker://ghcr.io#bradleylab/depth-anything-3:v1'
```

`enroot import` can exit 0 after its `mksquashfs` child is OOM-killed, so
check the artifact rather than the exit status:

```bash
file -b bradleylab+depth-anything-3+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. The test

Two 512×384 images of random noise. Noise is deliberate: the test is whether
the graph runs and returns the documented shapes on a GPU, not whether the
depth means anything. A model that silently returned only depth, or collapsed
the batch, fails here.

```bash
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=48G --time=00:20:00 \
       -J da3-smoke -o da3-smoke-%j.out --wrap='
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+depth-anything-3+v1.sqsh \
     --container-workdir=/tmp \
     bash -lc "export PYTHONNOUSERSITE=1; export HF_HUB_OFFLINE=1; python - <<PY
import numpy as np, torch
from PIL import Image
from depth_anything_3.api import DepthAnything3

assert torch.cuda.is_available(), \"no GPU visible to the job\"
print(\"gpu:\", torch.cuda.get_device_name(0))

rng = np.random.default_rng(42)
paths = []
for i in range(2):
    p = f\"/tmp/chip{i}.png\"
    Image.fromarray(rng.integers(0, 255, (384, 512, 3), dtype=np.uint8)).save(p)
    paths.append(p)

m = DepthAnything3.from_pretrained(\"depth-anything/DA3-LARGE-1.1\")
out = m.inference(paths, export_dir=\"/tmp/out\")
print(\"returned:\", type(out).__name__)
for name in (\"depth\", \"conf\", \"confidence\", \"pose\", \"intrinsics\"):
    v = getattr(out, name, None)
    if v is not None:
        print(f\"  {name}: {tuple(np.asarray(v).shape)}\")
print(\"SMOKE OK\")
PY"'
```

## 2. What passing means

- The package imports and the baked checkpoint loads with no network.
- A forward pass runs on sm_90.
- Depth comes back for both images, and the multi-view outputs — pose and
  intrinsics — are present rather than silently absent.

Record the measured wall time here after the first successful run; nothing
below has been timed on Compute2 yet.
