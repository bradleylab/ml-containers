# depth-anything-3

Monocular depth, multi-view geometry, and camera pose from one model.

Lin, Yang et al. (2025), [Depth Anything 3](https://arxiv.org/abs/2511.10647).
Upstream: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3).

## What this is

Give it one image and it returns depth. Give it several views of the same
scene and it additionally recovers where the camera was for each — which is
the part that stands in for a structure-from-motion pass when you want a
quick reconstruction rather than a survey-grade one.

Per input image:

| Output | Shape |
|---|---|
| depth | `[N, H, W]` float32 |
| confidence | `[N, H, W]` float32 |
| camera pose (world-to-camera) | `[N, 3, 4]` float32 |
| camera intrinsics | `[N, 3, 3]` float32 |

## Why this replaced two other candidates

The 2026-08 vision sweep shortlisted Depth-Anything-V2 for monocular depth and
VGGT for multi-view geometry. DA3 does both, and it is Apache-2.0 where both of
those are CC-BY-NC-4.0. One permissive image instead of two restricted ones.

## The licence trap

**The code is Apache-2.0. The weights are not uniformly Apache-2.0.**

| Checkpoint | Licence |
|---|---|
| `DA3-LARGE-1.1` (baked here) | Apache-2.0 |
| `DA3MONO-LARGE`, `DA3METRIC-LARGE`, `DA3-BASE` | Apache-2.0 |
| `DA3-LARGE` | CC-BY-NC-4.0 |
| `DA3NESTED-GIANT-LARGE-1.1` | CC-BY-NC-4.0 |

Upstream's own README example loads `depth-anything/da3-large`, which redirects
to `DA3-LARGE` — non-commercial. Copying that example would put NC weights
inside an image labelled Apache-2.0.

This container bakes `DA3-LARGE-1.1`. If you swap the checkpoint, check the
licence of what you swapped to, and update the image label to match.

## Usage

```bash
docker run --rm --gpus all -v "$PWD":/work ghcr.io/bradleylab/depth-anything-3:v1 \
  python -c "
from depth_anything_3.api import DepthAnything3
m = DepthAnything3.from_pretrained('depth-anything/DA3-LARGE-1.1')
p = m.inference(['/work/img1.jpg', '/work/img2.jpg'], export_dir='/work/out')
"
```

### Compute2 (enroot)

```bash
srun -A compute2-alexander.s.bradley -p general-gpu \
     --gpus=1 --cpus-per-task=8 --mem=48G --time=00:30:00 \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+depth-anything-3+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; export HF_HUB_OFFLINE=1; python your_script.py'
```

Weights are baked, so `HF_HUB_OFFLINE=1` is safe and the job needs no network.

## Dependency notes

`depth-anything-3` declares 32 dependencies, including `open3d`, `pycolmap`
and `moviepy`, and pins `numpy<2`. The version is pinned exactly at 0.1.1 for
that reason — an unpinned install across that surface is how a working image
stops building a few months later. `libgl1`, `libglib2.0-0` and `libgomp1` are
installed because open3d, opencv and pycolmap dlopen them at import.

## Verification

The build smoke test runs offline on CPU: it imports the package, loads the
baked checkpoint with no network, asserts the parameter count and that numpy
stayed below 2.

**Not yet executed on Compute2.** See `SMOKE.md` for the smallest job that
would prove a real forward pass on an H100; it has not been run.
