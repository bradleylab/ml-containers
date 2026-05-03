# treex

Container for **treeX** (Burmeister et al. 2025) — unsupervised tree
instance segmentation in dense forest point clouds. Multi-platform
(TLS / PLS / ULS), classical / deterministic, no learned components,
no weights.

- Paper: [arXiv:2509.03633](https://doi.org/10.48550/arXiv.2509.03633)
- Upstream package: [`ai4trees/pointtree`](https://github.com/ai4trees/pointtree)
- License: MIT (package), Apache-2.0 / MIT permissive deps

## What's in the image

- `python:3.11-slim` (Debian bookworm).
- `pointtree==1.0.1` from PyPI; the C++ extensions
  (`_tree_x_algorithm_cpp`, `_operations_cpp`) compile during install
  via scikit-build-core + pybind11.
- `pointtorch` and `circle_detection` reinstalled from upstream `main`
  to match the upstream Dockerfile (PyPI lags source).
- The `pyclesperanto-prototype` dep is pinned to `<0.24.5` so pip
  resolves to a release whose wheel does NOT pin `numpy<2.0.0`
  (versions ≥0.24.5 reintroduced that pin and would conflict with
  pointtree's `numpy>=2.3.0` requirement).
- Wrapper at `/opt/treex/scripts/run_treex.py` for one-shot tile
  segmentation against a LAS/LAZ input.

CPU-only. The companion `CoarseToFineAlgorithm` from the same package
needs torch + torch-scatter and a learned semantic-segmentation
checkpoint and is **not** supported by this container — only
`TreeXAlgorithm` (the unsupervised path).

## Pull

```
docker pull ghcr.io/bradleylab/treex:v1
```

Tags: `:latest`, `:v1`.

## Quick start (Docker)

```
docker run --rm \
  -v $PWD:/work \
  ghcr.io/bradleylab/treex:v1 \
  python /opt/treex/scripts/run_treex.py \
    --input /work/tile.laz \
    --output-dir /work/treex_out \
    --preset uls \
    --crs EPSG:6347
```

Outputs:

- `tile_treex.laz` — input points + `instance_id` column (instance ID
  -1 for points not assigned to any tree).
- `tile_treex_trunks.parquet` — detected trunk centers (x, y) and
  diameter at breast height.

## Compute2 (WashU RIS) via enroot + pyxis

Import the GHCR image to a `.sqsh` cache once on a login node. The
`+` substitutions are how enroot encodes `/` and `:` from the source
URI:

```
ENROOT_USER=alexander.s.bradley
SQSH=/storage1/fs1/${ENROOT_USER}/Active/c2_jobs/bradleylab+treex+v1.sqsh

enroot import \
  -o "$SQSH" \
  'docker://ghcr.io#bradleylab/treex:v1'
```

Submit a SLURM job. treeX is CPU; no `--gpus`. Memory and time scale
with point count — a 100×100 m UAV tile at ~89 pts/m² (≈890 k points,
~1.3 M after voxel downsampling lookahead) typically runs in single-
digit minutes on 8 modern Xeon/EPYC cores. Confirm with a single-tile
benchmark before fanning out a job array.

```
#!/usr/bin/env bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -J treex_tile
#SBATCH -o /scratch2/fs1/%u/treex/logs/%x-%j.out

set -euo pipefail
ENROOT_USER=alexander.s.bradley
SQSH=/storage1/fs1/${ENROOT_USER}/Active/c2_jobs/bradleylab+treex+v1.sqsh
TILE_IN=/scratch2/fs1/${ENROOT_USER}/sat-test/test_tiles/tile_-10_10.laz
OUT_DIR=/scratch2/fs1/${ENROOT_USER}/treex/out

srun --container-image="$SQSH" \
     --container-mounts=/scratch2/fs1/${ENROOT_USER}:/scratch2/fs1/${ENROOT_USER} \
     --container-workdir=/work \
     bash -lc "
       export PYTHONNOUSERSITE=1
       python /opt/treex/scripts/run_treex.py \
         --input $TILE_IN \
         --output-dir $OUT_DIR \
         --preset uls \
         --workers \$SLURM_CPUS_PER_TASK \
         --crs EPSG:6347
     "
```

`PYTHONNOUSERSITE=1` is required: enroot bind-mounts `$HOME`, which
makes `~/.local/lib/pythonX.Y/site-packages/` shadow the container's
packages and can break imports unexpectedly. Standard Compute2
container hygiene (see `~/.claude/rules/research-infrastructure.md`).

## Wrapper CLI

```
python /opt/treex/scripts/run_treex.py --help
```

Required:
- `--input` LAS / LAZ / PLY / CSV path
- `--output-dir`

Useful options:
- `--preset {tls,uls}` (default `uls`; matches the published
  TreeXPresetULS / TreeXPresetTLS)
- `--seed` (default 42)
- `--workers` (default -1, i.e. all CPU threads)
- `--crs` (e.g. `EPSG:6347` for our Tyson UAV tiles, used only for
  georeferencing intermediate visualization outputs)
- `--vis-dir` for intermediate visualization PNGs — slow, only useful
  on small tiles for debugging

## What treeX does (one-paragraph overview)

Unsupervised. Step 1 classifies terrain points with the Cloth
Simulation Filter (Zhang et al. 2016). Step 2 builds a rasterized
DTM from the terrain points and computes height-above-ground.
Step 3 detects stem cross-sections at multiple height layers via
M-estimator/RANSAC circle fitting (the `circle_detection` package).
Step 4 grows tree instances upward into the canopy via a region-
growing rule implemented in C++. Reported ULS F1 on Wytham Woods +
FOR-instance is 0.58 with the ULS preset (Burmeister et al. 2025,
Table 4).

There are no learned weights. Reproducibility is controlled by the
`random_seed` argument; the upstream README notes some operations
are not strictly deterministic even with the seed set, because some
transitive dependencies (numba JIT, OpenCL kernels) are not seedable.

## Limitations / known caveats

- This image supports `TreeXAlgorithm` only. `CoarseToFineAlgorithm`
  (urban / semantic-seg-driven) needs torch + torch-scatter + a
  pretrained checkpoint and would require a separate, much larger
  container variant.
- `pyclesperanto-prototype` is installed (declared dep) but the
  TreeXAlgorithm code path does not appear to call it. The image
  ships `ocl-icd-libopencl1` so `import pyopencl` succeeds, but
  no working OpenCL ICD is configured — adding `pocl-opencl-icd`
  later would enable any visualization or evaluation modules that
  rely on it.
- ULS performance (F1 = 0.58 on Wytham + FOR-instance) is materially
  lower than the TLS/PLS scores reported in the same paper. For
  closed-canopy hardwood UAV tiles like Tyson, expect noticeable
  recall gaps at small / suppressed stems; benchmark against
  AMS3D and SegmentAnyTree before treating any one method as
  authoritative.

## When NOT to use this container

- For learned (deep) UAV tree segmentation: prefer
  `ghcr.io/bradleylab/segment-any-tree-h100:v2` (DL, GPU).
- For TLS-specific stem detection + DBH:
  `ghcr.io/bradleylab/3dfin:v1` (also classical, narrower scope).
- For TLS DL instance segmentation:
  `ghcr.io/bradleylab/treelearn:v1` (DL, sm_90).
