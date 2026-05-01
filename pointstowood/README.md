# PointsToWood — H100 container

Deep-learning **semantic leaf–wood segmentation** of high-resolution
TLS point clouds, from tree base to branch tips. Owen, Allen, Grieve,
Wilkes, Lines 2025 (*arXiv:2503.04420*, in review).

- Upstream: https://github.com/harryjfowen/PointsToWood
- Paper: https://arxiv.org/abs/2503.04420
- Training data: diverse mature European forests, RIEGL VZ400i TLS;
  authors aggregate ForestSemantic (Mspace Lab 2024), LeWoS (Wang 2021),
  a Beijing plot-level set (Wan 2021), HeiDATA (Weiser 2024), and their
  own Zenodo plot-level dataset (Owen 2024).
- Architecture: PointNet / PointNeXt derivative with a gated reflectance
  integration module (uses lidar intensity where available).
- Classes: 2 — wood vs leaf.

## Image tag

`ghcr.io/bradleylab/pointstowood:latest` (also `:v1`,
`:torch2.5-cu121`)

## Contents

- PyTorch 2.5.1 + CUDA 12.1 (native H100 sm_90 via cu121 wheels)
- PyG ecosystem matched to torch 2.5.1+cu121
- PointsToWood repo at `/opt/PointsToWood/` on default branch
  `version1.0-paper`, which ships `pointstowood/model/global.pth`
  (73 MB) — the general cross-biome checkpoint — in-tree.

## Running on Compute2

Import once:
```bash
cd /storage1/fs1/alexander.s.bradley/Active/c2_jobs
ENROOT_SQUASH_OPTIONS="-comp gzip" \
  XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
  ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
  ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import docker://ghcr.io#bradleylab/pointstowood:latest
```

Run inference on a TLS cloud:
```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

srun --container-image=/storage1/.../bradleylab+pointstowood+latest.sqsh \
     --container-mounts=/scratch2/...:/scratch2/...,/storage1/...:/storage1/... \
     --container-writable \
     bash -lc '
        cd /opt/PointsToWood/pointstowood &&
        python3 predict.py \
          --point-cloud /scratch2/.../inputs/voxel02cm.laz \
          --model model/global.pth \
          --batch_size 8 \
          --is-wood 0.50 \
          --grid_size 2.0 4.0 \
          --min_pts 128 \
          --max_pts 16384
     '
```

## Input expectations

- Columns: `x y z` required. `reflectance` optional — the model uses
  intensity/reflectance when present via the gated integration module.
  If your TLS LAZ has intensity, keep it; if it's lossy-quantized to
  uint8 reflectance, that still helps.
- Output: per-point wood probability and binary wood/leaf label at the
  `--is-wood` threshold (0.5 default).

## Limitations vs FSCT / Sen-net

- **Two classes only** (wood, leaf). No ground/CWD/understory stratum.
  Upstream of stem-isolation workflows, run a separate ground classifier
  (SMRF, CSF, or FSCT's terrain head) first.
- **Training corpus is European.** No North American broadleaf in the
  public training set. Oak-hickory performance at Tyson is an empirical
  question.
- **Intensity helps but isn't required.** If your TLS has been
  voxel-downsampled in a way that drops reflectance (e.g. PDAL's
  `filters.voxelcenternearestneighbor` keeps a single point's
  attributes; simple gridding averages), check the LAZ before
  inference.
