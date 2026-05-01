# TreeLearn — H100 container

Deep-learning **instance segmentation** of individual trees from
ground-based lidar point clouds (TLS / MLS). Henrich et al. 2024,
*Ecological Informatics*.

- Upstream: https://github.com/ecker-lab/TreeLearn
- Paper: https://doi.org/10.1016/j.ecoinf.2024.102795
- Training data: 6,665 German MLS beech + 877 Wytham Woods TLS →
  best publicly available training-data match for temperate broadleaf
  TLS (e.g. Tyson BP7 oak-hickory).
- Architecture: sparse-conv U-Net (spconv) + offset prediction head.
  Clustering-based instance assignment at inference time.

## Image tag

`ghcr.io/bradleylab/treelearn:latest` (also `:v1`)

## Contents

- PyTorch 2.0.0 + CUDA 11.8 (native H100 sm_90)
- spconv-cu118 (sparse conv kernels with sm_90 shipped)
- TreeLearn package installed at `/opt/TreeLearn` (editable)
- `download_weights.sh` helper at `/opt/TreeLearn/download_weights.sh`
  for fetching pretrained weights at runtime (see *Weights* below)

## Weights

Weights are NOT baked into the image. The Göttingen dataverse that
hosts them has been observed returning HTTP 500 / connection timeouts
for extended windows; coupling image build to that server makes every
rebuild fragile. Download once on the target host into a persistent
bind-mounted directory and reuse across SLURM jobs.

Three variants are attempted by `download_weights.sh`; the script
exits 0 if at least one lands:

| File | Use |
|------|-----|
| `model_weights_with_small_20241213.pth` | preferred — handles small trees |
| `model_weights_20241213.pth` | default, ≥10 m trees |
| `model_weights_diverse_training_data.pth` | older diverse-training variant |

```bash
# On Compute2, into persistent scratch:
mkdir -p /scratch2/fs1/alexander.s.bradley/treelearn_weights
srun --container-image=/storage1/.../bradleylab+treelearn+latest.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc \
       '/opt/TreeLearn/download_weights.sh /scratch2/fs1/alexander.s.bradley/treelearn_weights'
```

Then point `pipeline.yaml` `pretrain` at the resulting `.pth` under
that scratch dir.

## Running on Compute2 (Pyxis/enroot)

```bash
# On C2, import once:
cd /storage1/fs1/alexander.s.bradley/Active/c2_jobs
ENROOT_SQUASH_OPTIONS="-comp gzip" \
  XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
  ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
  ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import docker://ghcr.io#bradleylab/treelearn:latest
```

Then submit a SLURM job with:

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

srun --container-image=/storage1/.../bradleylab+treelearn+latest.sqsh \
     --container-mounts=/scratch2/...:/scratch2/...,/storage1/...:/storage1/... \
     --container-writable \
     bash -lc '
        cd /opt/TreeLearn &&
        python tools/pipeline/pipeline.py \
          --config /scratch2/.../bp7_treelearn.yaml \
          > /scratch2/.../logs/treelearn.log 2>&1
     '
```

## Pipeline invocation

Copy `configs/pipeline/pipeline.yaml` from the upstream repo and override:

- `forest_path`: input `.laz`/`.npz`/`.npy`/`.txt`
- `pretrain`: path to downloaded `.pth` (baked into container at
  `/opt/TreeLearn/data/model_weights/...`)
- `save_cfg.results_dir`: output directory
- `save_cfg.save_formats`: e.g. `['laz']`
- `save_cfg.return_type`: `'original'` to propagate labels back to the
  full input cloud (slower, use `'voxelized'` or
  `'voxelized_and_filtered'` for debug).

The pipeline centers coordinates internally, so large absolute XYZ
(e.g. EPSG:6344 UTM) is safe but the output will be re-centered —
remember to add `xyz_mean` back if a downstream tool needs georef.

## BP7 TLS first-run target

With 10 cm-voxel-downsampled BP7 (≈50 M points over ~2.3 ha), TreeLearn
paper reports inference on an 85 M-point plot at ~9 GB GPU memory; an
H100 80 GB slot has ample headroom. Expect single-digit hours per plot
with `return_type='original'`, faster with `'voxelized'`.

## Limitations vs other models in this repo

- **Not semantic seg.** TreeLearn outputs per-tree instance labels
  only. For leaf/wood/ground separation use FSCT or (future) a
  leaf-wood-only model like PointsToWood.
- **Training corpus is leaf-off European beech + Wytham.** Oak-hickory
  leaf-on performance at Tyson is an empirical question — first full
  run on BP7 will be the test.
