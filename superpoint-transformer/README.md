# superpoint-transformer — H100 container

Large-scale point-cloud segmentation from one codebase, two models:

- **SPT** — **semantic** segmentation (per-point class labels).
- **SuperCluster** — **panoptic** segmentation (semantics + instances),
  via superpoint-graph clustering.

Damien Robert et al. (`drprojects/superpoint_transformer`).

- Upstream: https://github.com/drprojects/superpoint_transformer (MIT)
- Papers: SPT — [arXiv:2306.08045](https://arxiv.org/abs/2306.08045)
  (ICCV 2023); SuperCluster — [arXiv:2401.06704](https://arxiv.org/abs/2401.06704)
  (3DV 2024)
- Architecture: hierarchical **superpoint partition** (cut-pursuit) +
  transformer over the superpoint graph. No sparse-conv backbone — no
  spconv, no MinkowskiEngine, no FlashAttention. The one compiled CUDA
  dependency is FRNN (fixed-radius nearest neighbours).

## Image tag

`ghcr.io/bradleylab/superpoint-transformer:latest` (also `:v1`,
`:torch2.2-cu121`). GPU / amd64-only (sm_90), like the other CUDA images
in this repo.

## Contents

- PyTorch 2.2.2 + CUDA 12.1 (native H100 sm_90)
- FRNN compiled for sm_90; PyG (`pyg_lib`, `torch_scatter`,
  `torch_cluster`, `torch_geometric==2.3.0`); pgeof / pycut-pursuit /
  pygrid-graph (prebuilt wheels)
- SPT source at `/opt/superpoint_transformer` (run via `PYTHONPATH`, no
  pip install — it is a Hydra project)
- Five checkpoints baked at `/opt/superpoint_transformer/checkpoints/`
  (see below)

## Baked checkpoints

Indoor (S3DIS Area 5 = fold 5) and outdoor (DALES), for both tasks. All
from Zenodo ([SPT 8042712](https://doi.org/10.5281/zenodo.8042712),
[SuperCluster 10689037](https://doi.org/10.5281/zenodo.10689037)).

| File | Model / task | Dataset | Classes |
|------|--------------|---------|---------|
| `spt-2_dales.ckpt` | SPT, semantic | DALES (ALS, urban/aerial) | ground, vegetation, cars, trucks, power lines, fences, poles, buildings |
| `spt-2_s3dis_fold5.ckpt` | SPT, semantic | S3DIS Area 5 (indoor) | ceiling, floor, wall, beam, column, window, door, table, chair, sofa, bookcase, board, clutter |
| `supercluster_dales.ckpt` | SuperCluster, panoptic | DALES | as above |
| `supercluster_s3dis_fold5.ckpt` | SuperCluster, panoptic | S3DIS Area 5 | as above |
| `supercluster_s3dis_with_stuff_fold5.ckpt` | SuperCluster, panoptic | S3DIS Area 5 | wall/floor/ceiling treated as **stuff** — the BIM-appropriate split |

Other folds / KITTI-360 / ScanNet checkpoints exist on the same Zenodo
records; fetch them at runtime if needed.

## Inference

Inference runs through the repo's Hydra entrypoint:

```bash
# Semantic (SPT)
python src/eval.py \
  experiment=semantic/dales \
  ckpt_path=checkpoints/spt-2_dales.ckpt

# Panoptic (SuperCluster)
python src/eval.py \
  experiment=panoptic/s3dis \
  ckpt_path=checkpoints/supercluster_s3dis_fold5.ckpt
```

`eval.py` runs `Trainer.test()` over a **preprocessed dataset**, not a
raw file path. On first dataset instantiation the pre-transforms compute
the superpoint partition and cache tiles to
`data/<dataset>/processed/...`. SuperCluster additionally needs its
graph-clustering hyperparameters set (grid-searched post-training — see
the upstream `notebooks/demo_panoptic_parametrization.ipynb`).

### Running on your own point cloud

There is **no turnkey "raw LAS/PLY in → labeled cloud out"** path. SPT
expects data organised into a dataset structure, with partition
parameters (voxel size, `pcp_regularization`, graph params) tuned per
dataset. To run on lab TLS scans you subclass `src.datasets.BaseDataset`
(implement `read_single_raw_cloud()` for your reader), add a
`configs/datamodule/<task>/<your_dataset>.yaml`, and tune the partition —
the upstream tutorial notebook + slides walk through this. For colorized
ALS, reusing the DALES dataset/config (which already reads `.ply` and has
tuned aerial partition params) is the shortest path. **This wrapper is
the documented follow-up to this image, not part of v1.**

## Running on Compute2 (Pyxis/enroot)

```bash
# Import once on a C2 login node:
cd /storage1/fs1/alexander.s.bradley/Active/c2_jobs
enroot import \
  -o bradleylab+superpoint-transformer+v1.sqsh \
  'docker://ghcr.io#bradleylab/superpoint-transformer:v1'
```

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# PYTHONNOUSERSITE=1 is mandatory — enroot bind-mounts $HOME, and any
# ~/.local package would otherwise shadow the container's baked stack.
srun --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+superpoint-transformer+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     --container-workdir=/opt/superpoint_transformer \
     bash -lc '
        export PYTHONNOUSERSITE=1 &&
        python src/eval.py experiment=semantic/dales \
          ckpt_path=checkpoints/spt-2_dales.ckpt \
          datamodule.data_dir=/scratch2/fs1/alexander.s.bradley/spt_data
     '
```

## Validation status

**`experimental`.** The image is build-smoke validated (imports,
FRNN `.so` load, model class reached, checkpoints present). Build-smoke
does **not** exercise an sm_90 kernel — FRNN arch correctness only
surfaces on a real H100 forward pass. The shipping gate is a benchmark
run on a preprocessed sample (semantic **mIoU** on DALES / S3DIS Area 5;
panoptic **PQ** on S3DIS for SuperCluster), run on Compute2. Until that
passes, treat outputs as unverified.

## Notes

- **FRNN is GPU-only** — no CPU fallback. This image does nothing useful
  without a CUDA device, even for SPT (the cleanest stack in the AEC
  set).
- Checkpoints are tiny; the rest of the image is the CUDA/PyG stack.
- License is **MIT** (code and weights) — commercial use OK, unlike the
  CC-BY-NC Sonata/Concerto backbones planned for the same track.
