# point-transformer-v3 — H100 container

Point-cloud **semantic segmentation** with Point Transformer V3 (PTv3),
the marquee indoor scan-to-BIM model. Wu et al., from the Pointcept
framework.

- Upstream: https://github.com/Pointcept/Pointcept (MIT)
- Paper: PTv3 — [arXiv:2312.10035](https://arxiv.org/abs/2312.10035) (CVPR 2024 Oral)
- Architecture: serialized-neighbourhood point transformer over a
  space-filling-curve ordering; spconv backbone. No FlashAttention in
  this build (see below).

## Pinned to Pointcept v1.5.1 — read this

The only published PTv3 checkpoints (HuggingFace
[`Pointcept/PointTransformerV3`](https://huggingface.co/Pointcept/PointTransformerV3))
were trained for **Pointcept v1.5.1**. Loaded against current `main`
(v1.7.0) they silently collapse to ~0.198 mIoU
([issue #364](https://github.com/Pointcept/Pointcept/issues/364)) — the
model structure + preprocessing were redesigned in v1.5.2. There is no
separate "fixed" zoo: that HF repo *is* the zoo, and the fix is to match
the **codebase** to v1.5.1. So this image checks out Pointcept at tag
`v1.5.1` and bakes its S3DIS checkpoint.

## Image tag

`ghcr.io/bradleylab/point-transformer-v3:latest` (also `:v1`,
`:torch2.2-cu121`). GPU / amd64-only (sm_90).

## Contents

- PyTorch 2.2.2 + CUDA 12.1 (native H100 sm_90), `spconv-cu121`
- Pointcept `v1.5.1` at `/opt/Pointcept` (run via `PYTHONPATH`, not
  pip-installed)
- Pointcept's three in-tree CUDA libs compiled for sm_90: `pointops`,
  `pointops2`, `pointgroup_ops`. (torchsparse / ocnn / clip /
  torch_points_kernels are guarded upstream and intentionally omitted.)
- S3DIS checkpoint baked at `/opt/Pointcept/checkpoints/`

## Baked checkpoint

`s3dis-semseg-pt-v3m1-0-rpe.pth` (~529 MB), the v1.5.1 S3DIS Area-5
semantic model. Reported Area-5 mIoU 73.6%. Its config
(`configs/s3dis/semseg-pt-v3m1-0-rpe.py`) sets `enable_flash=False`, so
it runs without FlashAttention.

- **S3DIS 13 classes:** ceiling, floor, wall, beam, column, window,
  door, table, chair, sofa, bookcase, board, clutter.

The ScanNet checkpoint (`semseg-pt-v3m1-0-base`) was trained *with* flash
(needs it for its 77.6%) and is not baked here. Fetch it from the same HF
repo if you add a flash-enabled build.

## Inference

```bash
export PYTHONPATH=/opt/Pointcept
python tools/test.py \
  --config-file configs/s3dis/semseg-pt-v3m1-0-rpe.py \
  --num-gpus 1 \
  --options save_path=exp/s3dis/eval \
            weight=checkpoints/s3dis-semseg-pt-v3m1-0-rpe.pth
```

`tools/test.py` runs over a **preprocessed dataset**, not a raw file. The
data pipeline applies `GridSample` (voxel 0.02 m) and builds the input
dict (`coord`, `feat`=xyz+rgb, `grid_coord`, `offset`). The model returns
`dict(seg_logits=...)`; the tester softmaxes and argmaxes for per-point
labels.

### Running on your own point cloud

No turnkey raw-LAS/PLY entrypoint. To run on lab TLS scans you subclass
the Pointcept dataset (`read_single_raw_cloud`), add a
`configs/.../<your_dataset>.py`, and match the S3DIS preprocessing
(centering, 0.02 m grid, xyz+rgb features) — colorized scans are required
since the S3DIS checkpoint leans on RGB. This wrapper is the documented
follow-up, not part of v1.

## Running on Compute2 (Pyxis/enroot)

```bash
# Import once on a C2 login node:
cd /storage1/fs1/alexander.s.bradley/Active/c2_jobs
enroot import \
  -o bradleylab+point-transformer-v3+v1.sqsh \
  'docker://ghcr.io#bradleylab/point-transformer-v3:v1'
```

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# PYTHONNOUSERSITE=1 is mandatory — enroot bind-mounts $HOME, and a
# ~/.local package would otherwise shadow the container's baked stack.
srun --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+point-transformer-v3+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     --container-workdir=/opt/Pointcept \
     bash -lc '
        export PYTHONNOUSERSITE=1 PYTHONPATH=/opt/Pointcept &&
        python tools/test.py --config-file configs/s3dis/semseg-pt-v3m1-0-rpe.py \
          --num-gpus 1 --options weight=checkpoints/s3dis-semseg-pt-v3m1-0-rpe.pth \
          save_path=/scratch2/fs1/alexander.s.bradley/ptv3_eval \
          data_root=/scratch2/fs1/alexander.s.bradley/s3dis_processed
     '
```

## Validation status

**`experimental`.** Build-smoke validated (versions, `pointops` import,
reached `PointTransformerV3`, sm_90 cubins in `pointops` via cuobjdump,
S3DIS config + checkpoint present). The shipping gate is a benchmark
**mIoU on S3DIS Area 5** (target 73.6%) on Compute2. Note: v1.5.1 was
documented under torch 1.12.1/cu113; this image runs it on torch
2.2.2/cu121. The state_dict loads (version-agnostic), but exact numerical
reproduction across that torch gap is untested upstream — the Compute2
mIoU run is the definitive check.

## Notes

- **GPU-only** — pointops/spconv kernels need a CUDA device; useless on a
  CPU host.
- License **MIT** (code and weights) — commercial use OK, unlike the
  CC-BY-NC Sonata/Concerto backbones planned for the same track.
- Sister to `superpoint-transformer`: PTv3 is the dense per-point
  transformer; SPT is the superpoint-graph approach. Different speed /
  memory / accuracy trade-offs on the same task.
