# forestformer3d

ForestFormer3D (Xiang et al., ICCV 2025 Oral, [arXiv:2506.16991](https://arxiv.org/abs/2506.16991))
— transformer-panoptic 3D forest instance segmentation built on
OneFormer3D, fine-tuned on FOR-instanceV2. Replaces PointGroup-style
clustering with learned instance queries, removing the post-hoc
clustering parameters that complicate SegmentAnyTree tuning.

> **EXPERIMENTAL.** Plan B build path. Honors the upstream pinned
> stack (mmengine 0.7.3 / mmcv 2.0.0 / mmdet 3.0.0 / mmdet3d @
> 22aaa47 / MinkowskiEngine @ 02fc608 / spconv 2.3.x) and only swaps
> CUDA toolchain so MinkowskiEngine and spconv emit native sm_90
> kernels. Plan A (full mm-stack upgrade to torch 2.2 / cu121) is the
> documented fallback if Plan B fails to compile. See "Plan A
> fallback" below.

## Image tag

- `ghcr.io/bradleylab/forestformer3d:v1` (also `:latest`)

## Stack (Plan B)

| Layer | Pin | Source / note |
|---|---|---|
| Base image | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` | cu118 nvcc supports sm_90 natively |
| Python | 3.10 | deadsnakes PPA on Ubuntu 22.04 |
| PyTorch | 1.13.1+cu117 | closest 1.13.1 wheel (no cu118 wheel of 1.13.1 was ever published; cu117 is binary-compatible with cu118 runtime) |
| mmengine | 0.7.3 | upstream pin |
| mmcv | 2.0.0 (cu117/torch1.13 wheel) | upstream pin; only cu117 wheels exist for torch 1.13 |
| mmdet | 3.0.0 | upstream pin |
| mmsegmentation | 1.0.0 | upstream pin |
| mmdet3d | git @ 22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61 | upstream pin |
| MinkowskiEngine | NVIDIA git @ 02fc608 | rebuilt with `TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"` so H100 gets native kernels |
| spconv | spconv-cu118==2.3.6, cumm-cu118==0.4.11 | swapped from upstream cu116 wheels; same 2.3.x line |
| segmentator | Karbo123 csrc @ 76efe46 | upstream pin |
| torch-scatter | 2.0.9 (source build) | upstream pin |
| torch-points-kernels | 0.7.0 (`--no-deps`) | upstream README §2 fix |
| torch-cluster | reinstalled `--no-deps` | upstream README §3 fix |

The `replace_mmdetection_files/` overlay (3,515 lines: `loops.py`
466 L, `base_model.py` 348 L, `transforms_3d.py` 2,701 L) is applied
inside the Dockerfile by copying the three files over the
site-packages locations of the installed `mmengine` and `mmdet3d` —
exactly mirroring the manual `cp` step in the upstream `readme.md`
§4. Because Plan B preserves the upstream mm* versions, no re-diffing
is required.

### Why not the literal `pytorch:1.13.1-cuda11.8-cudnn8-devel`?

That Docker Hub tag does not exist. The PyTorch project shipped only
`1.13.1-cuda11.6-cudnn8-{runtime,devel}`. To get cu118 nvcc
(required for native sm_90 compile) we base on
`nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` and pip-install the
torch 1.13.1+cu117 wheel — same approach used in
[`treelearn/Dockerfile`](../treelearn/Dockerfile).

## Pre-trained weights — runtime fetch

Weights are NOT baked into the image.

The pretrained checkpoint `epoch_3000_fix.pth` ships inside
`clean_forestformer.zip` on
[Zenodo record 16742708](https://zenodo.org/records/16742708) (~198 MB).
License: CC BY-NC 4.0 (inherited from the OneFormer3D base
model) — academic use is permitted. Coupling image build to Zenodo
risks transient CI failures during high-traffic windows, so the fetch
runs once on the target host into a bind-mounted scratch dir and is
reused across SLURM jobs.

Use the bundled `download_weights.sh`. It retries with backoff,
verifies md5 (`553d67379331966509076f3fbb409e57`), and unzips into
the destination so the layout matches what FF3D's config expects:

```text
<DEST>/clean_forestformer/epoch_3000_fix.pth
```

Run once on Compute2 (one-time, into a persistent scratch dir):

```bash
mkdir -p /scratch2/fs1/alexander.s.bradley/ff3d_weights
srun \
  -A compute2-alexander.s.bradley \
  -p general-cpu \
  --time=00:30:00 \
  --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+forestformer3d+latest.sqsh \
  --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
  bash -lc '/workspace/download_weights.sh /scratch2/fs1/alexander.s.bradley/ff3d_weights'
```

Then bind-mount that scratch directory at `/workspace/work_dirs`
inside the inference container.

## Run on Compute2 (H100)

Quick path — single-tile inference on a `.ply` file:

```bash
# Compute2 paths (operator: alexander.s.bradley)
SQSH=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+forestformer3d+latest.sqsh
WEIGHTS_DIR=/scratch2/fs1/alexander.s.bradley/ff3d_weights
WORK_DIR=/scratch2/fs1/alexander.s.bradley/ff3d_run_$(date +%Y%m%d-%H%M%S)
TILE=tile_-10_10
mkdir -p "$WORK_DIR/test_data" "$WORK_DIR/meta_data" "$WORK_DIR/out"

# Stage the .ply tile and write the test list
cp /scratch2/fs1/alexander.s.bradley/sat-test/test_tiles/${TILE}.ply "$WORK_DIR/test_data/"
echo "${TILE}" > "$WORK_DIR/meta_data/test_list.txt"

sbatch -A compute2-alexander.s.bradley \
       -p general-gpu \
       --gpus=1 --mem=96G --time=04:00:00 \
       --wrap='srun \
         --container-image='"$SQSH"' \
         --container-mounts='"$WEIGHTS_DIR"':/workspace/work_dirs,'"$WORK_DIR"'/test_data:/workspace/data/ForAINetV2/test_data,'"$WORK_DIR"'/meta_data:/workspace/data/ForAINetV2/meta_data,'"$WORK_DIR"'/out:/workspace/out \
         --container-writable \
         bash -lc "
           export PYTHONNOUSERSITE=1
           cd /workspace
           # Preprocess (run inside the container)
           cd data/ForAINetV2 && python batch_load_ForAINetV2_data.py && cd ../..
           python tools/create_data_forainetv2.py forainetv2
           # Inference
           bash /workspace/scripts/run_inference.sh
         "'
```

`PYTHONNOUSERSITE=1` is required by the ml-containers Compute2 rule
(otherwise `~/.local/lib/python3.10/site-packages/` leaks in via
enroot's `$HOME` bind-mount and shadows the container's pinned
`numpy 1.24.1` / `mmcv 2.0.0` / etc.).

For dense plots where one inference round leaves "blue points"
unsegmented, FF3D ships `tools/inference_bluepoint.sh` — see upstream
[readme.md §"Handling missed detections in dense test data"](https://github.com/SmartForest-no/ForestFormer3D#-handling-missed-detections-in-dense-test-data).

## Tyson calibration

ForestFormer3D vs SegmentAnyTree on shared deciduous test sites,
from the FF3D paper:

| Test site | Forest type | SAT F1 | FF3D F1 | Gap |
|---|---|---|---|---|
| TU_WIEN | Deciduous alluvial, leaf-off | 50.1% | 76.7% | +26.6 |
| Wytham  | Mixed deciduous, leaf-off    | 61.4% | 75.0% | +13.6 |
| BlueCat | Broadleaf temperate          | —     | 61.7% | — |

**Tyson realistic expectation: F1 ~60-70%, not 76%.**

- Tyson UAV lidar tile_-10_10: 89 pts/m², closed-canopy oak-hickory,
  leaf-on near peak senescence.
- FOR-instanceV2 training densities range ~500-9,500 pts/m². Tyson's
  89 pts/m² is below the training distribution and below the paper's
  graceful-degradation floor (~75 pts/m²).
- Closed-canopy leaf-on broadleaf is not present in FOR-instanceV2
  training (TU_WIEN is leaf-off alluvial, single-layer).

These are still expected to beat SAT (~50% on Tyson per the
SegmentAnyTree Wytham calibration in
[`tyson-forest-linkage/docs/METHOD_COMPARISON.md`](https://github.com/bradleylab/tyson-forest-linkage)
§3.3) by a wide margin.

## Plan A fallback

If Plan B's MinkowskiEngine fails to compile on sm_90 (most likely
failure mode: same PTX hang as SAT v1 had on sparse convs), drop to
Plan A — full port of the mm* stack onto torch 2.2 + cu121:

| Layer | Plan B pin | Plan A pin |
|---|---|---|
| Base | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` | `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel` |
| PyTorch | 1.13.1+cu117 | 2.2.2 (in base) |
| mmengine | 0.7.3 | 0.10.3+ |
| mmcv | 2.0.0 | 2.1.0 |
| mmdet | 3.0.0 | 3.3.0 |
| mmsegmentation | 1.0.0 | 1.2.2 |
| mmdet3d | git @ 22aaa47 | 1.4.0 |
| MinkowskiEngine | NVIDIA @ 02fc608 | [`CiSong10/MinkowskiEngine` @ `cuda12-installation`](https://github.com/CiSong10/MinkowskiEngine/tree/cuda12-installation) (proven on `segment-any-tree-h100`) |
| spconv | spconv-cu118 2.3.6 | spconv-cu121 2.3.7 |
| cumm | cumm-cu118 0.4.11 | cumm-cu121 0.5.x |

Plan A requires re-diffing the 3 `replace_mmdetection_files/` files
against mmengine 0.10.x and mmdet3d 1.4 source and porting FF3D's
customizations forward — the research-agent assessment from
[`tyson-forest-linkage/docs/METHOD_COMPARISON.md`](https://github.com/bradleylab/tyson-forest-linkage)
§4.3 is "API stable, signature drift possible, not a hard blocker"
but expect ~1 day of build debugging.

The matching `segment-any-tree-h100/Dockerfile` is the closest
template for a Plan A build — it solves the same H100 +
MinkowskiEngine + spconv problem class on torch 2.2 / cu121 with the
CiSong10 fork.

## License

ForestFormer3D source code: CC BY-NC 4.0 (inherited from the
OneFormer3D codebase by Danila Rukhovich). Pre-trained weights:
CC BY-NC 4.0. Academic use is permitted; commercial use requires
upstream permission.

This Dockerfile is bradleylab packaging only — no upstream
modifications beyond the documented `replace_mmdetection_files/`
overlay (which is itself a verbatim copy of upstream-shipped
files into upstream-installed package locations).
