# forestformer3d

ForestFormer3D (Xiang et al., ICCV 2025 Oral, [arXiv:2506.16991](https://arxiv.org/abs/2506.16991))
— transformer-panoptic 3D forest instance segmentation built on
OneFormer3D, fine-tuned on FOR-instanceV2. Replaces PointGroup-style
clustering with learned instance queries, removing the post-hoc
clustering parameters that complicate SegmentAnyTree tuning.

> **EXPERIMENTAL.** Plan A build path (cu121 + torch 2.2.2 + CiSong10
> MinkowskiEngine fork). Forward-ports FF3D's mm-stack from the
> upstream-pinned `mmengine 0.7.3 / mmcv 2.0.0 / mmdet 3.0.0 /
> mmdet3d @ 22aaa47` to `mmengine 0.10.3 / mmcv 2.1.0 / mmdet 3.3.0 /
> mmdet3d 1.4.0` so MinkowskiEngine and spconv emit native sm_90
> kernels for H100. The 3 `replace_mmdetection_files/` overlay files
> are still applied verbatim from upstream — see "replace_mmdetection_files"
> below for the signature-drift caveat.

## Image tag

- `ghcr.io/bradleylab/forestformer3d:v1` (also `:latest`,
  `:torch2.2-cu121-planA`)

## Stack (Plan A, live)

| Layer | Pin | Source / note |
|---|---|---|
| Base image | `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel` | matches `segment-any-tree-h100`; cu121 nvcc supports sm_90 natively |
| Python | 3.10 | from base image |
| PyTorch | 2.2.2 | in base |
| mmengine | 0.10.3 | in mmdet3d-1.4.0 accepted range `[0.8.0, 1.0.0)` |
| mmcv | 2.1.0 (cu121/torch2.1 wheel) | mmdet3d 1.4.0 strictly requires `<2.2.0`; the cu121/torch2.2 prebuilt index only ships 2.2.0, so we pull mmcv 2.1.0 from the cu121/torch2.1 index — same C++ ABI as torch 2.2.x |
| mmdet | 3.3.0 | in mmdet3d-1.4.0 accepted range `[3.0.0rc5, 3.4.0)` |
| mmsegmentation | 1.2.2 | latest 1.2.x |
| mmdet3d | 1.4.0 | released wheel; FF3D's in-tree customizations were against `@ 22aaa47` which is roughly 1.3.0-era — see overlay caveat below |
| MinkowskiEngine | [`CiSong10/MinkowskiEngine` @ `cuda12-installation`](https://github.com/CiSong10/MinkowskiEngine/tree/cuda12-installation) | proven on `segment-any-tree-h100`; rebuilt with `TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"` so H100 gets native kernels |
| spconv | spconv-cu121 == 2.3.8 | latest 2.3.x cu121 wheel; ships sm_90 kernels |
| cumm | cumm-cu121 == 0.7.11 | matched to spconv-cu121 2.3.8 deps (`>=0.7.11,<0.8.0`) |
| segmentator | Karbo123 csrc @ 76efe46 | upstream pin |
| torch-scatter / torch-cluster | torch-2.2.2+cu121 prebuilt | matches SAT v2 |
| torch-points-kernels | git source w/ SAT v2 patches | upstream README §2 fix; numpy/cython pins + `at::` namespace + arch ≥ sm_50 |

## replace_mmdetection_files overlay

FF3D ships 3 full-file replacements (3,515 lines total) authored against
`mmengine 0.7.3` + `mmdet3d @ 22aaa47`:

- `mmengine/runner/loops.py` (466 L)
- `mmengine/model/base_model/base_model.py` (348 L)
- `mmdet3d/datasets/transforms/transforms_3d.py` (2,701 L)

The Dockerfile applies them as a verbatim `cp` over the installed
`mmengine 0.10.x` and `mmdet3d 1.4.0` site-packages locations (paths
derived at build time via `python -c "import mmengine, os; ..."`).

The research-agent assessment from
[`tyson-forest-linkage/docs/METHOD_COMPARISON.md`](https://github.com/bradleylab/tyson-forest-linkage)
§4.3 is "API stable, signature drift possible, not a hard blocker".

If the smoke test (or first inference) fails on an `AttributeError` /
`TypeError` originating in one of those three files, the fix is to
forward-port FF3D's surgical customizations against the new
mmengine/mmdet3d source rather than blindly overwriting. The deltas
between FF3D's frozen versions and upstream `loops.py` /
`base_model.py` / `transforms_3d.py` are typically 50-200 lines of
real change against thousands of upstream lines.

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
mm-stack).

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

## Plan B (historical context — CI-failed 2026-05-03)

The original PR-33 build attempted to mirror upstream FF3D's exact
pinned stack and only swap CUDA toolchain:

| Layer | Plan B pin (CI-failed) |
|---|---|
| Base | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` |
| PyTorch | 1.13.1+cu117 |
| mmengine | 0.7.3 |
| mmcv | 2.0.0 (cu117/torch1.13 wheel) |
| mmdet | 3.0.0 |
| mmsegmentation | 1.0.0 |
| mmdet3d | git @ 22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61 |
| MinkowskiEngine | NVIDIA @ 02fc608 |
| spconv | spconv-cu118 2.3.6, cumm-cu118 0.4.11 |

Plan B failed in CI run
[25285693937](https://github.com/bradleylab/ml-containers/actions/runs/25285693937)
at MinkowskiEngine compile:

```
torch/utils/cpp_extension.py:1793, in _get_cuda_arch_flags
    raise ValueError(f"Unknown CUDA arch ({arch}) or GPU not supported")
ValueError: Unknown CUDA arch (8.9) or GPU not supported
```

PyTorch 1.13.1 was released 2022-12 — two years before Hopper /
Ada (sm_89, sm_90) entered its supported arch list. The arch
validation in `cpp_extension.py` rejects `8.9` (and `9.0`) outright
before nvcc is even invoked, so even with cu118 nvcc on the host the
compile path can never reach H100. Plan B was unsalvageable; Plan A
is the live build path.

## License

ForestFormer3D source code: CC BY-NC 4.0 (inherited from the
OneFormer3D codebase by Danila Rukhovich). Pre-trained weights:
CC BY-NC 4.0. Academic use is permitted; commercial use requires
upstream permission.

This Dockerfile is bradleylab packaging only — no upstream
modifications beyond the documented `replace_mmdetection_files/`
overlay (which is itself a verbatim copy of upstream-shipped
files into upstream-installed package locations).
