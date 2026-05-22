# octformer — H100 container

Octree-transformer point-cloud **semantic segmentation** (ScanNet /
ScanNet200). Wang, SIGGRAPH 2023. An alternate-architecture comparison
to `point-transformer-v3` / `superpoint-transformer`.

- Upstream code: https://github.com/octree-nn/octformer (MIT)
- Paper: [arXiv:2305.03045](https://arxiv.org/abs/2305.03045) (SIGGRAPH 2023)
- Stack: `ocnn==2.2.6` (octree CNN, pure PyTorch) + `dwconv` (octree
  depthwise-conv CUDA extension). OctFormer itself is pure Python.

## Image tag

`ghcr.io/bradleylab/octformer:latest` (also `:v1`, `:torch2.2-cu121`).
GPU / amd64-only (sm_90).

## Weights are NOT baked — mount at runtime

The ScanNet / ScanNet200 checkpoints live on OneDrive, which **refuses
scripted download** (403 on every non-interactive path), and they are
**non-commercial** — bound by the [ScanNet Terms of
Use](http://www.scan-net.org/) (the weights carry no explicit license
upstream, but the ScanNet data terms restrict to research/education and
forbid redistribution). So they are neither baked into the image nor
re-hostable. Download them **interactively** from the upstream README's
OneDrive links, stage to NAS / Compute2 scratch, and mount at runtime.

Upstream links (interactive browser only):
- ScanNet (21-class): the `scannet` OneDrive link in the OctFormer README
- ScanNet200: the `scannet200` OneDrive link there

Each bundle contains the training log + `best_model.pth`.

## Contents

- PyTorch 2.2.2 + CUDA 12.1 (native H100 sm_90)
- `ocnn==2.2.6` (pinned — pure PyTorch, pre-Triton; do not float to 2.3.x,
  which adds Triton kernels incompatible with torch 2.2.2's Triton)
- `dwconv` compiled for sm_90 (the only CUDA extension)
- OctFormer source at `/opt/octformer` (run via `PYTHONPATH`)

## Inference

There is **no single-cloud CLI** — inference is dataset-batch evaluation.
For ScanNet semantic segmentation (with a mounted checkpoint):

```bash
python scripts/run_seg_scannet.py \
  --gpu 0 --alias scannet --run validate \
  --ckpt /data/octformer/scannet/best_model.pth
```

Input is ScanNet preprocessed to `.npz` (via
`tools/seg_scannet.py --run process_scannet`); the **octree is built at
runtime inside the model forward** (`depth=11, full_depth=2`). Running on
your own cloud means adapting it into the dataset's `.npz` schema. ScanNet
uses 21 classes (20 valid + ignore@0); ScanNet200 uses 200.

## Running on Compute2 (Pyxis/enroot)

```bash
cd /storage1/fs1/alexander.s.bradley/Active/c2_jobs
enroot import -o bradleylab+octformer+v1.sqsh \
  'docker://ghcr.io#bradleylab/octformer:v1'
```

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Mount the interactively-downloaded weights + preprocessed ScanNet data.
# PYTHONNOUSERSITE=1 mandatory — enroot bind-mounts $HOME.
srun --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+octformer+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     --container-workdir=/opt/octformer \
     bash -lc 'export PYTHONNOUSERSITE=1 PYTHONPATH=/opt/octformer;
               python scripts/run_seg_scannet.py --gpu 0 --alias scannet --run validate \
                 --ckpt /scratch2/fs1/alexander.s.bradley/octformer/scannet/best_model.pth'
```

## Validation status

**`experimental`.** Build-smoke validated: versions, `ocnn` import,
**`dwconv` import** (exercises the compiled `.so` — OctFormer lazy-imports
it, so the smoke does it explicitly), reached the `OctFormer` model class,
and `dwconv` carries sm_90 cubins (cuobjdump). Runtime gate (ScanNet mIoU)
needs the mounted weights + preprocessed data on Compute2.

## License

- **Code:** MIT.
- **Weights:** unstated upstream, but **ScanNet-trained → non-commercial
  research only** (ScanNet Terms of Use). Do not re-host publicly; this is
  an open ship-gate for any course distribution.

## Notes

- **GPU-only** (dwconv/ocnn).
- `ocnn==2.2.6` pin is load-bearing — see Contents.
- Same base/toolchain as the sibling point-cloud images for parity.
