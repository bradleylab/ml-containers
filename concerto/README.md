# concerto — H100 container

Joint 2D-3D self-supervised **Point Transformer V3 encoder** (Pointcept
group, Sonata-derived). This image runs the two *released* capabilities:

- **Per-point embeddings** — the encoder's learned features.
- **Closed-set semantic segmentation** on ScanNet-20 via the shipped
  linear-probe head.

- Upstream code: https://github.com/Pointcept/Concerto (Apache-2.0)
- Upstream weights: https://huggingface.co/Pointcept/Concerto (CC-BY-NC-4.0)
- Paper: [arXiv:2510.23607](https://arxiv.org/abs/2510.23607)

## Not included: the "zero-shot / open-world" path

The paper describes an open-world capability via a translator into CLIP's
language space. **That code and those weights are not in the public
release** (no `clip` / `open_clip` / text-translator anywhere in the
repo or on the HF model card). It is therefore not reproducible from
public artifacts and is **not** part of this image. What ships is
unsupervised feature embeddings + a closed-set ScanNet-20 head — not
open-vocabulary segmentation.

## Image tag

`ghcr.io/bradleylab/concerto:latest` (also `:v1`, `:torch2.2-cu121`).
GPU / amd64-only (sm_90).

## Contents

- PyTorch 2.2.2 + CUDA 12.1 (native H100 sm_90), `spconv-cu121`,
  `torch_scatter`. **No FlashAttention** (optional upstream; the encoder
  runs `enable_flash=False`). No ocnn/pointops (Concerto is standalone,
  not the Pointcept framework).
- Concerto package at `/opt/Concerto` (run via `PYTHONPATH`).
- Weights baked at `/opt/Concerto/checkpoints/`.

## Baked weights

| File | Size | Use |
|------|------|-----|
| `concerto_large.pth` | ~834 MB | the encoder (`{"config","state_dict"}`) |
| `concerto_large_linear_prob_head_sc.pth` | 0.14 MB | ScanNet-20 linear-probe semseg head |

Other variants (`tiny`/`small`/`base`, the 8.3 GB video-pretrain) live on
the same HF repo; fetch them if needed.

## Inference

Both paths run without FlashAttention (`enable_flash=False`). The encoder
is loaded from the baked checkpoint (its architecture config is in the
`.pth`):

```python
import torch
from concerto.model import PointTransformerV3

ck = torch.load("/opt/Concerto/checkpoints/concerto_large.pth", map_location="cpu")
ck["config"]["enable_flash"] = False           # flash-attn not installed
model = PointTransformerV3(**ck["config"]).cuda().eval()
model.load_state_dict(ck["state_dict"])

# point: dict of numpy arrays — coord (N,3), color (N,3), normal (N,3)
import concerto
point = concerto.transform.default()(point)    # GridSample @ 0.02 m, etc.
for k in point:
    if isinstance(point[k], torch.Tensor):
        point[k] = point[k].cuda(non_blocking=True)
point = model(point)                            # encoder forward
```

The encoder is **hierarchical**; for per-point features at input
resolution you walk back up the pooling chain and undo GridSample (see
the upstream README "feature extraction" snippet / `demo/0_pca.py`).

**ScanNet-20 segmentation** adds the probe head on top of the encoder
features — see upstream `demo/2_sem_seg.py` (loads
`concerto_large_linear_prob_head_sc.pth`, a single `nn.Linear`, classes =
ScanNet `CLASS_LABELS_20`).

### Input format

Not raw LAS/PLY — the model takes a dict (`coord`/`color`/`normal` numpy
arrays); `concerto.transform.default()` does GridSample (0.02 m),
center-shift, color-normalize. Bring your own LAS/PLY→dict adapter
(laspy/open3d → numpy). Colorless/normalless input is tolerated (set to
zeros) per the model card.

## Running on Compute2 (Pyxis/enroot)

```bash
cd /storage1/fs1/alexander.s.bradley/Active/c2_jobs
enroot import -o bradleylab+concerto+v1.sqsh \
  'docker://ghcr.io#bradleylab/concerto:v1'
```

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# PYTHONNOUSERSITE=1 mandatory — enroot bind-mounts $HOME.
srun --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+concerto+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     --container-workdir=/opt/Concerto \
     bash -lc 'export PYTHONNOUSERSITE=1 PYTHONPATH=/opt/Concerto;
               python /scratch2/fs1/alexander.s.bradley/scripts/concerto_embed.py'
```

## Validation status

**`experimental`.** Build-smoke validated: versions, package import, and
the model **builds from the baked checkpoint with a strict state-dict
load** (0 missing / 0 unexpected) on CPU, plus the ScanNet probe head
loads. The runtime gate is a Compute2 check — embedding extraction on a
real cloud + ScanNet mIoU via the probe head.

## License

- **Code:** Apache-2.0 (Concerto is built on Meta's Sonata).
- **Weights:** **CC-BY-NC-4.0** — non-commercial only (NC-restricted by
  training datasets like HM3D/ArkitScenes). This gates paid-course
  shipping; internal eval is fine.

## Notes

- **GPU-only** at inference (spconv/torch_scatter kernels).
- Base is torch 2.2.2/cu121 for parity with the sibling point-cloud
  images; upstream's tested combo is torch 2.5.0/cu124 (documented
  fallback in the Dockerfile header).
- Sister to `point-transformer-v3`: same PTv3 architecture, but Concerto
  is a self-supervised *encoder* (features), where PTv3 here is a
  supervised semantic segmenter (labels).
