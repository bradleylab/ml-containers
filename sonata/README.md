# sonata — H100 container

Self-supervised **Point Transformer V3 encoder** (Meta / facebookresearch).
The model `concerto` was derived from this one. Released capabilities:

- **Per-point embeddings** — the encoder's learned features.
- **Closed-set ScanNet-20 semantic segmentation** via the shipped
  linear-probe head.

- Upstream code: https://github.com/facebookresearch/sonata (Apache-2.0)
- Upstream weights: https://huggingface.co/facebook/sonata (CC-BY-NC-4.0)
- Paper: [arXiv:2503.16429](https://arxiv.org/abs/2503.16429) (CVPR 2025)

## Image tag

`ghcr.io/bradleylab/sonata:latest` (also `:v1`, `:torch2.2-cu121`).
GPU / amd64-only (sm_90).

## Contents

- PyTorch 2.2.2 + CUDA 12.1 (native H100 sm_90), `spconv-cu121`,
  `torch_scatter`. **No FlashAttention** (optional upstream; encoder runs
  `enable_flash=False`). No ocnn/pointops (standalone package).
- Sonata package at `/opt/Sonata` (run via `PYTHONPATH`).
- Weights baked at `/opt/Sonata/checkpoints/`.

## Baked weights

| File | Size | Use |
|------|------|-----|
| `sonata.pth` | ~434 MB | the encoder (`{"config","state_dict"}`) |
| `sonata_linear_prob_head_sc.pth` | 0.10 MB | ScanNet-20 linear-probe semseg head |

`sonata_small.pth` and the 1.95 GB pretrain checkpoint are on the same HF
repo if needed.

## Inference

Loaded from the baked checkpoint (architecture config is in the `.pth`),
flash off:

```python
import torch
from sonata.model import PointTransformerV3

ck = torch.load("/opt/Sonata/checkpoints/sonata.pth", map_location="cpu")
ck["config"]["enable_flash"] = False
model = PointTransformerV3(**ck["config"]).cuda().eval()
model.load_state_dict(ck["state_dict"])

import sonata
point = sonata.transform.default()(point)   # dict: coord/color/normal (N,3) numpy
for k in point:
    if isinstance(point[k], torch.Tensor):
        point[k] = point[k].cuda(non_blocking=True)
point = model(point)                          # encoder forward
```

The encoder is hierarchical; for per-point features at input resolution,
walk back up the pooling chain and undo GridSample (upstream README /
`demo/`). **ScanNet-20 segmentation** adds the probe head
(`sonata_linear_prob_head_sc.pth`, a single `nn.Linear`) on the encoder
features.

Equivalently, the upstream loader `sonata.model.load("sonata",
repo_id="facebook/sonata")` does the same thing over the network; loading
the baked local `.pth` keeps the container offline.

### Input format

A dict of numpy arrays (`coord`/`color`/`normal`), not raw LAS/PLY;
`sonata.transform.default()` does GridSample (0.02 m), center-shift,
color-normalize. Bring your own LAS/PLY→dict adapter.

## Running on Compute2 (Pyxis/enroot)

```bash
cd /storage1/fs1/alexander.s.bradley/Active/c2_jobs
enroot import -o bradleylab+sonata+v1.sqsh \
  'docker://ghcr.io#bradleylab/sonata:v1'
```

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# PYTHONNOUSERSITE=1 mandatory — enroot bind-mounts $HOME.
srun --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+sonata+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     --container-workdir=/opt/Sonata \
     bash -lc 'export PYTHONNOUSERSITE=1 PYTHONPATH=/opt/Sonata;
               python /scratch2/fs1/alexander.s.bradley/scripts/sonata_embed.py'
```

## Validation status

**`experimental`.** Build-smoke validated: versions, package import, the
model **builds from the baked checkpoint with a strict state-dict load**
(0 missing / 0 unexpected) on CPU, and the ScanNet probe head loads. The
runtime gate is a Compute2 check — embedding extraction + ScanNet mIoU via
the probe head.

## License

- **Code:** Apache-2.0.
- **Weights:** **CC-BY-NC-4.0** — non-commercial only (NC-restricted by
  training datasets). Gates paid-course shipping, not internal eval.

## Notes

- **GPU-only** at inference (spconv/torch_scatter kernels).
- Base is torch 2.2.2/cu121 for parity; upstream's tested combo is
  2.5.0/cu124 (documented fallback in the Dockerfile header).
- Sister to `concerto` (which extends Sonata with joint 2D-3D
  pre-training) and `point-transformer-v3` (the supervised PTv3
  segmenter). Same PTv3 architecture; Sonata/Concerto are the
  self-supervised encoders.
