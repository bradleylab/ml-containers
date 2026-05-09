# crossearth

[CrossEarth](https://github.com/VisionXLab/CrossEarth) (TPAMI 2025) is
a vision foundation model targeting Remote Sensing Domain
Generalization (RSDG): trained on a set of source domains and used
zero-shot on unseen target domains that differ in region, resolution,
spectral bands, climate, or combinations thereof. It pairs a frozen
DINOv2 ViT backbone with two domain-bridging mechanisms — a data-level
**Earth-Style Injection** pipeline and a model-level **Multi-Task
Training** scheme — and is benchmarked across a curated 32-scenario
RSDG suite spanning regions, sensors, and climates.

This container vendors the upstream `VisionXLab/CrossEarth` repo at a
pinned commit. There is no PyPI release; CrossEarth is a research
codebase that registers its models, heads, and segmentors with mmseg
via `from CrossEarth import *` side effects. Weights are NOT baked —
they download from HF Hub on first call.

GPU-primary. The DINOv2 ViT-Large backbone wants 8+ GB VRAM for
inference; segmentation training benefits from H100/A100-class.

## Image tag

`ghcr.io/bradleylab/crossearth:v1` (also `:latest`,
`:torch2.0-cu117`)

## Stack

- Base: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` (amd64-only —
  matches upstream conda recipe; CUDA 11.7)
- Python 3.10 (from base)
- PyTorch 2.0.1 + cuDNN 8 + CUDA 11.7
- `mmengine`, `mmcv >=2.0.0,<2.2`, `mmsegmentation >=1.0.0,<1.3`,
  `mmdet >=3.0.0,<3.4` — installed via `mim` to resolve correct CUDA wheels
- `xformers ==0.0.20` — exact pin per upstream README; newer xformers
  break the DINOv2 attention path
- Vendored CrossEarth at SHA `644a5a1b` (HEAD as of 2026-04-02)
- requirements.txt deps: numpy, ftfy, scipy, prettytable, matplotlib,
  regex, timm, einops

`PYTHONPATH=/opt/CrossEarth` is set in ENV so `import CrossEarth` works
from any cwd.

## Weights

Not baked. Pull from HF Hub
[`Cusyoung/CrossEarth`](https://huggingface.co/Cusyoung/CrossEarth) on
first call. The HF org name has a typo (`Cusyoung`, not the author's
GitHub handle `Cuzyoung`) — verify before pinning.

| Checkpoint | Approx. size | Use |
|---|---|---|
| `dinov2_converted.pth` | ~700 MB | 512×512 inference / training |
| `dinov2_converted_1024x1024.pth` | ~700 MB | 1024×1024 inference variant |

Place under `/opt/CrossEarth/checkpoints/` (or any path; pass to
`tools/test.py` directly). Bind-mount `/opt/hf-cache` for cross-run
weight persistence.

## Quickstart

```bash
docker run --rm --gpus all \
  -v /path/to/data:/work \
  -v /shared/hf-cache:/opt/hf-cache \
  ghcr.io/bradleylab/crossearth:v1 bash
```

Inside the container, run inference using the upstream CLI:

```bash
cd /opt/CrossEarth
python tools/test.py \
  configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py \
  ./checkpoints/dinov2_converted.pth
```

Modify dataset paths in `configs/_base_/datasets/*.py` and class counts
in the model config before pointing at your own data. The test runner
wraps `mmseg.runner.Runner` end-to-end — there is no standalone
`init_model` shim in this codebase as of HEAD `644a5a1b`.

## Why this container exists

CrossEarth fills a slot in the catalog that the other RS foundation
models (DOFA, DOFA-CLIP, Prithvi-EO, TerraMind, RemoteCLIP, GeoCLIP)
do not occupy:

- **Domain generalization first**: DOFA generalizes by learning
  spectral conditioning; CrossEarth generalizes by data augmentation
  (Earth-Style Injection) + multi-task heads. Different bet on what
  closes the domain gap.
- **DINOv2 backbone**: the only RS FM in the catalog that builds on a
  generalist self-supervised vision backbone (DINOv2) rather than an
  RS-pretrained backbone. Useful for tasks where the geometric
  features matter more than the spectral specificity.
- **Mask2Former segmentation head out of the box**: sister containers
  ship encoders; CrossEarth ships the encoder + the trained head, so
  it's ready for segmentation inference without head fine-tuning.

The mmcv 2.x + mmseg 1.x + xformers 0.0.20 + torch 2.0 stack is pinned
because it's what upstream tested. Newer torch (and the matching
mmcv 2.1+) almost certainly works but isn't validated in this image.

## License

Code: MIT (this Dockerfile, `VisionXLab/CrossEarth`).
Weights: per the [`Cusyoung/CrossEarth`](https://huggingface.co/Cusyoung/CrossEarth)
HF model card — verify before redistribution.

## References

- Gong et al. (2025) "CrossEarth: Geospatial Vision Foundation Model
  for Domain Generalizable Remote Sensing Semantic Segmentation",
  TPAMI 2025 [arXiv:2410.22629](https://arxiv.org/abs/2410.22629).
- Upstream code: [VisionXLab/CrossEarth](https://github.com/VisionXLab/CrossEarth)
  (configs, training scripts, benchmark collection).
- Project page:
  [cuzyoung.github.io/CrossEarth-Homepage/](https://cuzyoung.github.io/CrossEarth-Homepage/)
