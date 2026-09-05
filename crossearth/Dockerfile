# CrossEarth — vision foundation model for cross-domain remote sensing
# semantic segmentation (TPAMI 2025).
#
# CrossEarth uses a frozen DINOv2 backbone with an Earth-Style Injection
# pipeline + multi-task training to bridge domain gaps in spectral
# bands, region, resolution, and climate. It targets RSDG (Remote Sensing
# Domain Generalization) — i.e., training on one set of source domains
# and zero-shot inference on unseen target domains.
#
# This container vendors the upstream `VisionXLab/CrossEarth` repo at a
# pinned commit (no PyPI release; not pip-installable). Pretrained
# DINOv2-converted weights are NOT baked — they download from HF Hub
# `Cusyoung/CrossEarth` on first call.
#
# ---------------------------------------------------------------------
# WHY THIS IS NOT ON UPSTREAM'S TORCH 2.0.1 / CUDA 11.7 ANY MORE.
# ---------------------------------------------------------------------
# Upstream's conda recipe is torch 2.0.1 + cuda 11.7 + xformers 0.0.20.
# CUDA 11.7 predates sm_90, so a torch built on it cannot target an H100 —
# and Compute2's GPUs are all H100. The 2026-09-01 harness probe of this
# image did not fail: it HUNG, warning that the device's capability was not
# in torch's compiled arch list, and burned its whole time limit. That is
# the worst kind of failure for a batch job.
#
# The move is torch 2.1.0 + cu121, and the two pins that had to move with it:
#   - mmcv 2.1.0, which openmim publishes for torch 2.1.0 on cu121 (checked
#     2026-09-04 at download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/).
#   - xformers 0.0.22.post7, the build paired with torch 2.1.0. Upstream's
#     exact 0.0.20 pin is an environment snapshot, not an API constraint:
#     CrossEarth uses memory_efficient_attention, SwiGLU, fmha,
#     scaled_index_add and index_select_cat (models/backbones/dino_layers/),
#     all present and unchanged through 0.0.22.
# mmsegmentation and mmdet keep their upstream ranges.
#
# The smoke test asserts sm_90 is in torch's compiled arch list. That is a
# compile-time property of the wheel, so it holds on the GPU-less CI runner
# and is the regression test for the hang.
#
# Upstream: https://github.com/VisionXLab/CrossEarth (MIT)
# Paper:    https://arxiv.org/abs/2410.22629
# Weights:  https://huggingface.co/Cusyoung/CrossEarth
# Stack:    pytorch/pytorch 2.1.0-cuda12.1 + mmcv 2.1.0 + mmseg 1.x +
#           mmdet 3.x + xformers 0.0.22.post7

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    PYTHONPATH=/opt/CrossEarth

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pin to upstream HEAD as of 2026-04-02. No PyPI release; vendor the
# repo directly. CrossEarth/__init__.py imports models/heads/segmentors,
# which side-effect-registers them with mmseg's MODELS registry.
ARG CROSSEARTH_SHA=644a5a1b3c01b2e5531820b5291d4397597f75de
RUN git clone https://github.com/VisionXLab/CrossEarth.git /opt/CrossEarth \
    && cd /opt/CrossEarth \
    && git checkout ${CROSSEARTH_SHA} \
    && rm -rf /opt/CrossEarth/.git

# The mm* stack, resolved for torch 2.1.0 + cu121 (see the header):
#   - openmim for mmcv/mmengine (mim resolves the cu121/torch2.1.0 wheel)
#   - mmcv ==2.1.0 — the openmim wheel that exists for this torch
#   - mmsegmentation >=1.0.0,<1.3 and mmdet >=3.0.0,<3.4, upstream's ranges
#   - xformers ==0.0.22.post7 — the build paired with torch 2.1.0
# requirements.txt: numpy ftfy scipy prettytable matplotlib regex timm einops
#
# numpy<2 stays load-bearing: torch 2.1.0 + torchvision 0.16.0 were compiled
# against the numpy 1.x ABI, and CrossEarth's unpinned `numpy` resolves to
# 2.x otherwise — surfacing as `RuntimeError: Numpy is not available` the
# first time torchvision touches a tensor.
RUN pip install --upgrade pip wheel \
    && pip install -U openmim \
    && mim install mmengine \
    && mim install "mmcv==2.1.0" \
    && pip install \
        "mmsegmentation>=1.0.0,<1.3" \
        "mmdet>=3.0.0,<3.4" \
        "xformers==0.0.22.post7" \
        "numpy<2" \
    && pip install -r /opt/CrossEarth/requirements.txt \
    && pip install future tensorboard "huggingface_hub>=0.20.0" "numpy<2"

RUN mkdir -p /opt/hf-cache

# Smoke test — imports clean, registry side-effects fire, the
# CrossEarth-specific classes (Reins, LoRAReins, ReinMask2FormerHead,
# PEFTEncoderDecoder, ReinsDinoVisionTransformer) land in mmseg's
# MODELS / BACKBONES registries. Per `~/.claude/rules/ml-containers.md`,
# version printout uses importlib.metadata, not __version__, for every
# package. No weight fetch at build time (HF cache empty; first
# runtime call downloads).
RUN python -c "import torch; \
import torchvision; \
import torchvision.transforms.functional as TF; \
import numpy as np; \
TF.to_tensor(np.zeros((4, 4, 3), dtype=np.uint8)); \
import importlib.metadata as im; \
import sys; sys.path.insert(0, '/opt/CrossEarth'); \
import CrossEarth; \
from mmseg.registry import MODELS; \
expected = ['Reins', 'LoRAReins', 'ReinMask2FormerHead', 'PEFTEncoderDecoder', 'ReinsDinoVisionTransformer']; \
all_registered = set(MODELS.module_dict).union(*(set(c.module_dict) for c in MODELS.children.values())); \
found = [n for n in expected if n in all_registered]; \
assert len(found) >= 3, f'CrossEarth registry side-effects missing: found={found}, expected at least 3 of {expected}'; \
assert int(np.__version__.split('.')[0]) < 2, f'numpy must be <2 for the torch 2.1 ABI; got {np.__version__}'; \
assert 'sm_90' in torch.cuda.get_arch_list(), f'torch must target sm_90 (H100) — this image hung on Compute2 without it; arch_list={torch.cuda.get_arch_list()}'; \
from xformers.ops import memory_efficient_attention, SwiGLU, fmha; \
print('mmcv', im.version('mmcv'), \
'| mmsegmentation', im.version('mmsegmentation'), \
'| mmengine', im.version('mmengine'), \
'| mmdet', im.version('mmdet'), \
'| xformers', im.version('xformers'), \
'| torch', torch.__version__, \
'| torchvision', torchvision.__version__, \
'| numpy', np.__version__, \
'| cuda-build', torch.version.cuda); \
print('CrossEarth-specific classes registered:', found)"

LABEL org.opencontainers.image.title="crossearth" \
      org.opencontainers.image.description="CrossEarth (TPAMI 2025) — vision foundation model for cross-domain remote sensing semantic segmentation. Frozen DINOv2 backbone + Earth-Style Injection + multi-task training; targets Remote Sensing Domain Generalization (RSDG). Container vendors the upstream repo + mmcv/mmseg stack; weights via Hugging Face Hub at runtime." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/crossearth/README.md" \
      org.opencontainers.image.version="v2" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/VisionXLab/CrossEarth" \
      bradleylab.model.paper="https://arxiv.org/abs/2410.22629" \
      bradleylab.model.task="cross-domain semantic segmentation — domain-generalizable RSDG; frozen DINOv2 backbone + Mask2Former head + Earth-Style Injection" \
      bradleylab.model.sensor="image:multi (optical RS, region/resolution/spectral domain transfer)" \
      bradleylab.model.status="utility" \
      bradleylab.model.gpu="required — torch 2.1.0 cu121 with sm_90; v1 (cu117) could not target the H100 and hung" \
      bradleylab.model.weights.source="huggingface-hub:Cusyoung/CrossEarth (runtime fetch into HF_HOME=/opt/hf-cache)"

WORKDIR /work
CMD ["bash"]
