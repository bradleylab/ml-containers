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
# pinned commit (no PyPI release; not pip-installable). The mmcv +
# mmsegmentation + mmdet stack mirrors upstream's conda recipe — torch
# 2.0.1 + cuda 11.7 + mmcv 2.x + mmseg 1.x + xformers 0.0.20.
# Pretrained DINOv2-converted weights are NOT baked — they download
# from HF Hub `Cusyoung/CrossEarth` on first call.
#
# Upstream: https://github.com/VisionXLab/CrossEarth (MIT)
# Paper:    https://arxiv.org/abs/2410.22629
# Weights:  https://huggingface.co/Cusyoung/CrossEarth
# Stack:    pytorch/pytorch 2.0.1-cuda11.7 + mmcv 2.x + mmseg 1.x +
#           mmdet 3.x + xformers 0.0.20

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
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

# Stack matches upstream conda recipe exactly:
#   - openmim for mmcv/mmengine install (mim resolves the right cuda wheel)
#   - mmcv >=2.0.0,<2.2 (mmcv 2.0.x supports torch 2.0; >=2.1 needs torch 2.1+)
#   - mmsegmentation >=1.0.0,<1.3
#   - mmdet >=3.0.0,<3.4
#   - xformers ==0.0.20 (exact pin per upstream — newer breaks DINOv2 path)
# requirements.txt: numpy ftfy scipy prettytable matplotlib regex timm einops
RUN pip install --upgrade pip wheel \
    && pip install -U openmim \
    && mim install mmengine \
    && mim install "mmcv>=2.0.0,<2.2" \
    && pip install \
        "mmsegmentation>=1.0.0,<1.3" \
        "mmdet>=3.0.0,<3.4" \
        "xformers==0.0.20" \
    && pip install -r /opt/CrossEarth/requirements.txt \
    && pip install future tensorboard "huggingface_hub>=0.20.0"

RUN mkdir -p /opt/hf-cache

# Smoke test — imports clean, registry side-effects fire, the
# CrossEarth-specific classes (Reins, LoRAReins, ReinMask2FormerHead,
# PEFTEncoderDecoder, ReinsDinoVisionTransformer) land in mmseg's
# MODELS / BACKBONES registries. Per `~/.claude/rules/ml-containers.md`,
# version printout uses importlib.metadata, not __version__, for every
# package. No weight fetch at build time (HF cache empty; first
# runtime call downloads).
RUN python -c "import torch; \
import importlib.metadata as im; \
import sys; sys.path.insert(0, '/opt/CrossEarth'); \
import CrossEarth; \
from mmseg.registry import MODELS; \
expected = ['Reins', 'LoRAReins', 'ReinMask2FormerHead', 'PEFTEncoderDecoder', 'ReinsDinoVisionTransformer']; \
all_registered = set(MODELS.module_dict); \
for child in MODELS.children.values(): all_registered.update(child.module_dict); \
found = [n for n in expected if n in all_registered]; \
assert len(found) >= 3, f'CrossEarth registry side-effects missing: found={found}, expected at least 3 of {expected}'; \
print('mmcv', im.version('mmcv'), \
'| mmsegmentation', im.version('mmsegmentation'), \
'| mmengine', im.version('mmengine'), \
'| mmdet', im.version('mmdet'), \
'| xformers', im.version('xformers'), \
'| torch', torch.__version__, \
'| cuda-build', torch.version.cuda); \
print('CrossEarth-specific classes registered:', found)"

LABEL org.opencontainers.image.title="crossearth" \
      org.opencontainers.image.description="CrossEarth (TPAMI 2025) — vision foundation model for cross-domain remote sensing semantic segmentation. Frozen DINOv2 backbone + Earth-Style Injection + multi-task training; targets Remote Sensing Domain Generalization (RSDG). Container vendors the upstream repo + mmcv/mmseg stack; weights via Hugging Face Hub at runtime." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/crossearth/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/VisionXLab/CrossEarth" \
      bradleylab.model.paper="https://arxiv.org/abs/2410.22629" \
      bradleylab.model.task="cross-domain semantic segmentation — domain-generalizable RSDG; frozen DINOv2 backbone + Mask2Former head + Earth-Style Injection" \
      bradleylab.model.sensor="image:multi (optical RS, region/resolution/spectral domain transfer)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:Cusyoung/CrossEarth (runtime fetch into HF_HOME=/opt/hf-cache)"

WORKDIR /work
CMD ["bash"]
