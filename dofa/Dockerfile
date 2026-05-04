# dofa — Dynamic One-For-All multispectral foundation model.
#
# Xiong et al. (2024) "Neural Plasticity-Inspired Foundation Model for
# Observing the Earth Crossing Modalities" — a single ViT backbone
# adaptable to arbitrary spectral configurations via a wavelength-
# conditioning hypernetwork. Trained with masked image modelling on
# SatlasPretrain + Five-Billion-Pixels + HySpecNet-11k. Accepts
# Sentinel-1 SAR, Sentinel-2 multispectral, Gaofen, NAIP, and
# hyperspectral imagery; output is a 768-D image embedding (Base) or
# 1024-D (Large).
#
# This container ships the BASE model (`dofa_base_patch16_224`,
# DOFA_MAE weights, ~445 MB checkpoint, ~111 M parameters). The Large
# variant can be selected at runtime with `--variant large` — its
# weights download lazily on first call (1.35 GB).
#
# Upstream: https://github.com/zhu-xlab/DOFA
# Paper:    https://arxiv.org/abs/2403.15356
# Weights:  https://huggingface.co/torchgeo/dofa  (CC-BY-4.0)
# Stack:    python:3.11-slim + PyTorch 2.5 CPU + torchgeo
#
# Multi-arch: linux/amd64 + linux/arm64. torchgeo, torch CPU, and
# timm all publish aarch64 wheels. CPU inference is acceptable for
# embedding-only / one-shot use (~1-3 s/image at 224x224 for Base);
# for batched throughput, run on Compute2 H100 with the GPU base
# image variant (not yet built).

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_HOME=/opt/torch-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU wheels — matches seisbench / remoteclip / geoclip /
# neuralhydrology for ABI consistency across the catalog.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1

RUN pip install \
        "torchgeo>=0.6" \
        "timm>=1.0" \
        "huggingface_hub>=0.25" \
        "Pillow>=10.0" \
        "numpy>=1.26,<2.3"

RUN mkdir -p /opt/hf-cache /opt/torch-cache /opt/scripts
COPY scripts/dofa_embed.py /opt/scripts/dofa_embed.py

# Build-time smoke test #1: imports + DOFA loaders reachable.
# importlib.metadata.version per ml-containers rules.
RUN python -c "import torch, torchgeo, timm; \
import importlib.metadata as im; \
print('torchgeo', im.version('torchgeo'), '| torch', torch.__version__, '| timm', im.version('timm')); \
from torchgeo.models import (dofa_base_patch16_224, DOFABase16_Weights, \
                             dofa_large_patch16_224, DOFALarge16_Weights); \
assert callable(dofa_base_patch16_224); \
assert callable(dofa_large_patch16_224)"

# Build-time smoke test #2 + weight bake (Base only): instantiate the
# Base model with pretrained weights, run a synthetic 12-band S2-shaped
# tensor through forward_features, and assert the output shape.
# `torch.set_grad_enabled(False)` is used instead of `with torch.no_grad():`
# because compound statements can't follow `;` in inline Python.
RUN python -c "import torch; torch.set_grad_enabled(False); \
from torchgeo.models import dofa_base_patch16_224, DOFABase16_Weights; \
m = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE); m.eval(); \
x = torch.randn(1, 12, 224, 224); \
waves = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]; \
emb = m.forward_features(x, wavelengths=waves); \
print('S2-12band -> embedding shape:', tuple(emb.shape)); \
assert tuple(emb.shape) == (1, 768), f'expected (1, 768), got {tuple(emb.shape)}'" \
    && du -sh /opt/torch-cache 2>/dev/null \
    && echo 'baked DOFA-Base weights at /opt/torch-cache'

LABEL org.opencontainers.image.title="dofa" \
      org.opencontainers.image.description="DOFA — Dynamic One-For-All multispectral / SAR / optical foundation model with wavelength-conditioning hypernetwork (Xiong et al. 2024). CPU runtime; Base weights baked at build time." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/dofa/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="CC-BY-4.0" \
      bradleylab.model.upstream="https://github.com/zhu-xlab/DOFA" \
      bradleylab.model.paper="https://arxiv.org/abs/2403.15356" \
      bradleylab.model.task="multispectral / SAR / optical embedding extraction (foundation model)" \
      bradleylab.model.sensor="image:multispectral,sar,rgb,hyperspectral" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="HuggingFace Hub torchgeo/dofa via torchgeo loader; Base weights baked at build time, Large fetched lazily on --variant large"

WORKDIR /work
CMD ["bash"]
