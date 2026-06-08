# croma — Contrastive Radar-Optical Masked Autoencoders.
#
# Fuller, Millard & Green (2023) "CROMA: Remote Sensing Representations
# with Contrastive Radar-Optical Masked Autoencoders" (NeurIPS 2023).
# A foundation model pretrained jointly on Sentinel-1 SAR (2-channel
# VV/VH) and Sentinel-2 optical (12-band, cirrus removed) with combined
# contrastive + masked-autoencoding objectives. It exposes a SAR
# encoder, an optical encoder, and a cross-modal joint encoder; each
# emits patch tokens plus a global-average-pooled (GAP) embedding.
#
# This container ships the BASE model (ViT-B, 768-D embeddings),
# weights baked at build time. The Large variant (ViT-L, 1024-D) is
# selected at runtime with `--variant large`; its weights download
# lazily from the HF Hub on first call.
#
# Upstream: https://github.com/antofuller/CROMA  (MIT)
# Paper:    https://arxiv.org/abs/2311.00566
# Weights:  https://huggingface.co/antofuller/CROMA  (MIT)
# Stack:    python:3.11-slim + PyTorch 2.5.1 CPU + einops
#
# Multi-arch: linux/amd64 + linux/arm64. CROMA imports only torch +
# einops (no compiled extensions, no torchvision), all of which publish
# aarch64 wheels. CPU inference is acceptable for embedding-only / one-
# shot use; for batched embedding across many chips, run on Compute2
# H100 with a GPU base image (GPU variant not yet built).

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/opt/croma

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU wheels — matches dofa / seisbench / remoteclip / geoclip /
# neuralhydrology for ABI consistency across the catalog. CROMA needs
# only torch + einops; no torchvision.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

RUN pip install \
        "einops>=0.7" \
        "huggingface_hub>=0.25" \
        "numpy>=1.26,<2.3"

# Upstream model code is a single module (not a pip package); pin to a
# commit SHA for reproducibility and import via PYTHONPATH=/opt/croma.
ARG CROMA_SHA=59505a6bcadbf36ba20767270154bf9f3067c5e7
RUN mkdir -p /opt/croma /opt/scripts \
    && curl -fsSL \
        "https://raw.githubusercontent.com/antofuller/CROMA/${CROMA_SHA}/use_croma.py" \
        -o /opt/croma/use_croma.py

COPY scripts/croma_embed.py /opt/scripts/croma_embed.py

# Bake the Base weights into the image's HF cache (Large stays lazy).
RUN python -c "from huggingface_hub import hf_hub_download; \
print('baked', hf_hub_download('antofuller/CROMA', 'CROMA_base.pt'))"

# Build-time smoke test #1: imports + upstream module reachable.
# importlib.metadata.version per ml-containers rules.
RUN python -c "import torch, einops; \
import importlib.metadata as im; \
print('torch', torch.__version__, '| einops', im.version('einops'), '| huggingface_hub', im.version('huggingface_hub')); \
from use_croma import PretrainedCROMA; \
assert callable(PretrainedCROMA)"

# Build-time smoke test #2 (Base, from baked weights): instantiate the
# joint model, run synthetic S1 (2-ch) + S2 (12-ch) 120x120 tensors
# through forward, and assert the encoding/embedding shapes.
# `torch.set_grad_enabled(False)` instead of `with torch.no_grad():`
# because compound statements can't follow `;` in inline Python.
RUN python -c "import torch; torch.set_grad_enabled(False); \
from huggingface_hub import hf_hub_download; \
from use_croma import PretrainedCROMA; \
w = hf_hub_download('antofuller/CROMA', 'CROMA_base.pt'); \
m = PretrainedCROMA(pretrained_path=w, size='base', modality='both', image_resolution=120).eval(); \
o = m(SAR_images=torch.randn(1, 2, 120, 120), optical_images=torch.randn(1, 12, 120, 120)); \
print('shapes:', {k: tuple(v.shape) for k, v in o.items()}); \
assert tuple(o['SAR_GAP'].shape) == (1, 768), o['SAR_GAP'].shape; \
assert tuple(o['optical_GAP'].shape) == (1, 768), o['optical_GAP'].shape; \
assert tuple(o['joint_GAP'].shape) == (1, 768), o['joint_GAP'].shape; \
assert tuple(o['joint_encodings'].shape) == (1, 225, 768), o['joint_encodings'].shape" \
    && du -sh /opt/hf-cache 2>/dev/null \
    && echo 'baked CROMA-Base weights at /opt/hf-cache'

LABEL org.opencontainers.image.title="croma" \
      org.opencontainers.image.description="CROMA — Contrastive Radar-Optical Masked Autoencoders (Fuller, Millard & Green, NeurIPS 2023). Sentinel-1/Sentinel-2-native foundation model; SAR + optical + joint embeddings. CPU runtime; Base weights baked at build time." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/croma/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/antofuller/CROMA" \
      bradleylab.model.paper="https://arxiv.org/abs/2311.00566" \
      bradleylab.model.task="Sentinel-1/Sentinel-2 embedding extraction (radar-optical foundation model)" \
      bradleylab.model.sensor="image:sar,multispectral" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="HuggingFace Hub antofuller/CROMA (MIT); Base baked at build, Large fetched lazily on --variant large"

WORKDIR /work
CMD ["bash"]
