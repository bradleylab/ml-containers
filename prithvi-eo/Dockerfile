# Prithvi-EO — IBM/NASA geospatial foundation model.
#
# Prithvi-EO is a family of ViT-based foundation models (v1.0 100M;
# v2.0 300M and 600M, with TL "temporal+location" variants) trained
# by IBM Research and NASA on Harmonized Landsat–Sentinel-2 (HLS)
# data. Pre-fine-tuned task variants exist for burn-scar mapping,
# flood detection, and multi-temporal crop classification.
#
# This container ships the IBM-supported [TerraTorch](https://github.com/IBM/terratorch)
# fine-tuning toolkit, which provides the canonical inference and
# training entrypoints for Prithvi via the BACKBONE_REGISTRY +
# Lightning task scaffolding. Prithvi weights themselves are NOT
# baked — TerraTorch downloads them from Hugging Face Hub on first
# call (e.g. `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL`).
#
# Upstream:  https://github.com/IBM/terratorch  (Apache-2.0)
# Weights:   https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
#            https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL
#            https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
# Stack:     nvidia/cuda 12.1 + Python 3.11 + PyTorch 2.5 cu121 +
#            terratorch + Lightning + segmentation-models-pytorch +
#            torchgeo + timm + diffusers (transitive)

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
    NO_ALBUMENTATIONS_UPDATE=true \
    WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
        wget git ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
        gdal-bin libgdal-dev \
    && wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.11 /tmp/get-pip.py && rm /tmp/get-pip.py \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.5.1 cu121 — H100 sm_90; matches satlas / clay / deepforest.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1

# TerraTorch pulls in a heavy transitive set (Lightning, torchgeo,
# segmentation-models-pytorch, diffusers, timm, geopandas, rasterio).
# Loose lower bound; verify on first build.
RUN pip install \
        "terratorch>=1.2.5" \
        "huggingface_hub>=0.25.0"

RUN mkdir -p /opt/hf-cache

# Smoke test: import terratorch + reach BACKBONE_REGISTRY (which is
# what registers Prithvi backbones). No weight fetch at build time.
# importlib.metadata for portable version
# (per ~/.claude/rules/ml-containers.md — applies to every package
# referenced in the printout, not just the headline).
RUN python -c "import torch, terratorch; \
import importlib.metadata as im; \
from terratorch import BACKBONE_REGISTRY; \
print('terratorch', im.version('terratorch'), '| torch', torch.__version__, '| cuda-build', torch.version.cuda); \
assert BACKBONE_REGISTRY is not None"

LABEL org.opencontainers.image.title="prithvi-eo" \
      org.opencontainers.image.description="Prithvi-EO (IBM/NASA) geospatial foundation model — ViT pre-trained on HLS (Harmonized Landsat-Sentinel-2). Container ships TerraTorch fine-tuning toolkit; weights via Hugging Face Hub at runtime." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/prithvi-eo/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/IBM/terratorch" \
      bradleylab.model.paper="https://arxiv.org/abs/2412.02732" \
      bradleylab.model.task="geospatial foundation model — pretrained backbone for burn-scar / flood / crop classification / segmentation; downstream fine-tuning via TerraTorch task heads" \
      bradleylab.model.sensor="image:multi (HLS — Harmonized Landsat 8/9 + Sentinel-2)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:ibm-nasa-geospatial/Prithvi-EO-{1.0-100M,2.0-300M,2.0-300M-TL,2.0-600M,2.0-600M-TL} (runtime fetch into HF_HOME=/opt/hf-cache)"

WORKDIR /work
CMD ["bash"]
