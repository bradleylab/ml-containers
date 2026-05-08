# TerraMind 1.0 — IBM / ESA / FAST-EO any-to-any generative EO foundation model.
#
# TerraMind is a family of multimodal foundation models trained on
# Sentinel-1 GRD, Sentinel-1 RTC, Sentinel-2 L2A, DEM, NDVI, and
# LULC. Four scale variants (tiny / small / base / large) all live
# under the IBM-ESA-Geospatial HF org. The model is fully integrated
# into TerraTorch's BACKBONE_REGISTRY and supports embeddings,
# segmentation heads, and any-to-any modality translation
# (Thinking-in-Modalities approach).
#
# This container ships the same TerraTorch toolkit as the
# bradleylab/prithvi-eo container, plus the diffusers pin TerraMind
# requires for its any-to-any generation. Weights are NOT baked —
# TerraTorch downloads them from HF Hub on first call (e.g.
# `ibm-esa-geospatial/TerraMind-1.0-base`).
#
# Upstream: https://github.com/IBM/terramind        (Apache-2.0)
#           https://github.com/terrastackai/terratorch (Apache-2.0)
# Weights:  https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny
#           https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small
#           https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base
#           https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large
# Stack:    nvidia/cuda 12.1 + Python 3.11 + PyTorch 2.5.1 cu121 +
#           terratorch>=1.2.5 + diffusers==0.30.0 (TerraMind any-to-any pin)

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

# PyTorch 2.5.1 cu121 — H100 sm_90; matches prithvi-eo / satlas / clay.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1

# TerraTorch >=1.2.5 (matches prithvi-eo) + diffusers 0.30.0 (TerraMind
# upstream pins this exact version for any-to-any generation; newer
# diffusers break the generation pipeline per upstream README).
# `setuptools<81` mirrors the upstream README install.
RUN pip install \
        "terratorch>=1.2.5" \
        "diffusers==0.30.0" \
        "huggingface_hub>=0.25.0" \
        "setuptools<81"

RUN mkdir -p /opt/hf-cache

# Smoke test — imports clean and at least one terramind backbone is
# registered. Per ~/.claude/rules/ml-containers.md, version printout
# uses importlib.metadata, not __version__, for every package.
# No weight fetch at build time (HF cache is empty; first runtime
# call downloads).
RUN python -c "import torch, terratorch, diffusers; \
import importlib.metadata as im; \
from terratorch import BACKBONE_REGISTRY; \
backbones = list(BACKBONE_REGISTRY); \
terramind = [b for b in backbones if 'terramind' in b.lower()]; \
assert terramind, f'no terramind backbones in BACKBONE_REGISTRY; sample: {backbones[:20]}'; \
print('terratorch', im.version('terratorch'), \
'| diffusers', im.version('diffusers'), \
'| torch', torch.__version__, \
'| cuda-build', torch.version.cuda); \
print('terramind backbones registered:', len(terramind), \
'(e.g.', terramind[:3], ')')"

LABEL org.opencontainers.image.title="terramind" \
      org.opencontainers.image.description="TerraMind 1.0 (IBM / ESA / FAST-EO) any-to-any generative geospatial foundation model — S1 GRD, S1 RTC, S2 L2A, DEM, NDVI, LULC. Container ships TerraTorch toolkit + diffusers 0.30.0 pin; weights via Hugging Face Hub at runtime." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/terramind/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/IBM/terramind" \
      bradleylab.model.paper="https://arxiv.org/abs/2504.11171" \
      bradleylab.model.task="any-to-any generative EO foundation model — backbone for S1+S2 fusion / segmentation / modality translation; downstream fine-tuning via TerraTorch task heads + Thinking-in-Modalities" \
      bradleylab.model.sensor="image:multi (S1 GRD, S1 RTC, S2 L2A, DEM, NDVI, LULC)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:ibm-esa-geospatial/TerraMind-1.0-{tiny,small,base,large} (runtime fetch into HF_HOME=/opt/hf-cache)"

WORKDIR /work
CMD ["bash"]
