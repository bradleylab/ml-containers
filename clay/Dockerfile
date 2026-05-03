# Clay Foundation Model — multi-sensor Earth-observation foundation model.
#
# The Clay Foundation (Made With Clay) — Vision-Transformer Masked
# Autoencoder pretrained on Sentinel-2, Sentinel-1 SAR, Landsat,
# NAIP, and MODIS imagery, producing per-patch embeddings usable for
# similarity search, clustering, or lightweight downstream
# classification with minimal labels.
#
# Tier: Compute2 primary. Batch embedding is where H100 throughput
# matters; single-tile embedding works on laptop GPU but isn't where
# Clay earns its keep.
#
# Upstream: https://github.com/Clay-foundation/model
# Weights:  https://huggingface.co/made-with-clay/Clay  (v1.5/clay-v1.5.ckpt)
# Stack:    nvidia/cuda 12.1 + Python 3.11 + PyTorch 2.5 cu121 + claymodel
#
# claymodel is not yet on PyPI as a versioned release — install
# directly from a pinned git commit for reproducibility. The pinned
# SHA below corresponds to the source tree carrying claymodel
# package version 1.5.0 per upstream pyproject.toml (verified
# 2026-05-01).
#
# Weights NOT baked. The v1.5 checkpoint (~3 GB) downloads on first
# call to `hf_hub_download("made-with-clay/Clay", "v1.5/clay-v1.5.ckpt")`
# into HF_HOME=/opt/hf-cache. Bind-mount a persistent host dir there.

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
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

# PyTorch 2.5.1 cu121 — H100 sm_90; matches deepforest / pointstowood /
# satlas for ABI consistency.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1

# claymodel pinned to a specific upstream commit (claymodel 1.5.0).
# Rebase the SHA when bumping the image version. huggingface_hub is
# already a transitive dep of timm but pin explicitly so weight-fetch
# semantics are stable.
ARG CLAY_GIT_SHA=f14e698f3c237cabf8d28dec669a362d66625381
RUN pip install \
        "git+https://github.com/Clay-foundation/model.git@${CLAY_GIT_SHA}" \
        "huggingface_hub>=0.25.0" \
        "Pillow>=10.0"

RUN mkdir -p /opt/hf-cache

# v1 had a runtime gap: the upstream pip install of claymodel does not
# include configs/metadata.yaml as package data, but
# ClayMAEModule.load_from_checkpoint(...) consults it via a relative
# default path of 'configs/metadata.yaml'. Fetch the file from upstream
# at the pinned SHA and place it at /work/configs/metadata.yaml so the
# default lookup resolves under WORKDIR=/work. Users bind-mounting
# their own data should mount at /data, not /work. CLAY_METADATA_PATH
# is also exported for callers who'd rather pass an explicit absolute
# path via ClayMAEModule(..., metadata_path=os.environ['CLAY_METADATA_PATH']).
RUN mkdir -p /work/configs \
    && wget -qO /work/configs/metadata.yaml \
       "https://raw.githubusercontent.com/Clay-foundation/model/${CLAY_GIT_SHA}/configs/metadata.yaml" \
    && test -s /work/configs/metadata.yaml
ENV CLAY_METADATA_PATH=/work/configs/metadata.yaml

# Smoke test: reach claymodel's public API + torch CUDA build, plus
# verify the bundled metadata.yaml exists. No weight fetch at build
# time. importlib.metadata for portable version
# (per ~/.claude/rules/ml-containers.md).
RUN python -c "import os, torch, claymodel; \
import importlib.metadata as im; \
from claymodel.module import ClayMAEModule; \
from claymodel.datamodule import ClayDataModule; \
assert os.path.exists('/work/configs/metadata.yaml'), 'configs/metadata.yaml missing'; \
print('claymodel', im.version('claymodel'), '| torch', torch.__version__, '| cuda-build', torch.version.cuda); \
print('metadata: /work/configs/metadata.yaml ok'); \
assert ClayMAEModule is not None and ClayDataModule is not None"

LABEL org.opencontainers.image.title="clay" \
      org.opencontainers.image.description="Clay Foundation Model — multi-sensor Earth-observation foundation model (ViT-MAE) for per-patch embeddings; H100 sm_90; weights via Hugging Face Hub at made-with-clay/Clay" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/clay/README.md" \
      org.opencontainers.image.version="v2" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/Clay-foundation/model" \
      bradleylab.model.paper="https://clay-foundation.github.io/model" \
      bradleylab.model.task="multi-sensor Earth-observation foundation model — per-patch embeddings, similarity search, clustering, lightweight downstream classification" \
      bradleylab.model.sensor="image:multi (Sentinel-2, Sentinel-1 SAR, Landsat, NAIP, MODIS)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:made-with-clay/Clay (v1.5/clay-v1.5.ckpt; runtime fetch into HF_HOME=/opt/hf-cache)" \
      bradleylab.upstream.git_sha="${CLAY_GIT_SHA}"

WORKDIR /work
CMD ["bash"]
