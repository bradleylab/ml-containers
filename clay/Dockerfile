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
# v3 adds one runnable task, `clay-embed` (the ENTRYPOINT): Clay v1.5
# embeddings of a 4-band PlanetScope surface-reflectance scene, one per
# chip, under the geospatial executor's entrypoint contract v1. The
# encoder's weights are baked in, cut at build time by
# scripts/convert_checkpoint.py from the published checkpoint at a pinned
# revision and checksum, so the task runs offline. For a shell, run with
# `--entrypoint bash`. The full v1.5 checkpoint (5.16 GB) is not in the
# image; `hf_hub_download("made-with-clay/Clay", "v1.5/clay-v1.5.ckpt")`
# still fetches it into HF_HOME=/opt/hf-cache for other uses.

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

# GeoTIFF reading and writing for the embedding entrypoint; the rasterio wheel
# bundles its own GDAL, independent of the system GDAL above.
RUN pip install "rasterio==1.4.3" "safetensors==0.7.0"

# The v1.5 encoder, cut from the full checkpoint in one layer so the 5.16 GB
# checkpoint never lands in the image. The revision pins the Hub commit and the
# checksum pins the file (the Hub's LFS oid for it).
ARG WEIGHTS_REVISION=70200ebcccdf67bf2a0cb9984c77ddee26c10ed2
ARG WEIGHTS_FILE=v1.5/clay-v1.5.ckpt
ARG WEIGHTS_SHA256=21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0
COPY scripts/convert_checkpoint.py /opt/clay-tools/convert_checkpoint.py
RUN mkdir -p /opt/weights /tmp/checkpoint \
    && python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('made-with-clay/Clay', '${WEIGHTS_FILE}', revision='${WEIGHTS_REVISION}', local_dir='/tmp/checkpoint')" \
    && python /opt/clay-tools/convert_checkpoint.py \
        "/tmp/checkpoint/${WEIGHTS_FILE}" /opt/weights/clay_v1.5_encoder.safetensors \
        --expect-sha256 "${WEIGHTS_SHA256}" \
        --source "huggingface.co/made-with-clay/Clay@${WEIGHTS_REVISION}/${WEIGHTS_FILE}" \
    && rm -rf /tmp/checkpoint

COPY bin/clay-embed /usr/local/bin/clay-embed

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

# The embedding encoder builds and takes the baked weights with no key missing
# or left over, on CPU (no GPU on the runner), and the entrypoint's GeoTIFF
# stack imports.
RUN python -c "import importlib.machinery, importlib.util, sys, torch, rasterio; \
import importlib.metadata as im; \
loader = importlib.machinery.SourceFileLoader('clay_embed', '/usr/local/bin/clay-embed'); \
spec = importlib.util.spec_from_loader('clay_embed', loader); \
clay_embed = importlib.util.module_from_spec(spec); sys.modules['clay_embed'] = clay_embed; \
loader.exec_module(clay_embed); \
encoder = clay_embed.build_encoder(256, torch.device('cpu')); \
print('encoder:', sum(p.numel() for p in encoder.parameters()), 'parameters;', clay_embed.weights_provenance()); \
print('bands:', clay_embed.sensor_constants(clay_embed.METADATA)); \
print('rasterio', im.version('rasterio'), '| gdal', rasterio.__gdal_version__, \
'| safetensors', im.version('safetensors'))"

LABEL org.opencontainers.image.title="clay" \
      org.opencontainers.image.description="Clay Foundation Model — multi-sensor Earth-observation foundation model (ViT-MAE) for per-patch embeddings; H100 sm_90; ENTRYPOINT: per-chip embeddings of a 4-band PlanetScope SR scene (v1.5 encoder weights baked in, geospatial executor contract v1); full checkpoint via Hugging Face Hub at made-with-clay/Clay" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/clay/README.md" \
      org.opencontainers.image.version="v3" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/Clay-foundation/model" \
      bradleylab.model.paper="https://clay-foundation.github.io/model" \
      bradleylab.model.task="multi-sensor Earth-observation foundation model — per-patch embeddings, similarity search, clustering, lightweight downstream classification" \
      bradleylab.model.sensor="image:multi (Sentinel-2, Sentinel-1 SAR, Landsat, NAIP, MODIS)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:made-with-clay/Clay; v1.5 encoder baked at /opt/weights (safetensors), full v1.5/clay-v1.5.ckpt fetched at runtime into HF_HOME=/opt/hf-cache" \
      bradleylab.upstream.git_sha="${CLAY_GIT_SHA}"

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/clay-embed"]
