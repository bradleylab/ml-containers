# syntax=docker/dockerfile:1
# ssl4eo-landsat — SSL4EO-L self-supervised encoders for Landsat TM, ETM+ and
# OLI/TIRS (Stewart et al., NeurIPS 2023 Datasets and Benchmarks).
#
# ResNet-18, ResNet-50 and ViT-S/16 backbones pretrained with MoCo v2 and
# SimCLR on the SSL4EO-L corpus, one set per sensor and processing level:
# Landsat 4-5 TM TOA, Landsat 7 ETM+ TOA and SR, Landsat 8-9 OLI/TIRS TOA and
# OLI SR. They load through TorchGeo's weight enums (ResNet18_Weights,
# ResNet50_Weights, ViTSmall16_Weights) as timm models. The TM and ETM+ sets are
# what the lab's other Landsat-capable encoders lack: the Landsat data Clay and
# Prithvi-EO were trained on is Landsat 8-9 only.
#
# ---------------------------------------------------------------------
# WEIGHTS ARE BAKED: the 30 checkpoints TorchGeo 0.10.0 references, 2.28 GB.
# ---------------------------------------------------------------------
# All 30 go in because they are CC0 and small together, and because a compute
# node without egress cannot fetch a checkpoint left out.
# The Hugging Face repository holds 48 .pth files; the other 18 are 12 files of
# about 600 bytes whose state dict is empty and 6 alternate ViT-S/16 files that
# no TorchGeo enum loads, so baking them would add 531 MB nothing can reach.
# They sit in TORCH_HOME/hub/checkpoints, where torchvision's loader looks
# before it downloads.
#
# Code:    https://github.com/microsoft/torchgeo @ v0.10.0 (MIT)
# Weights: https://huggingface.co/torchgeo/ssl4eo_landsat @ 1c88bb5 (CC0-1.0)
# Paper:   https://arxiv.org/abs/2306.09424

FROM python:3.12-slim-bookworm

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the user's ~/.local off sys.path: enroot
# bind-mounts $HOME, and a host package shadowing an image one has broken
# images in this catalog before. TORCH_HOME is fixed so the loader finds the
# baked checkpoints whatever the caller's environment says.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TORCH_HOME=/opt/torch-cache

# TorchGeo 0.10.0 sets no upper bound on torch. 2.13.0 with torchvision 0.28.0
# was the newest pair on PyPI when TorchGeo 0.10.0 was released (2026-08-14).
# The CUDA 12.9 build matches nasa-ibm-lunar-fm, and Compute2's driver
# (580.105.08) supports it.
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu129 \
        torch==2.13.0 torchvision==0.28.0

# constraints.txt is the full dependency set resolved as of TorchGeo 0.10.0's
# release; TorchGeo pins only lower bounds, and floating ranges are what broke
# `terramind` and `prithvi-eo` months after they last built. It also keeps the
# resolver from swapping the CUDA 12.9 torch above for a different build.
COPY constraints.txt /opt/constraints.txt
RUN python -m pip install -c /opt/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu129 torchgeo==0.10.0

COPY scripts/bake_weights.py /opt/ssl4eo/bake_weights.py
COPY scripts/ssl4eo-check scripts/ssl4eo-benchmark /usr/local/bin/

RUN python /opt/ssl4eo/bake_weights.py && du -sh "$TORCH_HOME"

# Build smoke test, with no network so every load must come from the bake.
# torch and torchgeo import in separate steps under faulthandler: a
# compiled-extension segfault in this stack has twice been silent in
# `terramind` and `prithvi-eo` until the imports were split.
RUN --network=none python -X faulthandler -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda); assert torch.version.cuda and torch.version.cuda.startswith('12.9'), 'CUDA 12.9 torch was replaced'"
RUN --network=none python -X faulthandler -c "import importlib.metadata as im, torchgeo.models; print('torchgeo', im.version('torchgeo'), '| timm', im.version('timm'))"
# Loads all 30 checkpoints (TM, ETM+ and OLI at every architecture) and runs a
# forward pass on synthetic input of each one's band count on the CPU.
RUN --network=none python -X faulthandler /usr/local/bin/ssl4eo-check cpu

LABEL org.opencontainers.image.title="ssl4eo-landsat" \
      org.opencontainers.image.description="SSL4EO-L self-supervised ResNet-18, ResNet-50 and ViT-S/16 encoders (MoCo v2, SimCLR) for Landsat 4-5 TM, Landsat 7 ETM+ and Landsat 8-9 OLI/TIRS, TOA and SR, loaded through TorchGeo. All 30 TorchGeo-referenced checkpoints baked" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/ssl4eo-landsat/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT AND CC0-1.0" \
      bradleylab.model.upstream="https://github.com/microsoft/torchgeo" \
      bradleylab.model.paper="https://arxiv.org/abs/2306.09424" \
      bradleylab.model.task="Landsat image encoder (self-supervised pretraining) for embeddings and frozen-backbone segmentation or classification" \
      bradleylab.model.sensor="image:multispectral (Landsat 4-5 TM TOA; Landsat 7 ETM+ TOA and SR; Landsat 8-9 OLI/TIRS TOA and OLI SR)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.weights.source="baked — huggingface.co/torchgeo/ssl4eo_landsat @ 1c88bb5: the 30 checkpoints TorchGeo 0.10.0 references (2.28 GB) in /opt/torch-cache/hub/checkpoints" \
      bradleylab.model.licence.note="TorchGeo code MIT. SSL4EO-L weights and the SSL4EO-L benchmark datasets CC0-1.0"

WORKDIR /work
CMD ["bash", "-c", "echo 'ssl4eo-landsat ready: load weights with torchgeo.models (see README.md); ssl4eo-check verifies every baked checkpoint'"]
