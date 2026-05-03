# mars-bench (kerner-lab MOMO) — Mars Orbital foundation model.
#
# MOMO is a Vision-Transformer foundation model pre-trained on ~12M
# samples (~4M each) from three Mars orbital sensors: HiRISE
# (0.25 m/px), CTX (5 m/px), and THEMIS (100 m/px). Ships as a single
# multi-sensor checkpoint plus three sensor-specific checkpoints, in
# ViT-Small / ViT-Base / ViT-Large variants. Mars-Bench wraps it with
# 9 downstream evaluation tasks (4 classification + 5 segmentation):
# crater segmentation, boulder detection, dust devils, S5Mars rover
# surface, DoMars16k landmark classification, and others.
#
# Tier: Compute2 H100 sm_90. ViT-Base is small enough to run on a
# laptop GPU in principle, but the canonical use is batch processing
# across HiRISE/CTX/THEMIS image strips where H100 throughput matters.
#
# Upstream code:    https://github.com/kerner-lab/MOMO   (MIT)
# Upstream weights: https://huggingface.co/Mirali33/MOMO (CC-BY-4.0)
# Stack:            nvidia/cuda 12.1 + Python 3.11 + PyTorch 2.5 cu121
#                   + pytorch-lightning + hydra-core + timm 0.6.12 +
#                   segmentation-models-pytorch + albumentations +
#                   rasterio + shapely (full requirement.txt baked in)
#
# Weights NOT baked. Fetch at runtime via `huggingface_hub`:
#   from huggingface_hub import hf_hub_download
#   ckpt = hf_hub_download("Mirali33/MOMO", "vit-b-16/momo.pth")
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

# PyTorch 2.5.1 cu121 — H100 sm_90; matches satlas / clay / prithvi-eo
# for ABI consistency. MOMO's requirement.txt allows torch >=2.4,<2.6,
# so 2.5.1 fits.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1

# Upstream MOMO has a self-conflict in its requirement.txt: it pins
# `timm==0.6.12` (an old version) but lists `segmentation-models-pytorch`
# without a version, which resolves to smp 0.5.x that itself pins
# `timm>=0.9`. Resolve by installing a compatible smp BEFORE MOMO so
# pip leaves the existing smp alone (MOMO has no smp version pin to
# fight us). smp 0.3.x is the last series compatible with timm 0.6.x
# (smp 0.4.0 bumped its timm floor); pinning <0.4 lets pip pick the
# best 0.3.x release.
RUN pip install \
        "timm==0.6.12" \
        "segmentation-models-pytorch<0.4"

# kerner-lab/MOMO at a pinned commit (2026-05-01). Bumping the SHA
# requires bumping the image tag too. The package installs three
# top-level modules: `models`, `utils`, `datasets_finetune` (note:
# generic names — fine inside this container, but a user installing
# this on a host alongside other tools could see import collisions).
ARG MOMO_GIT_SHA=a837ab5ea1e4b1711bb3d06cc21606eb3d49df25
RUN pip install \
        "git+https://github.com/kerner-lab/MOMO.git@${MOMO_GIT_SHA}" \
        "huggingface_hub>=0.25.0" \
        "Pillow>=10.0"

RUN mkdir -p /opt/hf-cache

# Smoke test: torch + cuda build, huggingface_hub, and the three MOMO
# top-level packages import cleanly. No weight fetch at build time.
# importlib.metadata for portable version
# (per ~/.claude/rules/ml-containers.md).
RUN python -c "import torch, huggingface_hub; \
import importlib.metadata as im; \
import models, utils, datasets_finetune; \
from models import models_vit; \
print('torch', torch.__version__, '| cuda-build', torch.version.cuda); \
print('huggingface_hub', im.version('huggingface_hub')); \
print('timm', im.version('timm')); \
assert models_vit is not None"

LABEL org.opencontainers.image.title="mars-bench" \
      org.opencontainers.image.description="kerner-lab MOMO — Mars Orbital foundation model (ViT) pre-trained on HiRISE / CTX / THEMIS imagery; weights via Hugging Face Hub at Mirali33/MOMO; Mars-Bench downstream task heads via the same upstream repo." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/mars-bench/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT AND CC-BY-4.0" \
      bradleylab.model.upstream="https://github.com/kerner-lab/MOMO" \
      bradleylab.model.paper="https://huggingface.co/Mirali33/MOMO" \
      bradleylab.model.task="Mars orbital imagery foundation model — pretrained ViT for downstream crater segmentation / boulder detection / dust devils / rover surface / landmark classification" \
      bradleylab.model.sensor="image:multi (HiRISE 0.25 m/px, CTX 5 m/px, THEMIS 100 m/px)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:Mirali33/MOMO (vit-{s,b,l}-16/{momo,hirise,ctx,themis}.pth; runtime fetch into HF_HOME=/opt/hf-cache)" \
      bradleylab.upstream.git_sha="${MOMO_GIT_SHA}"

WORKDIR /work
CMD ["bash"]
