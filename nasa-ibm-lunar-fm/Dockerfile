# nasa-ibm-lunar-fm — the NASA-IBM Lunar Foundation Model (NASA-IBM LFM).
#
# A ViT-B encoder-decoder pretrained from scratch on SomBench: about two
# million co-registered lunar tile bundles, 11 modalities, at two scales
# (LROC NAC ~1 m/px and LROC WAC ~100 m/px). It adapts TerraMind's
# masked-token recipe to the Moon, adds per-tile illumination geometry as an
# encoder input, and trains NAC and WAC tiles in one loop so a single set of
# weights spans the 100x scale gap. Fine-tuning and inference run through
# TerraTorch, as for `terramind` and `prithvi-eo`.
#
# Released alongside the backbone are three fine-tuned checkpoints: crater
# detection (NAC and WAC), irregular mare patch segmentation, and polar ice
# prospectivity. The last regresses an expert fuzzy-overlay map, not measured
# ice, and upstream says so plainly; so does this image's README.
#
# Unrelated to `lunarfm` (FDL / Trillium, PolyForm Strict), which the lab also
# runs. Different authors, data, architecture and license.
#
# ---------------------------------------------------------------------
# WEIGHTS ARE NOT BAKED.
# ---------------------------------------------------------------------
# The backbone is 2.4 GB, the four downstream checkpoints 5.6 GB together, and
# the nine VQ-VAE tokenizers (generation only) another 10.3 GB. All are
# Apache-2.0 and ungated, so nothing forbids baking them; they stay out
# because they are large and each task needs a different subset. The lab's
# copy is staged on Storage3 at Active/nasa-ibm-lunar-fm, each repository
# pinned to a Hugging Face revision, and is mounted at /weights.
#
# The upstream configs resolve three paths relative to the working directory
# (terratorch_integration/, backbone/, data/). `lfm-link` creates them, so a
# run is: cd to a scratch directory, lfm-link, terratorch test|predict -c ...
#
# Upstream: https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model @ d54c67a
#           Apache-2.0; NOTICE credits 4M/MultiMAE and others, all Apache-2.0 or MIT
# Weights:  https://huggingface.co/collections/nasa-ibm-ai4science/nasa-ibm-lunar-fm-and-downstream-models-6a96f44c430791be886ca331
# Paper:    https://arxiv.org/abs/2609.13283

FROM python:3.12-slim-bookworm

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the user's ~/.local off sys.path: enroot
# bind-mounts $HOME, and a host package shadowing an image one has broken
# images in this catalog before. NO_ALBUMENTATIONS_UPDATE stops albumentations
# checking PyPI on import, which a compute node without egress waits out.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    NO_ALBUMENTATIONS_UPDATE=1

# libglib2.0-0 and libgl1 for the OpenCV that albumentations pulls in.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Upstream requires torch>=2.12,<3 with torchvision>=0.27,<0.28, which fixes
# the pair at 2.12 / 0.27. Of the CUDA 12.6, 12.8, 12.9 and 13.0 indexes,
# 2.12.1 is on 12.9 only (2.12.0 is on 12.6 and 13.0). Compute2's H100 nodes
# run driver 580.105.08, new enough for 12.9 (checked 2026-09-27).
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu129 \
        torch==2.12.1 torchvision==0.27.1

# Editable install, as upstream's README does: the configs and the trained text
# tokenizer are read from the source tree, so /opt/ni-lfm is the one copy.
# constraints.txt is the full resolved dependency set, frozen at the model's
# release; upstream pins only ranges, and floating ranges are what broke
# `terramind` and `prithvi-eo` months after they last built. It also keeps the
# resolver from swapping the CUDA 12.9 torch above for a different build.
ARG LFM_REF=d54c67aad513cb9daca444afa425cfb278e4fbf8
COPY constraints.txt /opt/constraints.txt
RUN git clone https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model.git /opt/ni-lfm \
    && cd /opt/ni-lfm \
    && git checkout "$LFM_REF" \
    && python -m pip install -c /opt/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu129 -e .

COPY scripts/lfm-link scripts/lfm-verify-benchmarks /usr/local/bin/

RUN mkdir -p /weights /work

# Build smoke test, offline and weight-free. torch and terratorch import in
# separate steps under faulthandler: a compiled-extension segfault in this
# stack has twice been silent in `terramind` and `prithvi-eo` until the
# imports were split.
RUN python -X faulthandler -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda); assert torch.version.cuda and torch.version.cuda.startswith('12.9'), 'CUDA 12.9 torch was replaced'"
RUN python -X faulthandler -c "import importlib.metadata as im, terratorch; print('terratorch', im.version('terratorch'))"
RUN python -X faulthandler -c "\
import importlib.metadata as im; \
from pathlib import Path; \
import terratorch_integration; \
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY as REG; \
import ni_lfm.models.ni_lfm; \
variants = ['ni_lfm_v1_tiny', 'ni_lfm_v1_base', 'ni_lfm_v1_large']; \
missing = [v for v in variants if v not in REG]; \
assert not missing, f'backbones not registered: {missing}'; \
cfgs = sorted(Path('/opt/ni-lfm/terratorch_integration/configs').rglob('*.yaml')); \
assert len(cfgs) > 50, f'only {len(cfgs)} configs in the source tree'; \
print('ni_lfm', im.version('ni_lfm'), '| terratorch', im.version('terratorch'), '| torchvision', im.version('torchvision'), '| lightning', im.version('lightning'), '| timm', im.version('timm'), '| torchgeo', im.version('torchgeo')); \
print('registered:', variants, '| configs:', len(cfgs)); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="nasa-ibm-lunar-fm" \
      org.opencontainers.image.description="NASA-IBM Lunar Foundation Model — ViT-B multimodal, multi-resolution lunar foundation model (LROC NAC ~1 m/px and WAC ~100 m/px), with fine-tuned crater detection, irregular mare patch segmentation and polar ice prospectivity checkpoints, run through TerraTorch. Weights mounted at runtime" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/nasa-ibm-lunar-fm/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model" \
      bradleylab.model.paper="https://arxiv.org/abs/2609.13283" \
      bradleylab.model.task="lunar remote-sensing foundation model; fine-tuned crater detection (NAC, WAC), irregular mare patch segmentation, polar ice-prospectivity regression" \
      bradleylab.model.sensor="image:multi (LROC NAC, LROC WAC, derived terrain; Diviner-derived polar layers for ice prospectivity)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.weights.source="NOT baked — huggingface.co/nasa-ibm-ai4science: backbone 2.4 GB, four downstream checkpoints 5.6 GB, nine generation tokenizers 10.3 GB; staged lab-side on Storage3 at Active/nasa-ibm-lunar-fm, mounted at /weights" \
      bradleylab.model.licence.note="Apache-2.0 for the code and for the backbone and all released downstream checkpoints. The SomBench benchmark datasets are CC-BY-4.0. Ice-prospectivity outputs emulate an expert fuzzy-overlay map, not measured ice"

WORKDIR /work
CMD ["bash", "-c", "echo 'nasa-ibm-lunar-fm ready: mount the staged weights at /weights, then run lfm-link in a scratch directory (see README.md)'"]
