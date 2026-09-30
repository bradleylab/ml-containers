# panopticon — Panopticon, an any-sensor Earth-observation foundation model.
#
# Waldmann, Shah et al. (2025), "Panopticon: Advancing Any-Sensor Foundation
# Models for Earth Observation", CVPR 2025 EarthVision workshop. A DINOv2
# ViT-B/14 whose patch embedding accepts any set of channels: each band is
# patchified separately, tagged with a sinusoidal embedding of its center
# wavelength in nm (or a learned SAR polarization/orbit embedding), and a
# cross-attention with one learned query folds the bands of each patch into a
# single token. Pretrained on fMoW, fMoW-Sentinel, SatlasPretrain, MMEarth and
# the EnMAP hyperspectral cubes of SpectralEarth (202 bands). Output is a
# 768-D class token and 768-D patch tokens.
#
# The teacher checkpoint is baked, so a run needs no network. Upstream's hub
# entrypoint `panopticon_vitb14` re-downloads the checkpoint on every call, so
# scripts/panopticon_model.py builds the same network with upstream's
# `_panopticon_vitb14` and loads the baked file with the same strict load.
#
# Upstream: https://github.com/Panopticon-FM/panopticon @ 24a229a (Apache-2.0;
#           DINOv2-derived files carry Meta's Apache-2.0 header, two
#           TorchGeo-derived dataset files are MIT)
# Weights:  https://huggingface.co/lewaldm/panopticon @ c8c2bb9 (declared MIT)
# Paper:    https://arxiv.org/abs/2503.10845

FROM python:3.12-slim-bookworm

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the user's ~/.local off sys.path: enroot bind-mounts
# $HOME, and a host package shadowing an image one has broken images in this
# catalog before. PYTHONPATH makes the lab loader importable from any script.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/scripts \
    PANOPTICON_REPO=/opt/panopticon \
    PANOPTICON_WEIGHTS=/opt/panopticon-weights/panopticon_vitb14_teacher.pth

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Upstream pins torch 2.0.0 from the CUDA 11.7 index for training. Inference
# touches only plain torch (xformers is optional upstream and serves the
# nested-tensor batches of training), so instead of that 2023 stack this uses
# the torch 2.12.1 / CUDA 12.9 build that `nasa-ibm-lunar-fm` runs on Compute2's
# H100s under driver 580.105.08. constraints.txt freezes the rest.
COPY constraints.txt /opt/constraints.txt
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install -c /opt/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        torch==2.12.1 numpy h5py rasterio

# Not pip-installed: setup.py names the package `dinov2` and would pull
# upstream's torch 2.0 / xformers 0.0.18 / cuml-cu11 requirements. torch.hub loads the
# checkout from its own directory instead.
ARG PANOPTICON_REF=24a229afb06a8a57a4c65b00ed615fad3b1fd82b
RUN git clone https://github.com/Panopticon-FM/panopticon.git /opt/panopticon \
    && cd /opt/panopticon \
    && git checkout "$PANOPTICON_REF" \
    && echo "$PANOPTICON_REF" > UPSTREAM_REF \
    && rm -rf .git figs

# The sha256 is the pin: a changed file fails the build instead of shipping.
ARG HF_REVISION=c8c2bb9555819e8b2bcedf5b3b00e3bf531554e7
ARG TEACHER_SHA256=55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e
RUN mkdir -p /opt/panopticon-weights \
    && curl -fsSL -o "$PANOPTICON_WEIGHTS" \
        "https://huggingface.co/lewaldm/panopticon/resolve/${HF_REVISION}/panopticon_vitb14_teacher.pth" \
    && echo "${TEACHER_SHA256}  ${PANOPTICON_WEIGHTS}" | sha256sum -c - \
    && du -h "$PANOPTICON_WEIGHTS"

COPY scripts/ /opt/scripts/

# Build smoke test. torch and the upstream model code import in separate
# steps under faulthandler, so a compiled-extension crash names its step.
RUN python -X faulthandler -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda); assert torch.version.cuda and torch.version.cuda.startswith('12.9'), 'CUDA 12.9 torch was replaced'"
RUN python -X faulthandler -W ignore -c "import sys; sys.path.insert(0, '/opt/panopticon'); import hubconf; assert callable(hubconf._panopticon_vitb14); print('panopticon hubconf OK')"

# Offline forward pass on the baked teacher: RGB and 12-band Sentinel-2
# wavelengths, shapes and finiteness asserted, with sockets disabled.
RUN python -X faulthandler -W ignore /opt/scripts/build_smoke.py

LABEL org.opencontainers.image.title="panopticon" \
      org.opencontainers.image.description="Panopticon — any-sensor Earth-observation foundation model (DINOv2 ViT-B/14 with a channel-attention patch embedding conditioned on each band's center wavelength or SAR category). Teacher weights baked; 768-D class and patch embeddings" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/panopticon/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0 AND CC-BY-4.0" \
      bradleylab.model.upstream="https://github.com/Panopticon-FM/panopticon" \
      bradleylab.model.paper="https://arxiv.org/abs/2503.10845" \
      bradleylab.model.task="any-sensor Earth-observation embedding (image-level class token, patch tokens) for linear probes, kNN and dense heads" \
      bradleylab.model.sensor="image:multispectral,hyperspectral,rgb,sar (bands identified by center wavelength in nm, SAR by polarization/orbit category)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.weights.source="baked — huggingface.co/lewaldm/panopticon @ c8c2bb9, panopticon_vitb14_teacher.pth (396 MB, sha256-pinned) at /opt/panopticon-weights" \
      bradleylab.model.licence.note="Code Apache-2.0 (DINOv2-derived files Apache-2.0 per Meta's headers; two TorchGeo-derived dataset files MIT). Weights labeled CC-BY-4.0, as the upstream GitHub README states; the Hugging Face repository that distributes them declares MIT. Crediting the authors satisfies both. Cite Waldmann, Shah et al. (2025)"

WORKDIR /work
CMD ["bash", "-c", "echo 'panopticon ready: python /opt/scripts/panopticon_embed.py --help (see README.md)'"]
