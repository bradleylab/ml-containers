# insar-unwrap — learned InSAR phase unwrapping (U-Net family).
#
# Singh & Singh, "When Less Is More: Simplicity Beats Complexity for
# Physics-Constrained InSAR Phase Unwrapping" (ML4RS @ ICLR 2026, oral;
# arXiv 2605.00896). Takes a wrapped interferogram patch plus coherence
# and the line-of-sight unit vector, and regresses unwrapped LOS
# displacement directly — no branch cuts, no network-flow solver, no
# residue handling.
#
# Four architectures were trained under one standardized protocol; the
# paper's result is that the PLAINEST one wins. Vanilla U-Net (7.76 M
# params) beats the 17.21 M-param ASPP hybrid on every metric and is
# ~2.5x faster. Default to `vanilla_unet_model.pth`; the other three
# are shipped for the ablation, not because they are better.
#
# Relevance here: the lab already unwraps Sentinel-1 interferograms with
# SNAP (SNAPHU). This is the learned alternative on the same inputs, and
# a candidate cross-check where SNAPHU produces unwrapping errors.
#
# Upstream code:    https://github.com/prabhjotschugh/When-Less-is-More-InSAR-Phase-Unwrapping  (MIT)
# Upstream weights: https://huggingface.co/Prabhjotschugh/InSAR-Phase-Unwrapping-Models  (CC-BY-4.0, ungated)
# Paper:            https://openreview.net/forum?id=liJldeR5ZX
# Stack:            pytorch/pytorch 2.5.1 + CUDA 12.1 (Python 3.11) + rasterio
#
# Weights are NOT baked. The eight checkpoints total 1.07 GB and are
# fetched at runtime from the Hugging Face Hub into HF_HOME=/opt/hf-cache.
# Bind-mount a persistent host dir there (see README, "Pre-staging the
# weights on Storage3") so a compute node with no outbound network still
# finds them.
#
# BASE-IMAGE CHOICE. This is a small model — 7.76 M params on 128x128
# patches, ~3 ms per patch on an H100 — so a GPU is optional for a single
# patch and unnecessary for a demo. It is NOT optional at frame scale: a
# LiCSAR frame tiled at PATCH_SIZE=128 / STRIDE=64 is tens of thousands of
# patches per interferogram, times a multi-year stack. That is the lab use
# case, so the image is CUDA-capable. It stays on the plain
# `pytorch/pytorch:*-runtime` base rather than an NGC PyTorch image: NGC
# would add ~7 GB and ships PIP_CONSTRAINT=/etc/pip/constraint.txt pinning
# numpy==1.26.4, which constrains this image's dependency resolution for
# no gain (nothing here needs NGC's fused kernels). Same base as
# `dinov3-sat` and `sam2`, so the torch ABI is coherent across the lab's
# GPU images. Torch falls back to CPU when no GPU is visible.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HF_HOME=/opt/hf-cache \
    INSAR_HOME=/opt/insar-unwrap/src \
    PYTHONPATH=/opt/insar-unwrap/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency pins. Upstream's requirements.txt states floors only
# (torch>=2.0.0, numpy>=1.24.0, ...), which does not reproduce; these are
# exact versions inside those floors, chosen to match the rest of
# ml-containers.
#
# torch: 2.5.1 comes from the base image and is deliberately BELOW 2.6.
# torch 2.6 flipped `torch.load` to weights_only=True by default, and both
# upstream's `visualize/result_*.py` and the Hugging Face model card call
# `torch.load(path, map_location=...)` with no weights_only argument. That
# trap was checked against the real checkpoint rather than assumed: on
# torch 2.10, `standardized/vanilla_unet_model.pth` loads fine under the
# weights_only=True default — its payload is {epoch, model, optimizer,
# stats, config, best_val_loss}, all tensors and plain Python types, none
# of which the weights_only unpickler rejects. So a future bump past 2.6
# is safe on this axis; the pin is for ABI coherence with `dinov3-sat` and
# `sam2`, not to dodge the trap.
#
# numpy: 1.26.4 rather than 2.x. rasterio 1.4.3 and scipy 1.14.1 both
# support numpy 2, but the base image's torch 2.5.1 was built against
# numpy 1.x, and nothing here needs numpy 2 features.
#
# cartopy is in upstream's requirements.txt and is deliberately absent —
# its only importer is `figures/map_plot.py`, which redraws the paper's
# global frame-location map. It pulls GEOS/PROJ and is not on any
# inference path. `figures/map_plot.py` will not run in this image.
RUN pip install --upgrade pip wheel \
    && pip install \
        "numpy==1.26.4" \
        "scipy==1.14.1" \
        "scikit-learn==1.5.2" \
        "rasterio==1.4.3" \
        "matplotlib==3.9.2" \
        "seaborn==0.13.2" \
        "tqdm==4.67.1" \
        "requests==2.32.3" \
        "huggingface_hub==0.26.5"

# Upstream at a pinned commit (2026-07-29, the "GRSL standardized protocol"
# reorganization). There is no PyPI package and no setup.py, so the source
# is unpacked rather than pip-installed. Only the code directories are
# kept: the repo also carries ~60 MB of paper figures, result PDFs, and
# PyGMTSAR exploration notebooks under `testing and resources/`, none of
# which belong in a runtime image.
#
# Bumping this SHA requires bumping the image tag too.
ARG INSAR_GIT_SHA=e62c4ea86f52e3447a4120f4a118300acf6393b7
ARG INSAR_REPO=When-Less-is-More-InSAR-Phase-Unwrapping
RUN mkdir -p "${INSAR_HOME}" \
    && curl -fsSL "https://codeload.github.com/prabhjotschugh/${INSAR_REPO}/tar.gz/${INSAR_GIT_SHA}" -o /tmp/insar-src.tar.gz \
    && tar -xzf /tmp/insar-src.tar.gz -C "${INSAR_HOME}" --strip-components=1 \
        "${INSAR_REPO}-${INSAR_GIT_SHA}/train" \
        "${INSAR_REPO}-${INSAR_GIT_SHA}/visualize" \
        "${INSAR_REPO}-${INSAR_GIT_SHA}/data" \
        "${INSAR_REPO}-${INSAR_GIT_SHA}/results" \
        "${INSAR_REPO}-${INSAR_GIT_SHA}/requirements.txt" \
        "${INSAR_REPO}-${INSAR_GIT_SHA}/LICENSE" \
        "${INSAR_REPO}-${INSAR_GIT_SHA}/README.md" \
    && rm /tmp/insar-src.tar.gz \
    && echo "${INSAR_GIT_SHA}" > /opt/insar-unwrap/UPSTREAM_SHA

# Compatibility shim — required, not cosmetic.
#
# At the pinned commit every `train/standardized/train_*.py` opens with
#     sys.path.insert(0, <repo>/train)
#     from train.base_config import BaseConfig, print_protocol_banner
# but `base_config.py` lives at `train/standardized/base_config.py`. The
# reorganization commit moved the file and did not update the import, so
# importing ANY of the four model classes fails with
# `ModuleNotFoundError: No module named 'train.base_config'`. Verified
# against the pinned commit before writing this.
#
# This one line re-exports rather than duplicating base_config.py, so it
# keeps working if upstream edits the hyperparameters — it only supplies
# the module path the upstream import already expects. `standardized` is
# importable as a top-level package here because the training script has
# already put <repo>/train on sys.path[0]. Delete this layer once upstream
# fixes the import.
RUN printf '%s\n' \
    '# Added by ml-containers/insar-unwrap: upstream train/standardized/train_*.py' \
    '# imports `train.base_config`, but the file lives in train/standardized/.' \
    '# Re-export it at the path the upstream import expects.' \
    'from standardized.base_config import (  # noqa: F401' \
    '    BaseConfig,' \
    '    get_base_kwargs,' \
    '    print_protocol_banner,' \
    ')' \
    > "${INSAR_HOME}/train/base_config.py"

RUN mkdir -p /opt/hf-cache

COPY smoke_test.py /opt/insar-unwrap/smoke_test.py

# Smoke test — fully offline (HF_HUB_OFFLINE=1), no weight fetch, so the
# build stays deterministic. The same script is the on-node smoke test
# (see SMOKE.md). Versions inside it are read through importlib.metadata,
# never pkg.__version__ (~/.claude/rules/ml-containers.md).
RUN HF_HUB_OFFLINE=1 python /opt/insar-unwrap/smoke_test.py

LABEL org.opencontainers.image.title="insar-unwrap" \
      org.opencontainers.image.description="Learned InSAR phase unwrapping — U-Net family from Singh & Singh (ML4RS @ ICLR 2026); regresses LOS displacement from wrapped phase + coherence + LOS unit vector. Vanilla U-Net (7.76M params, RMSE 1.07 cm) is the recommended checkpoint; three larger variants ship for the ablation." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/insar-unwrap/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT AND CC-BY-4.0" \
      bradleylab.model.upstream="https://github.com/prabhjotschugh/When-Less-is-More-InSAR-Phase-Unwrapping" \
      bradleylab.model.paper="https://openreview.net/forum?id=liJldeR5ZX" \
      bradleylab.model.task="InSAR phase unwrapping (wrapped phase + coherence + LOS vector -> LOS displacement)" \
      bradleylab.model.sensor="sar:sentinel-1 interferogram (LiCSAR frames, C-band, 0.056 m)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.domain="geodesy-insar" \
      bradleylab.model.weights.source="huggingface-hub:Prabhjotschugh/InSAR-Phase-Unwrapping-Models ({standardized,mixed_precision}/{vanilla,enhanced,attention}_unet_model.pth + hybrid_model.pth; runtime fetch into HF_HOME=/opt/hf-cache)" \
      bradleylab.upstream.git_sha="${INSAR_GIT_SHA}"

WORKDIR /work
CMD ["bash"]
