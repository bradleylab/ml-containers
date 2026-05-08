# TimesFM 2.5 — Google Research time-series foundation model.
#
# TimesFM is a decoder-only foundation model for univariate
# time-series forecasting (Das et al., ICML 2024). Version 2.5
# (Sept 2025) is 200M parameters, supports up to 16k context length,
# and ships an optional 30M continuous-quantile head for probabilistic
# forecasts. Apache-2.0 across code AND weights.
#
# This container ships the upstream `timesfm` package installed from
# GitHub at a pinned commit (PyPI's `timesfm 1.3.0` is the v1/v2
# archive — TimesFM 2.5 is GitHub-only as of this writing). Weights
# are NOT baked — `TimesFM_2p5_200M_torch.from_pretrained(...)` pulls
# from HF Hub on first call.
#
# CPU-primary by design: 200M params + Apache-2.0 across the board
# means short-horizon inference runs comfortably on a laptop. Add a
# CUDA variant later if a batch-forecasting workload at scale lands.
#
# Upstream: https://github.com/google-research/timesfm     (Apache-2.0)
# Weights:  https://huggingface.co/google/timesfm-2.5-200m-pytorch
#           https://huggingface.co/google/timesfm-2.5-200m-transformers
# Paper:    Das et al. (2024) "A decoder-only foundation model for
#           time-series forecasting", ICML 2024 — arXiv:2310.10688
# Stack:    python:3.11-slim + torch 2.5.1 (CPU) + timesfm @ pinned SHA

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    WANDB_MODE=offline

# `git` for the pinned-SHA install. `build-essential` not needed —
# every dependency (torch CPU wheel, numpy, safetensors, huggingface_hub)
# ships manylinux + manylinux_aarch64 wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Torch 2.5.1 CPU wheels (matches RemoteCLIP / neuralhydrology pattern).
# Multi-arch friendly: download.pytorch.org/whl/cpu publishes both
# manylinux1_x86_64 and manylinux_aarch64 wheels for 2.5.1.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

# TimesFM @ pinned commit (2026-04-15). Pre-clone instead of
# `pip install git+...` so the Dockerfile owns the SHA, not pip's
# resolver. The [torch] extra is a no-op once torch is already
# installed but documents the intended backend.
ARG TIMESFM_SHA=d720daa67865
RUN git clone --no-checkout https://github.com/google-research/timesfm /opt/timesfm \
    && cd /opt/timesfm \
    && git checkout ${TIMESFM_SHA} \
    && pip install ".[torch]"

RUN mkdir -p /opt/hf-cache

# Smoke test — imports clean and the 2.5 200M class exists. No weight
# fetch at build time. Per ~/.claude/rules/ml-containers.md, version
# printout uses importlib.metadata for portability.
RUN python -c "import torch, timesfm; \
import importlib.metadata as im; \
assert hasattr(timesfm, 'TimesFM_2p5_200M_torch'), \
       'timesfm.TimesFM_2p5_200M_torch missing — wrong commit?'; \
assert hasattr(timesfm, 'ForecastConfig'), 'timesfm.ForecastConfig missing'; \
print('timesfm', im.version('timesfm'), \
'| torch', torch.__version__, '(cpu wheels)'); \
print('TimesFM_2p5_200M_torch import OK; ForecastConfig OK')"

LABEL org.opencontainers.image.title="timesfm" \
      org.opencontainers.image.description="TimesFM 2.5 (Google Research) — decoder-only time-series foundation model. 200M params, 16k context, continuous-quantile head. CPU-primary; weights via Hugging Face Hub at runtime." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/timesfm/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/google-research/timesfm" \
      bradleylab.model.paper="https://arxiv.org/abs/2310.10688" \
      bradleylab.model.task="univariate time-series forecasting — zero-shot point + continuous-quantile predictions; LoRA fine-tuning via HF Transformers + PEFT" \
      bradleylab.model.sensor="time_series:univariate (any 1D regularly-sampled signal)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:google/timesfm-2.5-200m-pytorch (runtime fetch into HF_HOME=/opt/hf-cache)"

WORKDIR /work
CMD ["bash"]
