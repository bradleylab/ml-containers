# fourcastnet3 — FourCastNet 3 global AI weather forecast (NVIDIA Earth-2)
#
# Probabilistic global medium-range forecast on the 0.25 deg lat/lon grid
# (721 x 1440), 72 surface and pressure-level variables, 6 h steps. Spherical
# neural operator with a stochastic (SFNO-family) core.
#
# Upstream:  https://github.com/NVIDIA/earth2studio  (framework, Apache-2.0)
#            https://github.com/NVIDIA/makani         (the FCN3 network, Apache-2.0)
# Weights:   https://huggingface.co/nvidia/fourcastnet3
#            Apache-2.0, ungated, 2.85 GB (checkpoint 2.84 GB + orography /
#            land mask / normalization arrays).
# Paper:     https://arxiv.org/abs/2507.12144
#
# ---------------------------------------------------------------------------
# Base-image choice
# ---------------------------------------------------------------------------
# A plain CUDA base plus an explicitly installed torch, NOT an NGC PyTorch
# image, for two reasons:
#
#   1. FCN3 ends up requiring torch >= 2.10 whether you ask for it or not.
#      earth2studio's own `fcn3` extra says nothing about torch, but it pulls
#      `makani`, whose pyproject declares an unpinned `nvidia-physicsnemo
#      >= 0.5.0a0`; the current physicsnemo (2.1.x) declares `torch >= 2.10.0`.
#      On an NGC base (25.04 ships torch 2.7.0a0) pip would satisfy that by
#      *replacing* NVIDIA's tuned build with a generic PyPI wheel — the exact
#      shape of the cuequivariance 0.11.1 incident. An NGC tag shipping
#      torch >= 2.10 may well exist by now, but NGC tag listings are not
#      reachable from the build environment, and pinning a base tag nobody has
#      verified is worse than pinning one that has been.
#   2. NGC PyTorch images ship a global pip constraint file
#      (PIP_CONSTRAINT=/etc/pip/constraint.txt) pinning numpy==1.26.4, which
#      has already cost this repo one CI failure (see esm/README.md). Once
#      torch is going to be replaced anyway, the NGC base contributes ~9 GB
#      and a constraint file and nothing else.
#
# `-devel` rather than `-runtime`: torch-harmonics compiles the DISCO CUDA
# kernels from source here (see below), which needs nvcc.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Ubuntu 24.04's system interpreter is Python 3.12. earth2studio requires
# >=3.11,<3.15 and physicsnemo >=3.11,<=3.14, so the tag is asserted rather
# than trusted — a base bump that moved the system python outside that window
# would otherwise fail several layers later with an unrelated-looking error.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && python -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), sys.version"

# torch is installed first and pinned exactly, so everything downstream
# resolves against it instead of dragging in a different build.
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# A pip constraint file of our own, holding torch and torchvision only. Any
# later resolve that wants to move torch now fails loudly at build time
# instead of silently swapping the GPU stack. This is the same mechanism NGC
# uses for numpy, aimed at the package that has actually burned us.
RUN mkdir -p /etc/pip \
    && printf 'torch==%s\ntorchvision==%s\n' "${TORCH_VERSION}" "${TORCHVISION_VERSION}" \
        > /etc/pip/e2s-constraint.txt
ENV PIP_CONSTRAINT=/etc/pip/e2s-constraint.txt

# Reproducibility pins. earth2studio resolves torch-harmonics and makani from
# git for the fcn3 extra via `[tool.uv.sources]`, which pip does not read — so
# `pip install earth2studio[fcn3]` fails outright (makani is not on PyPI) and
# the git revisions have to be restated here. These SHAs are upstream's own,
# copied from earth2studio 0.17.0's pyproject.toml.
ARG EARTH2STUDIO_VERSION=0.17.0
ARG TORCH_HARMONICS_REF=a632ca748a12bd9f74dbc1e00653317810991f74
ARG MAKANI_REF=b38fcb2799d7dbc146fa60459f3f9823394a8bf1
# makani declares `nvidia-physicsnemo>=0.5.0a0` with no upper bound. Pinned
# here so a physicsnemo release cannot change what this image contains.
ARG PHYSICSNEMO_VERSION=2.1.1
ENV EARTH2STUDIO_VERSION=${EARTH2STUDIO_VERSION} \
    TORCH_HARMONICS_REF=${TORCH_HARMONICS_REF} \
    MAKANI_REF=${MAKANI_REF} \
    PHYSICSNEMO_VERSION=${PHYSICSNEMO_VERSION}

# torch-harmonics builds the DISCO CUDA kernels only when it can see a GPU or
# when FORCE_CUDA_EXTENSION=1 is set; the CI runner has no GPU, so without the
# flag the build succeeds and produces a CPU-only wheel that degrades FCN3
# silently at inference time. TORCH_CUDA_ARCH_LIST is limited to 9.0 because
# every Compute2 GPU is an H100 — widening it multiplies compile time for
# architectures the lab does not have.
RUN FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
    pip install --no-cache-dir --no-build-isolation \
        "torch_harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@${TORCH_HARMONICS_REF}"

# One resolve for the rest, so pip sees the whole graph at once. The installed
# torch-harmonics (0.8.1) already satisfies makani's `torch-harmonics>=0.8.0`,
# so pip leaves the CUDA build in place rather than pulling the PyPI wheel;
# the smoke test asserts that outcome instead of assuming it.
RUN pip install --no-cache-dir \
        "earth2studio==${EARTH2STUDIO_VERSION}" \
        "nvidia-physicsnemo==${PHYSICSNEMO_VERSION}" \
        "makani @ git+https://github.com/NVIDIA/makani.git@${MAKANI_REF}" \
        "ruamel.yaml>=0.18.14" \
        "scipy>=1.10.0" \
        "moviepy"

# Weights are NOT baked. earth2studio fetches the checkpoint on the first
# `FCN3.load_model(FCN3.load_default_package())` and caches it here; bind-mount
# a persistent dir over this path on Compute2 so the 2.84 GB checkpoint is
# fetched once. EARTH2STUDIO_DATA_CACHE is left unset in the image so it falls
# back to EARTH2STUDIO_CACHE — override it separately at runtime to keep bulky,
# regenerable ERA5/GFS downloads off the same volume as the weights.
ENV EARTH2STUDIO_CACHE=/root/.cache/earth2studio

# Build-time smoke test — offline, no weight fetch. Versions come from package
# metadata per the ml-containers rule, never `pkg.__version__` (torch is the
# documented exception).
#
# Three things are checked beyond "it imports":
#   * torch is still the pinned build, i.e. nothing in the resolve swapped it;
#   * the DISCO CUDA kernels were actually compiled — the silent-CPU-build
#     failure mode above;
#   * earth2studio's own optional-dependency registry has no failure recorded
#     against fcn3.py. That registry is populated at import time by the
#     `try: import makani ... except ImportError` block in the model module,
#     so an empty entry means the fcn3 extra genuinely resolved.
RUN python -c "import importlib.metadata as im, torch; \
from torch_harmonics.disco import cuda_kernels_is_available; \
from earth2studio.models.px import FCN3; \
from earth2studio.utils.imports import OptionalDependencyFailure as ODF; \
bad = [k for k in ODF.failures if k.endswith('fcn3.py')]; \
print('earth2studio', im.version('earth2studio'), '| makani', im.version('makani'), \
'| torch_harmonics', im.version('torch_harmonics'), '| nvidia-physicsnemo', im.version('nvidia-physicsnemo'), \
'| torch', torch.__version__, '| disco cuda kernels', cuda_kernels_is_available()); \
assert not bad, ('fcn3 optional dependencies failed to import', bad); \
assert torch.__version__.split('+')[0] == '${TORCH_VERSION}', torch.__version__; \
assert cuda_kernels_is_available(), 'torch-harmonics built without CUDA kernels'; \
assert FCN3.load_default_package is not None"

WORKDIR /workspace

LABEL org.opencontainers.image.title="fourcastnet3" \
      org.opencontainers.image.description="FourCastNet 3 global AI weather forecast (NVIDIA Earth-2) — 0.25 deg, 72 variables, 6 h steps; earth2studio + makani, weights fetched at runtime; H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/fourcastnet3/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/NVIDIA/earth2studio" \
      bradleylab.model.paper="https://arxiv.org/abs/2507.12144" \
      bradleylab.model.task="global medium-range weather forecast, probabilistic" \
      bradleylab.model.sensor="ERA5 reanalysis / GFS analysis" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:nvidia/fourcastnet3 (runtime fetch; Apache-2.0, ungated)" \
      bradleylab.build.earth2studio_version="${EARTH2STUDIO_VERSION}" \
      bradleylab.build.torch_version="${TORCH_VERSION}" \
      bradleylab.build.torch_harmonics_ref="${TORCH_HARMONICS_REF}" \
      bradleylab.build.makani_ref="${MAKANI_REF}" \
      bradleylab.build.physicsnemo_version="${PHYSICSNEMO_VERSION}"
