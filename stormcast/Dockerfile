# stormcast — StormCast v1 convection-allowing regional nowcast (NVIDIA Earth-2)
#
# Autoregressive 1 h-step forecast on the HRRR 3 km CONUS grid, conditioned on
# a coarse global forecast (GFS by default). A deterministic U-Net predicts the
# next state and an EDM-preconditioned diffusion model sharpens it, sampled
# with an 18-step Heun sampler.
#
# Upstream:  https://github.com/NVIDIA/earth2studio  (framework, Apache-2.0)
#            https://github.com/NVIDIA/physicsnemo   (network defs, Apache-2.0)
# Weights:   https://huggingface.co/nvidia/stormcast-v1-era5-hrrr
#            Apache-2.0, ungated, 0.80 GB (EDMPrecond 0.48 GB +
#            StormCastUNet 0.32 GB + metadata.zarr.zip).
# Paper:     https://arxiv.org/abs/2408.10958
#
# ---------------------------------------------------------------------------
# Base-image choice
# ---------------------------------------------------------------------------
# Same reasoning as fourcastnet3/Dockerfile, and it applies more directly here:
# earth2studio's `stormcast` extra requires `nvidia-physicsnemo>=2.0`, and the
# current physicsnemo (2.1.x) declares `torch>=2.10.0`. On an NGC PyTorch base
# (25.04 ships torch 2.7.0a0) pip satisfies that by replacing NVIDIA's tuned
# torch with a generic PyPI wheel — the cuequivariance 0.11.1 failure again.
# NGC also carries a global PIP_CONSTRAINT pinning numpy==1.26.4, which has
# already cost this repo a CI failure (esm/README.md). So: a plain CUDA base
# plus an explicit, pinned torch.
#
# `-devel` rather than `-runtime` for parity with the two sibling Earth-2
# images, which do compile CUDA extensions. Nothing here needs nvcc; the cost
# is image size and the benefit is one base to keep current across the three.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Ubuntu 24.04's system interpreter is Python 3.12. earth2studio requires
# >=3.11,<3.15 and physicsnemo >=3.11,<=3.14 — asserted rather than trusted,
# so a base bump outside that window fails here and not five layers later.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && python -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), sys.version"

ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# Our own pip constraint file, holding torch and torchvision only, so any
# later resolve that wants to move torch fails at build time instead of
# silently swapping the GPU stack.
RUN mkdir -p /etc/pip \
    && printf 'torch==%s\ntorchvision==%s\n' "${TORCH_VERSION}" "${TORCHVISION_VERSION}" \
        > /etc/pip/e2s-constraint.txt
ENV PIP_CONSTRAINT=/etc/pip/e2s-constraint.txt

ARG EARTH2STUDIO_VERSION=0.17.0
# earth2studio asks for `nvidia-physicsnemo>=2.0` with no ceiling. Left alone,
# two builds a month apart would ship different networks. 2.1.1 is the newest
# release and carries the modules StormCast imports
# (physicsnemo.diffusion.preconditioners.legacy.EDMPrecond,
# physicsnemo.models.diffusion_unets.StormCastUNet) — both were verified
# present at the v2.1.1 tag before pinning.
ARG PHYSICSNEMO_VERSION=2.1.1
ENV EARTH2STUDIO_VERSION=${EARTH2STUDIO_VERSION} \
    PHYSICSNEMO_VERSION=${PHYSICSNEMO_VERSION}

# Everything the `stormcast` extra declares, resolved in one pass with the
# physicsnemo version pinned. Unlike the fcn3 and cosmo extras, nothing here
# comes from git, so this is `earth2studio[stormcast]` with one version fixed.
RUN pip install --no-cache-dir \
        "earth2studio==${EARTH2STUDIO_VERSION}" \
        "nvidia-physicsnemo==${PHYSICSNEMO_VERSION}" \
        "einops>=0.8.1" \
        "nvtx>=0.2.11" \
        "omegaconf>=2.3.0" \
        "pyproj>=3.7.1" \
        "scipy>=1.15.2"

# Weights are NOT baked — fetched on the first
# `StormCast.load_model(StormCast.load_default_package())` and cached here.
# Bind-mount a persistent dir over this path on Compute2. StormCast also pulls
# HRRR and GFS fields every run; point EARTH2STUDIO_DATA_CACHE at scratch
# separately so those regenerable downloads do not accumulate beside the
# weights.
ENV EARTH2STUDIO_CACHE=/root/.cache/earth2studio

# Build-time smoke test — offline, no weight fetch. Versions from package
# metadata per the ml-containers rule, never `pkg.__version__`.
#
# The load-bearing check is the last assert: earth2studio records an entry in
# OptionalDependencyFailure.failures keyed by the model module's filename when
# its `try: from physicsnemo... except ImportError` block fires. An empty entry
# for stormcast.py means every symbol StormCast needs resolved against the
# physicsnemo version pinned above — which is exactly what a physicsnemo
# reshuffle would break, and what a bare `import earth2studio` would not catch.
RUN python -c "import importlib.metadata as im, torch; \
from earth2studio.models.px import StormCast; \
from earth2studio.utils.imports import OptionalDependencyFailure as ODF; \
bad = [k for k in ODF.failures if k.endswith('stormcast.py')]; \
print('earth2studio', im.version('earth2studio'), '| nvidia-physicsnemo', im.version('nvidia-physicsnemo'), \
'| tensordict', im.version('tensordict'), '| torch', torch.__version__); \
assert torch.__version__.split('+')[0] == '${TORCH_VERSION}', torch.__version__; \
assert not bad, ('stormcast optional dependencies failed to import', bad); \
assert StormCast.load_default_package is not None"

WORKDIR /workspace

LABEL org.opencontainers.image.title="stormcast" \
      org.opencontainers.image.description="StormCast v1 (NVIDIA Earth-2) — convection-allowing 3 km HRRR-grid nowcast, 1 h steps, regression + EDM diffusion; earth2studio + physicsnemo, weights fetched at runtime; H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/stormcast/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/NVIDIA/earth2studio" \
      bradleylab.model.paper="https://arxiv.org/abs/2408.10958" \
      bradleylab.model.task="convection-allowing regional nowcast, 3 km CONUS" \
      bradleylab.model.sensor="HRRR analysis + GFS forecast conditioning" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:nvidia/stormcast-v1-era5-hrrr (runtime fetch; Apache-2.0, ungated)" \
      bradleylab.build.earth2studio_version="${EARTH2STUDIO_VERSION}" \
      bradleylab.build.torch_version="${TORCH_VERSION}" \
      bradleylab.build.physicsnemo_version="${PHYSICSNEMO_VERSION}"
