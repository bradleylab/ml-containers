# geolg-3dfaultnet — 3D seismic fault segmentation
#
# Voxelwise fault / non-fault segmentation of a 3D seismic amplitude volume.
# A 3D U-Net (32/64/128/256) with two additions the paper is named for: a
# local-global feature-enhancement block (parallel plain and dilated 3D
# convolutions) and an attention-guided continuity block that follows a
# windowed 3D self-attention layer. 5,956,226 parameters, 23.8 MB fp32.
#
# Upstream:  https://github.com/letsfly27/GeoLG-3DFaultNet  (MIT)
# Weights:   GitHub Release tag v1.0, asset best_model.pth, 71,617,015 bytes
#            (NOT baked — see scripts/fetch_weights.sh)
# Task:      3D seismic fault segmentation
# Input:     raw little-endian float32 binary, C-order, 128x128x128 (.dat)
#
# PROVENANCE: this upstream has 0 stars, one contributor, and is attached to a
# manuscript whose publication status the lab could not confirm. The image is
# built because the licence and weights are clean, not because the science is
# vetted. Read the "Provenance" section of README.md before trusting any
# output. Do not skip it.
#
# Base-image choice. Upstream states it was developed and tested against
# torch 2.8.0+cu128 on Python 3.9.21, so the requirement this image has to
# satisfy is "exactly torch 2.8.0+cu128" — which means the base must NOT ship
# its own torch. Two consequences:
#
#   1. Not an NGC base. `nvcr.io/nvidia/pytorch:25.04-py3` bakes torch
#      2.7.0a0, an NVIDIA alpha build that is not upstream's tested version,
#      and it exports a global pip constraint file
#      (PIP_CONSTRAINT=/etc/pip/constraint.txt) that pins numpy==1.26.4 and
#      has already broken one image in this repo (see esm/README.md). Here the
#      numpy pin would happen to agree with upstream's, but the torch version
#      would not, and torch is the pin that matters.
#   2. Not `pytorch/pytorch:*`. Same objection inverted — those tags fix torch
#      to whatever the tag says and ship Python 3.11.
#
# So: a plain CUDA runtime base (Python 3.12 from Ubuntu 24.04, the same base
# `esm` uses) with torch installed from the PyPI cu128 index at upstream's
# exact version. Deviating from upstream's Python 3.9 is deliberate and low
# risk — every source file in the repo is plain torch/numpy with no
# version-sensitive syntax, and the build-time smoke test runs a real forward
# pass rather than trusting that.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"

# Upstream's tested build, exactly: torch 2.8.0+cu128 (README.md section 1,
# and torch==2.8.0 with the nvidia-*-cu12 12.8 wheels in requirements.txt).
# The +cu128 local version is what the cu128 index serves; pinning the bare
# version here would let PyPI's default CPU/cu126 wheel win.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0

# The rest of the runtime. Upstream's requirements.txt is a `pip freeze` of the
# author's whole workstation — 80 lines including nnunetv2, batchgenerators,
# pydicom, SimpleITK, vtk and pyvista, none of which any file in the repo
# imports. Installing it verbatim would add several GB and a large surface of
# unrelated pins for no gain, so this image installs the actual import closure
# of the five source files (model, dataset, utils, inference, train), at
# upstream's pinned versions:
#
#   numpy        1.26.4  — volume I/O via np.fromfile; upstream's pin. torch
#                          2.8 wheels are built against the numpy 2 headers,
#                          whose ABI is back-compatible with 1.x, so the pair
#                          is valid; the smoke test exercises it rather than
#                          assuming it.
#   scikit-image 0.24.0  — remove_small_objects, the post-processing step in
#                          inference.py that drops speckle below min_size=50.
#   matplotlib   3.9.4   — the three-panel comparison PNG inference.py writes.
#   tqdm         4.67.1  — imported by dataset.py, so `import dataset` (and
#                          hence train.py) fails without it.
#
# If a future need pulls in one of the dropped packages, add it here with a
# reason; do not restore requirements.txt wholesale.
RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        scikit-image==0.24.0 \
        matplotlib==3.9.4 \
        tqdm==4.67.1

# Upstream ships no package and no release tarball — only a git tree — so the
# pin is a commit SHA. Change it by editing this default and committing, so the
# pin lives in git history; CI passes no build-args. Local experiment:
#   docker build --build-arg GEOLG_REF=<sha> geolg-3dfaultnet/
ARG GEOLG_REF=fe9bdb001ab9086c23ac77491ee2c03e35a38267
ENV GEOLG_REF=${GEOLG_REF} \
    GEOLG_HOME=/opt/GeoLG-3DFaultNet

# The clone carries the two 8 MB sample volumes (data/seis/2.dat and its
# ground truth data/fault/2.dat). They are kept: they are the only public
# input this model has, and they are what SMOKE.md scores against.
RUN git clone https://github.com/letsfly27/GeoLG-3DFaultNet.git "${GEOLG_HOME}" \
    && git -C "${GEOLG_HOME}" checkout --quiet "${GEOLG_REF}" \
    && test "$(git -C "${GEOLG_HOME}" rev-parse HEAD)" = "${GEOLG_REF}" \
    && echo "${GEOLG_REF}" > "${GEOLG_HOME}/.upstream-sha" \
    && rm -rf "${GEOLG_HOME}/.git"

COPY scripts/fetch_weights.sh /usr/local/bin/fetch_weights.sh
RUN chmod +x /usr/local/bin/fetch_weights.sh

# Upstream's modules import each other by bare name (`from model import ...`)
# and inference.py resolves ./best_model.pth, ./data/ and ./predict_images/
# against the working directory. PYTHONPATH lets the scripts be run from a
# writable, bind-mounted working directory instead of from GEOLG_HOME, which
# matters under enroot where the container filesystem is read-only.
ENV PYTHONPATH=/opt/GeoLG-3DFaultNet \
    GEOLG_WEIGHTS_DIR=/opt/geolg-weights \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p /opt/geolg-weights /workspace

# Build-time smoke test: offline, CPU, no weight fetch.
#
# It builds the architecture and runs a real 32^3 forward pass, then asserts
# two structural invariants. Both exist because inference.py calls
# `load_state_dict(..., strict=False)` after rewriting checkpoint key prefixes
# (lgfe.->lg., fcom.->aco.): under strict=False an architecture that has
# drifted from the checkpoint loads *silently* with randomly initialised
# layers and still produces a plausible-looking fault volume. Pinning the
# parameter count and the state-dict key count turns that silent failure into
# a build failure.
#
# Package versions come from importlib.metadata per the ml-containers rule,
# never pkg.__version__.
RUN python -c "import importlib.metadata as im, numpy as np, torch, os; \
torch.set_grad_enabled(False); \
from model import GeoLG3DFaultNet; \
m = GeoLG3DFaultNet(in_channels=1, num_classes=2).eval(); \
n = sum(p.numel() for p in m.parameters()); \
k = len(m.state_dict()); \
y = m(torch.randn(1, 1, 32, 32, 32)); \
assert tuple(y.shape) == (1, 2, 32, 32, 32), y.shape; \
assert n == 5956226, ('param count drifted from the pinned checkpoint', n); \
assert k == 136, ('state_dict key count drifted from the pinned checkpoint', k); \
v = np.fromfile(os.path.join(os.environ['GEOLG_HOME'], 'data/seis/2.dat'), dtype=np.float32); \
assert v.size == 128 ** 3, ('sample volume is not 128^3 float32', v.size); \
print('geolg-3dfaultnet', os.environ['GEOLG_REF'], '| params', n, '| state_dict keys', k, \
      '| torch', torch.__version__, '| numpy', im.version('numpy'), \
      '| scikit-image', im.version('scikit-image'), '| matplotlib', im.version('matplotlib'), \
      '| tqdm', im.version('tqdm'))"

WORKDIR /workspace

LABEL org.opencontainers.image.title="geolg-3dfaultnet" \
      org.opencontainers.image.description="GeoLG-3DFaultNet — 3D seismic fault segmentation (local-global feature enhancement + geometric constraints); torch 2.8.0+cu128 on CUDA 12.8 / Python 3.12; upstream pinned to a commit SHA; weights fetched at runtime" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/geolg-3dfaultnet/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/letsfly27/GeoLG-3DFaultNet" \
      bradleylab.model.paper="unpublished manuscript — publication status unconfirmed as of 2026-08-18" \
      bradleylab.model.task="3D seismic fault segmentation (voxelwise binary)" \
      bradleylab.model.sensor="3D reflection seismic amplitude volume" \
      bradleylab.model.status="experimental — unvalidated provenance, benchmark before use" \
      bradleylab.model.weights.source="github-release:letsfly27/GeoLG-3DFaultNet@v1.0 best_model.pth (71617015 bytes; runtime fetch)" \
      bradleylab.build.geolg_ref="${GEOLG_REF}"
