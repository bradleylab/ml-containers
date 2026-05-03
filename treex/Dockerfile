# treeX — unsupervised tree instance segmentation in dense forest point clouds.
#
# Burmeister, Tockner, Reder, Engel, Richter, Mund, Döllner (2025).
# "treeX: Unsupervised Tree Instance Segmentation in Dense Forest Point
# Clouds." arXiv:2509.03633.
#
# Re-engineered classical / unsupervised tree segmentation, multi-platform
# (TLS / PLS / ULS), evaluated on Wytham Woods + FOR-instance (reported
# ULS F1 = 0.58 with the TreeXPresetULS preset).
#
# Upstream package: pointtree (https://github.com/ai4trees/pointtree)
#   - PyPI:   https://pypi.org/project/pointtree/
#   - Author: Josafat-Mattias Burmeister (Universität Potsdam)
#   - License: MIT
#
# This image installs only what TreeXAlgorithm needs. The companion
# CoarseToFineAlgorithm in the same package requires torch + torch-scatter
# + a learned semantic-segmentation checkpoint and is intentionally NOT
# supported by this container. A separate container variant could be
# built later if we need CoarseToFine.
#
# Stack:
#   - python:3.11-slim (Debian bookworm)
#   - pointtree 1.0.1 from PyPI (compiles a pybind11 C++ extension at
#     install time → cmake + ninja + g++ are required at build time)
#   - pointtorch from upstream main (matches upstream Dockerfile: PyPI
#     1.0.3 lags the source, force-reinstall picks up bugfixes)
#   - circle_detection from upstream main (matches upstream Dockerfile)
#
# CPU-only by design. TreeXAlgorithm is classical / unsupervised — no
# torch, no GPU, no weights. Build context contains a thin wrapper
# (`scripts/run_treex.py`) that takes a LAS/LAZ tile, runs the ULS
# preset, and writes a segmented LAZ.
#
# pyclesperanto-prototype note: pointtree pyproject lists
# pyclesperanto-prototype>=0.22.0,<1.0.0. Versions ≥0.24.5 re-introduced
# a numpy<2.0.0 pin in their wheel METADATA, which conflicts with
# pointtree's numpy>=2.3.0 requirement. We pin <0.24.5 so pip resolves
# to 0.24.4 (no numpy<2 pin). pyclesperanto is not actually called by
# the TreeXAlgorithm code path, so a missing OpenCL runtime is non-fatal
# at runtime — but pyopencl's *import* needs libOpenCL.so.1, hence
# ocl-icd-libopencl1 in apt deps.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps:
#   build-essential, cmake, ninja-build, g++  — pointtree pybind11 build
#   git                                       — pip install from VCS
#   libgomp1                                  — numba / scikit-learn OpenMP
#   libgl1, libglib2.0-0                      — rasterio / matplotlib runtime
#   ocl-icd-libopencl1                        — pyopencl import
#   ca-certificates, curl                     — TLS for PyPI / git
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        g++ \
        git \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        ocl-icd-libopencl1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel setuptools

# Install numpy >=2.3 first so the resolver locks it in before
# pyclesperanto-prototype is considered. Pin pyclesperanto-prototype
# below the numpy<2 reintroduction (see header note).
RUN pip install 'numpy>=2.3,<3'

# Install pointtree 1.0.1 from PyPI. scikit-build-core + pybind11 will
# compile the C++ extensions (_tree_x_algorithm_cpp, _operations_cpp)
# during this step.
RUN pip install \
        'pyclesperanto-prototype>=0.22.0,<0.24.5' \
        'pointtree==1.0.1'

# Match upstream Dockerfile's force-reinstall of pointtorch + circle_detection
# from git main. PyPI lags source; the upstream's own CI uses these
# git installs to pick up bugfixes that haven't been released yet.
RUN pip install --force-reinstall --no-deps \
        'git+https://github.com/ai4trees/pointtorch.git' \
        'git+https://github.com/josafatburmeister/circle_detection.git'

# Wrapper script for one-shot tile segmentation.
WORKDIR /opt/treex
COPY scripts/ ./scripts/
RUN chmod +x ./scripts/run_treex.py

# Build-time smoke test: import pointtree, reach TreeXAlgorithm + presets,
# and print versions of every dependency we care about. Network-free —
# no weight fetches (treeX is unsupervised and ships no weights).
RUN python -c "import importlib.metadata as im; \
import numpy, scipy, sklearn, rasterio; \
from pointtree.instance_segmentation import TreeXAlgorithm, TreeXPresetTLS, TreeXPresetULS; \
import pointtorch; \
from pointtorch.operations.numpy import voxel_downsampling; \
import circle_detection; \
import CSF; \
preset = TreeXPresetULS(); \
algo = TreeXAlgorithm(**preset); \
print('pointtree', im.version('pointtree'), \
'| pointtorch', im.version('pointtorch'), \
'| circle_detection', im.version('circle_detection'), \
'| cloth-simulation-filter', im.version('cloth-simulation-filter'), \
'| numpy', im.version('numpy'), \
'| scipy', im.version('scipy'), \
'| scikit-learn', im.version('scikit-learn'), \
'| rasterio', im.version('rasterio'), \
'| numba', im.version('numba'), \
'| pygam', im.version('pygam'), \
'| pyclesperanto-prototype', im.version('pyclesperanto-prototype')); \
assert TreeXAlgorithm is not None and algo is not None"

LABEL org.opencontainers.image.title="treex" \
      org.opencontainers.image.description="treeX (Burmeister et al. 2025) unsupervised tree instance segmentation for dense forest point clouds (TLS/PLS/ULS); CPU classical, no learned components." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/treex/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/ai4trees/pointtree" \
      bradleylab.model.paper="https://doi.org/10.48550/arXiv.2509.03633" \
      bradleylab.model.task="tree-instance segmentation, point cloud (classical, unsupervised)" \
      bradleylab.model.sensor="lidar:tls,pls,uls" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="none (classical algorithm)"

WORKDIR /work
CMD ["bash"]
