# KPConv (KPConv-PyTorch) for H100 — kernel point convolution semantic seg
#
# KPConv (Thomas et al., ICCV 2019) — S3DIS / SemanticKITTI semantic
# segmentation. Alt-architecture comparison to the transformer-based
# models in the AEC set (PTv3, SPT, OctFormer).
#
# Upstream code: https://github.com/HuguesTHOMAS/KPConv-PyTorch  (MIT)
# Paper:         arXiv:1904.08889 (ICCV 2019)
#
# ## Build path — no CUDA compile
#
# Unlike the other point-cloud images, KPConv has NO custom CUDA op: the
# KPConv operator is pure PyTorch, and the only compiled extensions are
# two CPU C++11 modules (grid subsampling + radius neighbors via nanoflann)
# built by cpp_wrappers/compile_wrappers.sh with g++ only — no nvcc, no
# GPU at build, no torch coupling (their setup.py use numpy.distutils, not
# torch.utils.cpp_extension). So this base needs build-essential but not
# the CUDA arch dance. sm_90 is handled at runtime by stock torch 2.2.2.
#
# ## Two compatibility constraints (2019-era code on torch 2.2.2)
#
# 1. numpy MUST be < 1.26: the cpp_wrappers' setup.py import
#    `numpy.distutils` (removed in numpy 1.26), and the code uses
#    `np.bool` (removed in numpy 1.24 — patched below). The 2.2.2 base
#    ships numpy 1.26.x, so we downgrade.
# 2. matplotlib is a HARD import dependency (kernels/kernel_points.py
#    imports it at module top, on the inference path) — not optional as
#    the upstream INSTALL implies. MPLBACKEND=Agg keeps it headless.
#    mayavi/PyQt5 are lazy-imported (visualization only) and omitted.
#
# The torch surface is otherwise clean for 2.2.2 (no Variable, no removed
# APIs; bare torch.load works because torch 2.2.2 still defaults
# weights_only=False — a reason to stay at 2.2.2, not go newer).
#
# ## Weights are NOT baked
#
# Pretrained S3DIS checkpoints (Light/Heavy/Deform_KPFCNN) are on Google
# Drive; a live Drive pull in CI is throttling-prone and fragile. Fetch
# once with gdown by file ID and mount at runtime (see README). No
# SemanticKITTI pretrained weights exist upstream (train-only).
#
# ## Scope (v1)
#
# Environment + CPU C++ wrappers compiled + a build-time smoke test
# (numpy<1.26, matplotlib headless, the two compiled wrappers import,
# reach the KPFCNN model class). Ships experimental; runtime validation
# (S3DIS mIoU) needs mounted weights + the S3DIS dataset on Compute2.

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/opt/KPConv-PyTorch

# build-essential (g++) for the CPU C++11 wrappers; no nvcc/cmake needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# numpy < 1.26 (numpy.distutils for the wrappers + np.bool era). The 2.2.2
# base ships 1.26.x; downgrade. scikit-learn (KDTree), PyYAML, matplotlib
# (hard import dep). gdown for fetching the S3DIS weights at runtime.
RUN python -m pip install --upgrade pip && \
    python -m pip install "numpy<1.26" scikit-learn PyYAML matplotlib gdown

# KPConv-PyTorch source (run via PYTHONPATH).
RUN git clone https://github.com/HuguesTHOMAS/KPConv-PyTorch.git /opt/KPConv-PyTorch && \
    cd /opt/KPConv-PyTorch && \
    git checkout d19c575d3fa9fcfd5a74845b5b27aac7e50472c7 && \
    rm -rf .git

# Patch removed numpy alias `np.bool` -> `bool` (SemanticKITTI dataset
# only; not on the S3DIS path, but kept for correctness). The pattern
# matches `np.bool` only when not followed by a word char, so `np.bool_`
# is left intact.
RUN sed -i 's/np\.bool\([^_a-zA-Z0-9]\)/bool\1/g' /opt/KPConv-PyTorch/datasets/SemanticKitti.py

# Compile the two CPU C++11 wrappers in place (g++ only; uses the
# installed numpy headers, no PEP-517 isolation).
RUN cd /opt/KPConv-PyTorch/cpp_wrappers && sh compile_wrappers.sh

WORKDIR /opt/KPConv-PyTorch

# Build-time smoke: numpy is <1.26, matplotlib imports headless, the two
# compiled CPU wrappers import (validates the g++ build), and the KPFCNN
# model class is reachable (which pulls models.blocks + kernel_points,
# i.e. the KPConv core + the matplotlib dependency).
RUN python -c "import importlib.metadata as im; \
import numpy as np; print('numpy', im.version('numpy')); \
assert tuple(int(x) for x in np.__version__.split('.')[:2]) < (1, 26), 'numpy must be <1.26'; \
import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda); \
import sklearn, matplotlib; print('scikit-learn', im.version('scikit-learn'), '| matplotlib', im.version('matplotlib')); \
import cpp_wrappers.cpp_subsampling.grid_subsampling as gs; \
import cpp_wrappers.cpp_neighbors.radius_neighbors as rn; \
print('cpp wrappers compiled: grid_subsampling + radius_neighbors'); \
from models.architectures import KPFCNN; print('reached KPFCNN')"

LABEL org.opencontainers.image.title="kpconv" \
      org.opencontainers.image.description="KPConv (KPConv-PyTorch) kernel-point-convolution point-cloud semantic segmentation (S3DIS/SemanticKITTI), H100; weights mounted at runtime" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/kpconv/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/HuguesTHOMAS/KPConv-PyTorch" \
      bradleylab.model.paper="https://arxiv.org/abs/1904.08889" \
      bradleylab.model.task="point cloud semantic segmentation" \
      bradleylab.model.sensor="lidar:indoor/outdoor" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="NOT baked — S3DIS pretrained on Google Drive (gdown by file ID), mount at runtime; no SemanticKITTI weights upstream"

# Inference is via test_models.py (hardcoded chosen_log, no CLI args) —
# point it at a mounted Log_* checkpoint folder. See README.
CMD ["bash"]
