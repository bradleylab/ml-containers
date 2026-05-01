# PointsToWood for H100 — Owen, Allen, Grieve, Wilkes, Lines 2025 (arXiv:2503.04420)
# Semantic leaf–wood classification of TLS point clouds from tree base to
# branch tips. Trained on diverse European forests (RIEGL VZ400i TLS),
# with the "global" checkpoint shipped as a general-purpose model.
#
# Upstream: https://github.com/harryjfowen/PointsToWood
# Stack:    PyTorch 2.5.1 + CUDA 12.1 + PyG (torch-2.5.1+cu121 wheels)
#
# H100 note: CUDA 12.1 natively supports compute 9.0. The upstream README
# specifies CUDA 12.2 + driver 535; CUDA 12.1 wheels are compatible with
# a 12.x runtime and our c2-gpu nodes ship a newer driver.

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
    FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates git-lfs \
        build-essential cmake \
        libgl1 libglib2.0-0 \
        python3.10 python3.10-dev python3.10-distutils \
    && wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.10 /tmp/get-pip.py && rm /tmp/get-pip.py \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.5.1 + CUDA 12.1 (H100 sm_90 supported natively)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# PyG ecosystem — precompiled wheels for torch 2.5.1 + cu121
RUN pip install -f https://data.pyg.org/whl/torch-2.5.1+cu121.html \
        torch-scatter torch-sparse torch-cluster torch-spline-conv && \
    pip install torch-geometric

# Python deps — per README "pandas pykdtree numba" plus runtime helpers
# identified by reading predict.py (tqdm, laspy, plyfile, scikit-learn)
RUN pip install \
        pandas \
        pykdtree \
        numba \
        tqdm \
        laspy[lazrs] \
        plyfile \
        scikit-learn

# Clone repo — default branch is `version1.0-paper`, which ships the
# 73 MB global.pth checkpoint at pointstowood/model/global.pth.
WORKDIR /opt
RUN git clone --depth=1 https://github.com/harryjfowen/PointsToWood.git
WORKDIR /opt/PointsToWood/pointstowood

# Verify weight is in-tree (not LFS stub)
RUN test -s model/global.pth && \
    ls -la model/global.pth && \
    python -c "import torch; sd = torch.load('model/global.pth', map_location='cpu', weights_only=False); print('Checkpoint keys:', list(sd.keys()) if isinstance(sd, dict) else type(sd).__name__)"

LABEL org.opencontainers.image.title="pointstowood" \
      org.opencontainers.image.description="PointsToWood (Owen et al. 2025) DL semantic leaf-wood segmentation for high-resolution TLS; H100 sm_90; in-tree global.pth checkpoint" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/pointstowood/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      bradleylab.model.upstream="https://github.com/harryjfowen/PointsToWood" \
      bradleylab.model.paper="https://arxiv.org/abs/2503.04420" \
      bradleylab.model.task="semantic leaf-wood segmentation, point cloud" \
      bradleylab.model.sensor="lidar:tls" \
      bradleylab.model.status="exploratory" \
      bradleylab.model.weights.source="upstream-repo:in-tree (model/global.pth, version1.0-paper branch)"

CMD ["bash"]
