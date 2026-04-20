# TreeLearn for H100 — Henrich et al. 2024 (Ecol. Informatics)
# Deep-learning instance segmentation of trees from ground-based lidar.
# Training corpus: 6,665 MLS-segmented German beech trees + 877 Wytham
# Woods TLS trees (temperate deciduous). Closest public training match
# for temperate broadleaf TLS/MLS data (e.g. Tyson BP7 oak-hickory).
#
# Upstream: https://github.com/ecker-lab/TreeLearn
# Paper:    https://doi.org/10.1016/j.ecoinf.2024.102795
# Stack:    PyTorch 2.0.0 + CUDA 11.8 + spconv-cu118 (native sm_90)
#
# H100 note: CUDA 11.8 + PyTorch 2.0 officially supports compute 9.0
# (sm_90), and spconv-cu118 wheels ship sm_90 kernels. The CUDA-12.2
# incompatibility the upstream README reports applies to Colab only;
# cu118 is the supported path.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
    FORCE_CUDA=1

# System deps:
#   build-essential, cmake — C-extension compile (hdbscan, jakteristics)
#   libgl1, libglib2.0-0   — open3d / matplotlib runtime
#   software-properties-common — adds deadsnakes for Python 3.10 on 22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates \
        build-essential cmake ninja-build \
        libgl1 libglib2.0-0 \
        python3.10 python3.10-dev python3.10-distutils \
    && wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.10 /tmp/get-pip.py && rm /tmp/get-pip.py \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Modern get-pip.py installs pip without setuptools. munch==2.5.0 (a
# TreeLearn dep) imports pkg_resources at load, which lives in
# setuptools — without it, any tree_learn import crashes.
RUN pip install setuptools wheel

# PyTorch 2.0 + CUDA 11.8 (H100 sm_90 supported natively)
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0

# spconv matching CUDA 11.8 (sparse conv backbone used by TreeLearn)
RUN pip install spconv-cu118

# TreeLearn Python deps — mirror setup/requirements.txt except spconv
# (already installed above). numpy pinned to 1.24.x for compatibility
# with open3d 0.17 + pandas 2.0 in requirements.
RUN pip install \
        munch==2.5.0 \
        pandas==2.0.0 \
        plyfile==0.9 \
        pyyaml==6.0 \
        scikit-learn==1.5.0 \
        six==1.16.0 \
        tqdm==4.66.3 \
        open3d==0.17.0 \
        jakteristics==0.5.1 \
        shapely==2.0.1 \
        geopandas==0.12.2 \
        alphashape==1.3.1 \
        laspy[lazrs]==2.5.1 \
        timm==0.6.12 \
        numpy==1.24.4 \
        hdbscan \
        tensorboard tensorboardX jupyter

# Clone + install TreeLearn
WORKDIR /opt
RUN git clone https://github.com/ecker-lab/TreeLearn.git
WORKDIR /opt/TreeLearn
RUN pip install -e .

# Pretrained weights are NOT baked into the image.
#
# Rationale: the Göttingen dataverse that hosts them has been observed
# returning HTTP 500 (per-file) and connection timeouts for extended
# windows (>30 min). Coupling image build to that server means every
# rebuild risks total failure on transient dataverse problems. Instead,
# weights are downloaded once on the target host (Compute2 scratch)
# into a bind-mounted directory and reused across SLURM jobs.
#
# Use the bundled `download_weights.sh` helper. It retries with backoff
# and accepts success if at least one of the three variants lands:
#   * model_weights_with_small_20241213.pth     (preferred)
#   * model_weights_20241213.pth                (default / ≥10 m trees)
#   * model_weights_diverse_training_data.pth   (older visual variant)
#
# Example on Compute2 (run once into a persistent scratch dir):
#   mkdir -p /scratch2/fs1/alexander.s.bradley/treelearn_weights
#   srun --container-image=/storage1/.../bradleylab+treelearn+latest.sqsh \
#        --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
#        bash -lc '/opt/TreeLearn/download_weights.sh /scratch2/fs1/alexander.s.bradley/treelearn_weights'
# Then point pipeline.yaml `pretrain` at the file under that scratch dir.
RUN mkdir -p /opt/TreeLearn/data/model_weights
COPY download_weights.sh /opt/TreeLearn/download_weights.sh
RUN chmod +x /opt/TreeLearn/download_weights.sh

WORKDIR /opt/TreeLearn
CMD ["bash"]
