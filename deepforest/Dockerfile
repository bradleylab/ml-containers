# DeepForest — aerial RGB tree-crown detection.
#
# Weinstein et al. (DeepForest 2.x) — RetinaNet-style detector trained
# on NEON aerial RGB. The pretrained `weecology/deepforest-tree`
# checkpoint is fetched from Hugging Face Hub on first call to
# `deepforest.main.deepforest()`.
#
# Upstream: https://github.com/weecology/DeepForest
# PyPI:    https://pypi.org/project/deepforest/ (>= 2.1.0)
# Stack:    PyTorch 2.x + torchvision (CUDA 12.1 wheels, H100 sm_90)
#
# Bind-mount the HF cache to /opt/hf-cache and set HF_HOME so the
# model only downloads once per host (Compute2 nodes are sandboxed
# from /home and ~/.cache between jobs).

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
        wget ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
    && wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.11 /tmp/get-pip.py && rm /tmp/get-pip.py \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.x + CUDA 12.1 (H100 sm_90 supported). Install before
# deepforest so deepforest's torch>=2.2 dep is already satisfied with
# CUDA wheels rather than the default cpu wheel.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1

# DeepForest 2.x. opencv-python-headless replaces the default opencv
# pull (avoids libGL conflicts inside container).
RUN pip install \
        "deepforest>=2.1.0,<3" \
        "opencv-python-headless>=4.9" \
        "huggingface_hub>=0.25.0"

# Pre-create the HF cache mount point.
RUN mkdir -p /opt/hf-cache

# Smoke-test: import + instantiate (no inference yet — that needs the
# model fetched, which happens at runtime if HF cache is empty).
RUN python -c "import deepforest, torch; print('deepforest', deepforest.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

WORKDIR /work
CMD ["bash"]
