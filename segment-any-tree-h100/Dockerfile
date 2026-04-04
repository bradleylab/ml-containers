# SegmentAnyTree for H100 (sm_90 via PTX forward-compatibility)
# Stays on CUDA 11.8 ecosystem where both MinkowskiEngine and torchsparse compile,
# but includes PTX for sm_90 so H100 can JIT-compile at runtime.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV CMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"
ENV FORCE_CUDA=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build cmake build-essential libopenblas-dev \
    libsparsehash-dev libtbb-dev libglib2.0-0 libglu1-mesa-dev \
    libosmesa6-dev libsm6 libusb-1.0-0-dev libxcb-shm0 libxext6 \
    libxrender-dev libsndfile1 wget unzip software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-dev python3.10-distutils \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.0.1 + CUDA 11.8 (last version where both ME and torchsparse compile)
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Build MinkowskiEngine from source with PTX for sm_90
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    python setup.py install --force_cuda --blas=openblas && \
    cd .. && rm -rf MinkowskiEngine

# PyG ecosystem for PyTorch 2.0.1 + CUDA 11.8
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html && \
    pip install --no-cache-dir torch-geometric==2.3.1

# torchsparse v1.4.0
RUN pip install --no-cache-dir \
    git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# torch-points-kernels
RUN pip install --no-cache-dir torch-points-kernels==0.7.0

# SegmentAnyTree Python deps
RUN pip install --no-cache-dir \
    hydra-core==1.0.7 \
    omegaconf==2.0.6 \
    wandb \
    open3d \
    "laspy[lazrs]" \
    plyfile \
    h5py \
    scipy \
    numba \
    numpy \
    matplotlib \
    seaborn \
    gdown \
    pytorch-metric-learning \
    python-louvain \
    scikit-learn \
    jaklas \
    dask \
    pykdtree \
    cython \
    hdbscan

# Clone SegmentAnyTree
RUN git clone https://github.com/SmartForest-no/SegmentAnyTree.git /home/nibio/mutable-outside-world

WORKDIR /home/nibio/mutable-outside-world

# Create directories the pipeline expects
RUN mkdir -p /home/datascience/docker_in_folder \
    /home/datascience/tmp_out_folder \
    bucket_in_folder \
    bucket_out_folder

ENTRYPOINT ["bash", "run_oracle_pipeline.sh"]
