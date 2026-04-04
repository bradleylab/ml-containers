# SegmentAnyTree for H100 (sm_90 via PTX forward-compatibility)
# Stays on CUDA 11.8 ecosystem where both MinkowskiEngine and torchsparse compile,
# but includes PTX for sm_90 so H100 can JIT-compile at runtime.
# Uses PyTorch base image to get Python 3.10 + torch 2.0.1 + CUDA 11.8 pre-installed.
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV CMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"
ENV FORCE_CUDA=1
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build cmake build-essential libopenblas-dev \
    libsparsehash-dev libtbb-dev libglib2.0-0 libglu1-mesa-dev \
    libosmesa6-dev libsm6 libusb-1.0-0-dev libxcb-shm0 libxext6 \
    libxrender-dev libsndfile1 wget unzip git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Build MinkowskiEngine from source with PTX for sm_90
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    python setup.py install --force_cuda --blas=openblas && \
    cd .. && rm -rf MinkowskiEngine

# PyG ecosystem for PyTorch 2.0.1 + CUDA 11.8
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cu118.html && \
    pip install --no-cache-dir torch-geometric==2.5.3

# torchsparse v1.4.0
RUN pip install --no-cache-dir \
    git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# torch-points-kernels: PyPI 0.7.0 pins numpy<=1.21 which won't install on
# Python 3.10. Clone master, patch out the numpy constraint, build in-place.
RUN pip install --no-cache-dir "numpy>=1.24,<2.0" "cython<3.0"
RUN git clone https://github.com/torch-points3d/torch-points-kernels.git /tmp/tpk && \
    cd /tmp/tpk && \
    sed -i 's/"numpy<=1.21"/"numpy"/g' setup.py && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/tpk

# SegmentAnyTree Python deps
RUN pip install --no-cache-dir \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    wandb \
    open3d \
    "laspy[lazrs]" \
    plyfile \
    h5py \
    scipy \
    numba \
    matplotlib \
    seaborn \
    gdown \
    pytorch-metric-learning \
    python-louvain \
    scikit-learn \
    torchnet \
    torchfile \
    tensorboard \
    visdom \
    pandas \
    tqdm \
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
