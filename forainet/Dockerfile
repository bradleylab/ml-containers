# ForAINet for H100 — Xiang et al., ETH PRS group.
# Semantic + instance panoptic segmentation of airborne lidar point
# clouds with PointGroup-style architecture. Trained on FOR-Instance
# (10.5281/zenodo.10828636).
#
# Upstream: https://github.com/prs-eth/ForAINet
# Companion install repo: https://github.com/prs-eth/PanopticSegForLargeScalePointCloud
#
# Stack: PyTorch 2.2.2 + CUDA 12.1 + MinkowskiEngine (CiSong10 sm_90
# fork) + torchsparse 1.4 + torch-geometric ecosystem. This is a
# port — upstream ships a PyTorch 1.9 / CUDA 11.1 Dockerfile that
# cannot target sm_90 (CUDA 11.1's nvcc does not know about it).
#
# This recipe reuses the H100-proven stack from
# segment-any-tree-h100/Dockerfile. Both containers share the
# torch-points3d / MinkowskiEngine backbone, so the same sm_90 +
# CUDA 12 + PyTorch 2.x compatibility patches apply.
#
# ## Status
# EXPERIMENTAL. First end-to-end run scheduled 2026-05-01. The
# combination of (a) ForAINet code base targeting torch 1.9, (b)
# this container running torch 2.2.2, and (c) MinkowskiEngine on
# H100 means breakage at runtime is plausible. Iterate from there.

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
    CMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
    FORCE_CUDA=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps: same set proven on segment-any-tree-h100.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build cmake build-essential libopenblas-dev \
    libsparsehash-dev libtbb-dev libglib2.0-0 libglu1-mesa-dev \
    libosmesa6-dev libsm6 libusb-1.0-0-dev libxcb-shm0 libxext6 \
    libxrender-dev libsndfile1 wget unzip zip git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# CUDA 12 / std::to_address compile fix (libstdc++ 11)
RUN sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' \
    /usr/include/c++/11/bits/shared_ptr_base.h || true

# MinkowskiEngine — CiSong10 CUDA-12 fork with native sm_90.
WORKDIR /workspace
RUN git clone https://github.com/CiSong10/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    git checkout cuda12-installation && \
    python setup.py install --force_cuda --blas=openblas && \
    cd .. && rm -rf MinkowskiEngine

# PyG ecosystem matched to torch 2.2.2 + cu121.
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cu121.html && \
    pip install --no-cache-dir torch-geometric==2.5.3

# torchsparse v1.4.0 with PyTorch 2.2 .type() -> .scalar_type() patches.
RUN git clone --depth 1 --branch v1.4.0 https://github.com/mit-han-lab/torchsparse.git /tmp/torchsparse && \
    cd /tmp/torchsparse && \
    sed -i 's/\.type()/\.scalar_type()/g' \
        torchsparse/backend/convolution/convolution_cuda.cu \
        torchsparse/backend/devoxelize/devoxelize_cuda.cu \
        torchsparse/backend/voxelize/voxelize_cuda.cu && \
    pip install --no-cache-dir --no-build-isolation . && \
    rm -rf /tmp/torchsparse

# torch-points-kernels with CUDA 12 + numpy<2 + sm_50 floor.
RUN pip install --no-cache-dir "numpy>=1.24,<2.0" "cython<3.0"
RUN git clone https://github.com/torch-points3d/torch-points-kernels.git /tmp/tpk && \
    cd /tmp/tpk && \
    sed -i 's/"numpy<=1.21"/"numpy"/g' setup.py && \
    sed -i 's/-arch=sm_35/-arch=sm_50/g' setup.py && \
    sed -i 's/isIntegralType(/at::isIntegralType(/g' cuda/include/utils.h && \
    sed -i 's/isFloatingType(/at::isFloatingType(/g' cuda/include/utils.h && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/tpk

# ForAINet Python deps (mirrors PointCloudSegmentation/Dockerfile + SAT
# overlap). hydra-core / omegaconf pinned to versions that work with the
# torch-points3d code base ForAINet uses.
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
    pyarrow \
    pykdtree \
    cython \
    hdbscan

# hydra 1.0.7 / Python 3.10 print_exception compat
RUN sed -i 's/print_exception(etype=None, value=ex, tb=final_tb)/print_exception(ex)/' \
    /opt/conda/lib/python3.10/site-packages/hydra/_internal/utils.py || true

# Clone ForAINet
RUN git clone https://github.com/prs-eth/ForAINet.git /opt/ForAINet
WORKDIR /opt/ForAINet/PointCloudSegmentation

# torch_points3d compat patches (PyG 2.x, mixed-device tensors).
# Borrowed from segment-any-tree-h100/Dockerfile — same upstream codebase.
RUN find . -name "*.py" -exec sed -i 's/\bdata\.keys\b/data.keys()/g' {} + && \
    find . -name "*.py" -exec sed -i 's/data\.keys()()/data.keys()/g' {} +

# torch_points_kernels: device alignment in region_grow.
RUN python3 -c "import pathlib; p = pathlib.Path('/opt/conda/lib/python3.10/site-packages/torch_points_kernels/cluster.py'); c = p.read_text(); c = c.replace('local_ind = ind[label_mask]', 'local_ind = ind.to(label_mask.device)[label_mask]'); c = c.replace('local_xyz = xyz[label_mask]', 'local_xyz = xyz.to(label_mask.device)[label_mask]'); p.write_text(c); print('Patched cluster.py')"

# ForAINet directories. eval_data + eval_out remain bind-mount
# targets; model_file/ now ships with the checkpoint baked in below.
RUN mkdir -p model_file eval_data eval_out

# Bake the pretrained checkpoint at build time. Mirrored from the
# upstream Dropbox link to a release asset on bradleylab/ml-containers
# so GHA builds are deterministic. SHA-256 verified before use.
ARG WEIGHTS_URL=https://github.com/bradleylab/ml-containers/releases/download/forainet-weights-v1/PointGroup-PAPER.pt
ARG WEIGHTS_SHA256=97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861
RUN wget -q -O model_file/PointGroup-PAPER.pt "${WEIGHTS_URL}" && \
    echo "${WEIGHTS_SHA256}  model_file/PointGroup-PAPER.pt" | sha256sum -c - && \
    ls -la model_file/PointGroup-PAPER.pt

ENTRYPOINT ["bash"]
