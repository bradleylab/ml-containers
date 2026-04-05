# SegmentAnyTree for H100 (native sm_90 via CUDA 12.8)
# Uses CiSong10 MinkowskiEngine fork for CUDA 12 + patched torchsparse for PyTorch 2.2.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
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

# Patch std::to_address collision with CUDA 12 (issue #596)
RUN sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' \
    /usr/include/c++/11/bits/shared_ptr_base.h || true

# Build MinkowskiEngine from CiSong10 CUDA 12 fork (native sm_90)
WORKDIR /workspace
RUN git clone https://github.com/CiSong10/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    git checkout cuda12-installation && \
    python setup.py install --force_cuda --blas=openblas && \
    cd .. && rm -rf MinkowskiEngine

# PyG ecosystem for PyTorch 2.0.1 + CUDA 11.8
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cu121.html && \
    pip install --no-cache-dir torch-geometric==2.5.3

# torchsparse v1.4.0 — patched for PyTorch 2.2+
# PyTorch deprecated Tensor.type() (returns DeprecatedTypeProperties); the
# AT_DISPATCH macros now require c10::ScalarType from .scalar_type() instead.
# Affects 9 call sites across 3 CUDA files. See:
#   https://github.com/NVIDIAGameWorks/kaolin/issues/865 (identical fix)
#   https://github.com/pytorch/pytorch/issues/29161 (deprecation tracking)
RUN git clone --depth 1 --branch v1.4.0 https://github.com/mit-han-lab/torchsparse.git /tmp/torchsparse && \
    cd /tmp/torchsparse && \
    sed -i 's/\.type()/\.scalar_type()/g' \
        torchsparse/backend/convolution/convolution_cuda.cu \
        torchsparse/backend/devoxelize/devoxelize_cuda.cu \
        torchsparse/backend/voxelize/voxelize_cuda.cu && \
    pip install --no-cache-dir --no-build-isolation . && \
    rm -rf /tmp/torchsparse

# torch-points-kernels: patch for CUDA 12 + PyTorch 2.2 + numpy constraint
RUN pip install --no-cache-dir "numpy>=1.24,<2.0" "cython<3.0"
RUN git clone https://github.com/torch-points3d/torch-points-kernels.git /tmp/tpk && \
    cd /tmp/tpk && \
    sed -i 's/"numpy<=1.21"/"numpy"/g' setup.py && \
    sed -i 's/-arch=sm_35/-arch=sm_50/g' setup.py && \
    sed -i 's/isIntegralType(/at::isIntegralType(/g' cuda/include/utils.h && \
    sed -i 's/isFloatingType(/at::isFloatingType(/g' cuda/include/utils.h && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/tpk

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

# Patch hydra 1.0.7 for Python 3.10 (print_exception etype kwarg removed in 3.10)
RUN sed -i 's/print_exception(etype=None, value=ex, tb=final_tb)/print_exception(ex)/' \
    /opt/conda/lib/python3.10/site-packages/hydra/_internal/utils.py || true

# Clone SegmentAnyTree
RUN git clone https://github.com/SmartForest-no/SegmentAnyTree.git /home/nibio/mutable-outside-world

WORKDIR /home/nibio/mutable-outside-world

# Patch torch_points3d for PyG 2.x (data.keys is now a method, not property)
RUN find . -name "*.py" -exec sed -i 's/\bdata\.keys\b/data.keys()/g' {} + && \
    find . -name "*.py" -exec sed -i 's/data\.keys()()/data.keys()/g' {} +

# Patch torch_points_kernels: ensure all tensors on same device in region_grow
RUN python3 -c "import pathlib; p = pathlib.Path('/opt/conda/lib/python3.10/site-packages/torch_points_kernels/cluster.py'); c = p.read_text(); c = c.replace('local_ind = ind[label_mask]', 'local_ind = ind.to(label_mask.device)[label_mask]'); c = c.replace('local_xyz = xyz[label_mask]', 'local_xyz = xyz.to(label_mask.device)[label_mask]'); p.write_text(c); print('Patched cluster.py')"

# CPU/GPU device mismatch fixes — PyTorch 2.2+ is strict about mixed-device indexing
# Move GPU cluster indices to CPU before indexing CPU tensors
RUN sed -i 's/self\.input\.coords\[cluster\]/self.input.coords[cluster.cpu()]/g' torch_points3d/models/panoptic/PointGroup3heads.py && \
    sed -i 's/self\.input\.pos\[cluster\]/self.input.pos[cluster.cpu()]/g' torch_points3d/models/panoptic/PointGroup3heads.py && \
    sed -i 's/originids\[cluster\]/originids[cluster.cpu()]/g' torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py && \
    sed -i 's/\[originids\[valid_c_idx\]/[originids[valid_c_idx.cpu() if hasattr(valid_c_idx,"cpu") else valid_c_idx]/g' torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py || true

# Fix MeanShift multiprocessing deadlock in containers (saookami debug finding)
# Replace pool.map() with sequential execution to prevent child process deadlock
RUN sed -i 's/results = pool.map(partial_meanshift_cluster, all_clusters)/results = [partial_meanshift_cluster(args) for args in all_clusters]/' \
    torch_points3d/utils/meanshift_cluster.py && \
    sed -i 's/results = pool.map(meanshift_cluster, all_clusters)/results = [meanshift_cluster(args) for args in all_clusters]/' \
    torch_points3d/utils/meanshift_cluster.py

# Remove PyG 2.x ptr attribute that breaks torch_points3d Batch
RUN sed -i '/from_data_list/a\        if hasattr(batch, "ptr"): delattr(batch, "ptr")' \
    torch_points3d/datasets/batch.py || true

# Create all directories the pipeline expects
RUN mkdir -p /home/datascience/docker_in_folder \
    /home/datascience/docker_out_folder \
    /home/datascience/docker_out_folder/final_results \
    /home/datascience/tmp_in_folder \
    /home/datascience/tmp_out_folder \
    /home/datascience/results \
    bucket_in_folder \
    bucket_out_folder

ENTRYPOINT ["bash", "run_oracle_pipeline.sh"]
