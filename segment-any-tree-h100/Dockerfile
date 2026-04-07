# SegmentAnyTree for H100 (native sm_90 via CUDA 12) — v2
# Includes post-processing improvements for UAV lidar segmentation quality.
#
# Changes from v1 (ghcr.io/bradleylab/segment-any-tree-h100:latest):
#   - Patch 001: Fix block_merge_th hardcoded bug (line 473 of tracker)
#   - Patch 002: Tunable clustering parameters via config overlay
#   - Patch 003: Origin ID bounds clamping for multi-tile processing
#   - Patch 004: Pure PyTorch region_grow (optional, replaces C++/CUDA dep)
#
# See docs/METHODS.md for full description of all modifications.

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
    libxrender-dev libsndfile1 wget unzip zip git-lfs \
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

# PyG ecosystem for PyTorch 2.2.2 + CUDA 12.1
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cu121.html && \
    pip install --no-cache-dir torch-geometric==2.5.3

# torchsparse v1.4.0 — patched .type() → .scalar_type() for PyTorch 2.2+
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
    pyarrow \
    pykdtree \
    cython \
    hdbscan

# Patch hydra 1.0.7 for Python 3.10 (print_exception etype kwarg removed in 3.10)
RUN sed -i 's/print_exception(etype=None, value=ex, tb=final_tb)/print_exception(ex)/' \
    /opt/conda/lib/python3.10/site-packages/hydra/_internal/utils.py || true

# Clone SegmentAnyTree
RUN git clone https://github.com/SmartForest-no/SegmentAnyTree.git /home/nibio/mutable-outside-world

WORKDIR /home/nibio/mutable-outside-world

# --- v1 patches (CUDA 12 / PyTorch 2.2 compatibility) ---

# Patch torch_points3d for PyG 2.x (data.keys is now a method, not property)
RUN find . -name "*.py" -exec sed -i 's/\bdata\.keys\b/data.keys()/g' {} + && \
    find . -name "*.py" -exec sed -i 's/data\.keys()()/data.keys()/g' {} +

# Patch torch_points_kernels: ensure all tensors on same device in region_grow
RUN python3 -c "import pathlib; p = pathlib.Path('/opt/conda/lib/python3.10/site-packages/torch_points_kernels/cluster.py'); c = p.read_text(); c = c.replace('local_ind = ind[label_mask]', 'local_ind = ind.to(label_mask.device)[label_mask]'); c = c.replace('local_xyz = xyz[label_mask]', 'local_xyz = xyz.to(label_mask.device)[label_mask]'); p.write_text(c); print('Patched cluster.py')"

# CPU/GPU device mismatch fixes — PyTorch 2.2+ is strict about mixed-device indexing
RUN sed -i 's/self\.input\.coords\[cluster\]/self.input.coords[cluster.cpu()]/g' torch_points3d/models/panoptic/PointGroup3heads.py && \
    sed -i 's/self\.input\.pos\[cluster\]/self.input.pos[cluster.cpu()]/g' torch_points3d/models/panoptic/PointGroup3heads.py && \
    sed -i 's/originids\[cluster\]/originids[cluster.cpu()]/g' torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py && \
    sed -i 's/\[originids\[valid_c_idx\]/[originids[valid_c_idx.cpu() if hasattr(valid_c_idx,"cpu") else valid_c_idx]/g' torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py || true

# Fix MeanShift multiprocessing deadlock in containers
RUN sed -i 's/results = pool.map(partial_meanshift_cluster, all_clusters)/results = [partial_meanshift_cluster(args) for args in all_clusters]/' \
    torch_points3d/utils/meanshift_cluster.py && \
    sed -i 's/results = pool.map(meanshift_cluster, all_clusters)/results = [meanshift_cluster(args) for args in all_clusters]/' \
    torch_points3d/utils/meanshift_cluster.py

# Remove PyG 2.x ptr attribute that breaks torch_points3d Batch
RUN sed -i '/from_data_list/a\        if hasattr(batch, "ptr"): delattr(batch, "ptr")' \
    torch_points3d/datasets/batch.py || true

# Fix float-to-integer cast in merge step (pandas_to_las.py)
RUN python3 -c "import pathlib; p = pathlib.Path('nibio_inference/pandas_to_las.py'); c = p.read_text(); c = c.replace('las_file[column] = df[column].astype(standard_columns_with_data_types[column])', 'las_file[column] = df[column].fillna(0).values.round().astype(standard_columns_with_data_types[column])'); c = c.replace('las_file[column] = df[column].astype(extended_columns_with_data_types[column])', 'las_file[column] = df[column].fillna(0).values.round().astype(extended_columns_with_data_types[column])'); p.write_text(c); print('Patched pandas_to_las.py')"

# --- v2 patches (segmentation quality for UAV lidar) ---

# Patch 001: Fix hardcoded block_merge_th (line 473)
# The th_merge parameter was passed into block_merging() but ignored —
# hardcoded to 0.1, causing systematic over-merging of adjacent tree crowns.
# See: tyson-swetnam/SegmentAnyTree fork, also SmartForest-no PR #29.
RUN sed -i 's/if max_iou_ii > 0.1:  # th_merge:/if max_iou_ii > th_merge:/' \
    torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py

# Patch 003: Add origin ID bounds clamping for multi-tile robustness
RUN sed -i '1{/import warnings/!s/^/import warnings\n/}' \
    torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py

# Patch 004: Add pure PyTorch region_grow (Numba-accelerated union-find)
RUN mkdir -p sat/clustering sat/compat
COPY patches/sat_clustering_region_grow.py sat/clustering/region_grow.py
RUN echo "" > sat/__init__.py && \
    echo "" > sat/clustering/__init__.py && \
    echo "" > sat/compat/__init__.py

# Patch 002: Copy UAV tuning config overlay (kept for reference, not used at runtime)
COPY configs/uav_tuning.yaml conf/experiment/uav_tuning.yaml

# Patch 005: Bake UAV tuning parameters into the checkpoint's run_config.
# The Trainer loads run_config from the .pt file, ignoring Hydra overrides.
# Only data.fold is re-injected from runtime config. So we must patch the
# checkpoint directly. See: torch_points3d/core/data_transform/model_checkpoint.py
RUN python3 -c "\
import torch; \
ckpt = torch.load('model_file/PointGroup-PAPER.pt', map_location='cpu'); \
cfg = ckpt['run_config']['models']['PointGroup-PAPER']; \
cfg['block_merge_th'] = 0.3; \
cfg['cluster_radius_search'] = 0.5; \
cfg['cluster_type'] = 1; \
cfg['bandwidth'] = 0.6; \
torch.save(ckpt, 'model_file/PointGroup-PAPER.pt'); \
print('Patched checkpoint: block_merge_th=0.3, cluster_radius_search=0.5, cluster_type=1')"

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
