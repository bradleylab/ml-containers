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
# Working on H100 after two H100 fixes (2026-05-04, this commit):
#   1. torch-points-kernels build-arch: strip hardcoded -arch=sm_35
#      from upstream setup.py, let TORCH_CUDA_ARCH_LIST drive nvcc.
#   2. PointGroup3heads multiprocessing-fork + CUDA deadlock: patch
#      torch_points3d/utils/meanshift_cluster.py to use the spawn
#      start method.
# See README "H100 fixes" for detail. End-to-end inference verified;
# OOD behavior on low-density airborne UAV lidar (≲100 pts/m²) is a
# separate, real concern documented in README "Known caveats".

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

# torch-points-kernels with CUDA 12 + numpy<2 + multi-arch (sm_70 .. sm_90).
# Upstream setup.py hardcodes `-arch=sm_35` in nvcc_flags which overrides the
# per-arch `-gencode` flags PyTorch's CUDAExtension builder injects from
# TORCH_CUDA_ARCH_LIST. Strip the hardcoded flag entirely so the env list wins;
# without this, the resulting kernels run only on sm_35 (or sm_50 with the old
# patch) and stall/fail on H100 (sm_90).
RUN pip install --no-cache-dir "numpy>=1.24,<2.0" "cython<3.0"
RUN git clone https://github.com/torch-points3d/torch-points-kernels.git /tmp/tpk && \
    cd /tmp/tpk && \
    sed -i 's/"numpy<=1.21"/"numpy"/g' setup.py && \
    sed -i 's/"-arch=sm_35", //g' setup.py && \
    grep -q '"-arch=sm_' setup.py && { echo "ERROR: hardcoded arch flag still present in setup.py" >&2; exit 1; } || true && \
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

# Fix multiprocessing-fork + CUDA deadlock in PointGroup3heads clustering.
# torch_points3d/utils/meanshift_cluster.py::cluster_single spawns a
# multiprocessing.Pool with the default fork start method. Forked workers
# inherit the parent's already-initialized CUDA context and deadlock on the
# first sklearn-side numpy/threadpool touch. PointGroup3heads (cluster_type=7,
# the active path for the PointGroup-PAPER checkpoint) calls cluster_single
# inside model.forward(), so this freezes inference end-to-end on H100.
# The 'spawn' context starts workers with a fresh interpreter and rebuilds the
# CUDA context cleanly; sklearn's MeanShift target is module-level and
# pickleable, so spawn works without further changes.
# Reference: PyTorch+multiprocessing fork deadlock — well-documented class.
RUN python3 - <<'PYP'
import pathlib, re
fp = pathlib.Path('/opt/ForAINet/PointCloudSegmentation/torch_points3d/utils/meanshift_cluster.py')
src = fp.read_text()
pat = re.compile(r'\bmultiprocessing\.Pool\(', re.MULTILINE)
new, n = pat.subn('multiprocessing.get_context("spawn").Pool(', src)
if n == 0:
    raise SystemExit("ERROR: meanshift_cluster.py does not contain multiprocessing.Pool() — patch target moved")
fp.write_text(new)
print(f"patched {fp} ({n} substitution(s))")
PYP

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

LABEL org.opencontainers.image.title="forainet" \
      org.opencontainers.image.description="ForAINet (Xiang et al. 2024) panoptic segmentation of airborne lidar — H100 sm_90 port from upstream torch-1.9/CUDA-11.1 stack, with multiprocessing-fork + CUDA deadlock and torch-points-kernels build-arch fixes" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/forainet/README.md" \
      org.opencontainers.image.version="v2" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="NOASSERTION" \
      bradleylab.model.upstream="https://github.com/prs-eth/ForAINet" \
      bradleylab.model.paper="https://doi.org/10.1016/j.rse.2024.114078" \
      bradleylab.model.task="panoptic segmentation, point cloud" \
      bradleylab.model.sensor="lidar:airborne" \
      bradleylab.model.status="working" \
      bradleylab.model.weights.source="github-release:forainet-weights-v1 (mirrored from upstream Dropbox)" \
      bradleylab.model.weights.sha256="97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861" \
      bradleylab.model.notes="upstream has no LICENSE file; treat as research-use only"

ENTRYPOINT ["bash"]
