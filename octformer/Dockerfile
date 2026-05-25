# OctFormer for H100 — octree-transformer point-cloud semantic segmentation
#
# OctFormer (Wang, SIGGRAPH 2023) — ScanNet / ScanNet200 semantic
# segmentation on an octree transformer. Alt-architecture comparison to
# point-transformer-v3 / superpoint-transformer in the AEC set.
#
# Upstream code: https://github.com/octree-nn/octformer  (MIT)
# Paper:         arXiv:2305.03045 (SIGGRAPH 2023)
#
# ## Build path
#
# Base pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel, same proven sm_90
# toolchain as the sibling images. -devel is REQUIRED: dwconv compiles a
# CUDA extension.
#
# Stack: ocnn==2.2.6 (the version OctFormer pins — pure PyTorch, NO Triton
# implicit-gemm kernels; those arrived in ocnn 2.3.0 and need a newer
# Triton than torch 2.2.2 bundles, so the pin is load-bearing — do not let
# it float). thsolver==1.2.0. OctFormer itself is pure Python; the only
# CUDA compile is dwconv (octree depthwise conv).
#
# dwconv's setup.py imports torch and sets no arch flags, so it builds
# with --no-build-isolation + TORCH_CUDA_ARCH_LIST (incl. 9.0) + FORCE_CUDA
# to emit sm_90 cubins with no GPU at build time.
#
# ## Weights are NOT baked
#
# The ScanNet / ScanNet200 checkpoints are on OneDrive, which returns 403
# to every non-interactive download (no scriptable fetch). They are also
# non-commercial: bound by the ScanNet Terms of Use (the weights license
# is unstated upstream, but ScanNet data terms restrict to research and
# forbid redistribution), so they cannot be re-hosted publicly. Download
# them interactively, stage to NAS / Compute2 scratch, and mount at
# runtime — see README. This image ships the buildable environment only.
#
# ## Scope (v1)
#
# Environment + dwconv compiled for sm_90 + a build-time smoke test
# (import ocnn, import dwconv to exercise the compiled .so, reach the
# OctFormer model class, cuobjdump-verify sm_90). Ships experimental;
# runtime validation (ScanNet mIoU) needs mounted weights + preprocessed
# data on Compute2.

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
    CMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
    FORCE_CUDA=1 \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PYTHONPATH=/opt/octformer

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates \
        build-essential cmake ninja-build \
        libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Patch std::to_address collision with CUDA 12 + g++11 (NVIDIA #596) —
# defensive, before dwconv's CUDA compile.
RUN sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' \
    /usr/include/c++/11/bits/shared_ptr_base.h || true

RUN python -m pip install --upgrade pip && \
    python -m pip install "setuptools<70" wheel ninja "numpy<2"

# OctFormer's requirements (minus mmdet3d detection extras). ocnn pinned
# to 2.2.6 (Triton-free); protobuf bounded for tensorboard compat.
RUN python -m pip install \
        "numpy<2" \
        ocnn==2.2.6 \
        thsolver==1.2.0 \
        yacs tqdm scipy plyfile tensorboard \
        "protobuf>=3.1.0,<=3.20.0" \
        scikit-image trimesh wget timm pandas

# OctFormer source (pure Python; run via PYTHONPATH).
RUN git clone https://github.com/octree-nn/octformer.git /opt/octformer && \
    cd /opt/octformer && \
    git checkout c2975ab5739bff50d8fe5cfdc2e3f802482e3c89 && \
    rm -rf .git

# dwconv — the lone CUDA extension. --no-build-isolation so its setup.py
# sees the image's torch; ENV arch list + FORCE_CUDA emit sm_90 cubins
# with no GPU at build time.
RUN git clone https://github.com/octree-nn/dwconv.git /opt/dwconv && \
    cd /opt/dwconv && \
    git checkout ae53057eaf36dab01aa2727fcc93a749fd995af5 && \
    python -m pip install --no-build-isolation . && \
    rm -rf /opt/dwconv/.git

WORKDIR /opt/octformer

# Build-time smoke: versions via importlib.metadata; import dwconv
# explicitly (OctFormer lazy-imports it inside CPE.__init__, so a broken
# compile would otherwise pass), then reach the OctFormer model class.
RUN python -c "import importlib.metadata as im, torch; \
print('torch', torch.__version__, '| cuda', torch.version.cuda); \
import ocnn; print('ocnn', im.version('ocnn')); \
import dwconv; print('dwconv', im.version('dwconv'), '| so', dwconv.__file__); \
from models.octformer import OctFormer; print('reached OctFormer')"

# Verify the dwconv CUDA extension carries sm_90 cubins (build-time guard;
# cuobjdump ships in the -devel base). Hard-fail only when cuobjdump
# reports arches but sm_90 is absent.
RUN DWCONV_SO=$(python -c "import glob,os,dwconv; d=os.path.dirname(dwconv.__file__); c=glob.glob(os.path.join(d,'core*.so'))+glob.glob(os.path.join(d,'*.so')); print(c[0] if c else '')") && \
    if [ -n "$DWCONV_SO" ] && command -v cuobjdump >/dev/null 2>&1; then \
      ARCHES=$(cuobjdump "$DWCONV_SO" 2>/dev/null | grep -ioE "sm_[0-9]+" | sort -u | tr '\n' ' '); \
      echo "dwconv ext: $DWCONV_SO | arches: ${ARCHES:-none-detected}"; \
      if echo "$ARCHES" | grep -q "sm_90"; then echo "dwconv: sm_90 cubins present"; \
      elif [ -n "$ARCHES" ]; then echo "ERROR: dwconv built without sm_90 (has: $ARCHES)" && exit 1; \
      else echo "WARN: cuobjdump detected no arches; H100 runtime test must confirm sm_90"; fi; \
    else echo "WARN: dwconv .so or cuobjdump not found; skipping build-time arch check"; fi

LABEL org.opencontainers.image.title="octformer" \
      org.opencontainers.image.description="OctFormer octree-transformer point-cloud semantic segmentation (ScanNet/ScanNet200), H100 sm_90; weights mounted at runtime" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/octformer/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/octree-nn/octformer" \
      bradleylab.model.paper="https://arxiv.org/abs/2305.03045" \
      bradleylab.model.task="point cloud semantic segmentation" \
      bradleylab.model.sensor="lidar:indoor" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="NOT baked — OneDrive (not scriptable) + ScanNet non-commercial terms; mount at runtime"

# Inference is dataset-batch eval (no single-cloud CLI):
# python scripts/run_seg_scannet.py --run validate --ckpt <mounted>/best_model.pth
# See README. Default to an interactive shell.
CMD ["bash"]
