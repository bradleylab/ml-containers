# Point Transformer V3 (PTv3) for H100 — native sm_90 via CUDA 12.1
#
# Indoor/outdoor point-cloud SEMANTIC segmentation from the Pointcept
# framework (Pointcept/Pointcept). Marquee indoor scan-to-BIM model:
# S3DIS 13-class (ceiling/floor/wall/beam/column/window/door/...).
#
# Upstream: https://github.com/Pointcept/Pointcept  (MIT)
# Paper:    PTv3 arXiv:2312.10035 (CVPR 2024 Oral)
#
# ## Why Pointcept is pinned at tag v1.5.1 (not main)
#
# The only published PTv3 checkpoints (HuggingFace Pointcept/PointTransformerV3)
# were trained for Pointcept **v1.5.1**. Loaded against current main
# (v1.7.0, whose model structure + preprocessing were redesigned in
# v1.5.2) they collapse to ~0.198 mIoU (Pointcept issue #364) — a silent
# failure, not a crash. There is NO separate "corrected" zoo: the HF repo
# IS the zoo, and version-matching the *codebase* to v1.5.1 is the fix.
# So this image checks out Pointcept at v1.5.1 and bakes the v1.5.1 S3DIS
# checkpoint. (state_dict load is torch/CUDA-version-agnostic, so the
# proven 2.2.2/cu121 base below is fine; numerical reproduction vs
# v1.5.1's documented torch 1.12.1/cu113 is the Compute2 mIoU gate's job.)
#
# ## Build path
#
# Base: pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel + spconv-cu121 — the
# same proven sm_90 toolchain as forestformer3d / segment-any-tree-h100.
# -devel (not -runtime) is REQUIRED: Pointcept's pointops/pointops2/
# pointgroup_ops are CUDA extensions compiled here.
#
# ## Compiled-op surface (kept lean)
#
# `import pointcept.models` pulls in EVERY model family at load (the
# registry pattern), so PTv3 alone drags in the others' import-time deps.
# Auditing v1.5.1: torchsparse (spvcnn), dwconv (octformer), clip (PPT),
# torch_points_kernels (stratified) are used only inside methods, so
# their guards (try/except → None) hold and they are skipped. The
# exceptions that must be installed:
#   - ocnn: octformer guards `import ocnn` but still does
#     `class OctreeT(Octree)` at module level, so Octree must be real —
#     ocnn (pure Python) is required for the import to succeed.
#   - pointops / pointops2 / pointgroup_ops: hard module-level imports in
#     point_transformer / v2 / point_group; built from libs/ with
#     --no-build-isolation (their setup.py imports torch at build time,
#     which PEP 517 isolation would hide).
# TORCH_CUDA_ARCH_LIST incl. 9.0 + FORCE_CUDA force sm_90 cubins for the
# compiled libs with no GPU at build time.
#
# ## FlashAttention — off
#
# PTv3 imports flash_attn under try/except; the baked S3DIS
# `semseg-pt-v3m1-0-rpe` config already sets enable_flash=False, so it
# runs flash-free out of the box (no slow GPU-side flash-attn build). The
# ScanNet checkpoint was trained WITH flash for its 77.6% — not baked
# here; S3DIS is the flash-free indoor fit.
#
# ## Scope (v1)
#
# Buildable environment + the v1.5.1 S3DIS checkpoint + the Pointcept
# `tools/test.py` inference path + a build-time smoke test. No turnkey
# raw-cloud wrapper (Pointcept runs on preprocessed datasets; GridSample
# voxelization at 0.02 m happens in the data pipeline). Ships
# experimental pending the Compute2 mIoU gate.

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
    PYTHONPATH=/opt/Pointcept

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates \
        build-essential cmake ninja-build \
        libopenblas-dev libsparsehash-dev \
        libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
# libsparsehash-dev provides <google/dense_hash_map>, which
# pointgroup_ops/src/bfs_cluster.cpp includes — without it that lib's
# CUDA extension fails to compile.

# Patch std::to_address collision with CUDA 12 + g++11 (NVIDIA #596).
RUN sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' \
    /usr/include/c++/11/bits/shared_ptr_base.h || true

# Build tooling. numpy<2 pinned and held throughout — torch 2.2.2 and the
# compiled ops predate the numpy-2 ABI break.
RUN python -m pip install --upgrade pip && \
    python -m pip install "setuptools<70" wheel ninja cython "numpy<2"

# spconv (sparse-conv backbone for PTv3 + sparse_unet); prebuilt cu121
# wheel already ships sm_90 kernels (proven in forestformer3d).
RUN python -m pip install spconv-cu121

# PyG companions from the torch-2.2.2+cu121 index (Pointcept uses
# torch_scatter in PTv3; cluster/sparse for other families it imports).
RUN python -m pip install torch-scatter torch-cluster torch-sparse \
        -f https://data.pyg.org/whl/torch-2.2.2+cu121.html && \
    python -m pip install torch_geometric

# Pointcept framework deps (no PPT/CLIP/ocnn/torchsparse — all guarded).
# tensorboard/tensorboardx/yapf/termcolor are needed by the test engine
# at runtime; addict/einops/timm are import-time deps of the models.
RUN python -m pip install \
        "numpy<2" \
        addict einops timm termcolor ocnn \
        h5py pyyaml scipy scikit-learn plyfile open3d \
        SharedArray tensorboard tensorboardx yapf tqdm

# Pointcept source, pinned at tag v1.5.1 (the codebase the published
# checkpoints were trained for — see header). Run from source via
# PYTHONPATH (set above); not pip-installed.
RUN git clone https://github.com/Pointcept/Pointcept.git /opt/Pointcept && \
    cd /opt/Pointcept && \
    git checkout 72a799353c568c29a3365c02a15129ebe7fe637a && \
    rm -rf .git

# Pointcept's three in-tree CUDA libs. --no-build-isolation so their
# setup.py sees the image's torch; the ENV arch list + FORCE_CUDA emit
# sm_90 cubins with no GPU at build time.
RUN cd /opt/Pointcept && \
    python -m pip install --no-build-isolation \
        ./libs/pointops ./libs/pointops2 ./libs/pointgroup_ops

# Bake the v1.5.1 S3DIS semantic checkpoint (semseg-pt-v3m1-0-rpe, the
# flash-free Area-5 model, ~529 MB) from the HF zoo. Renamed to a flat
# filename for convenience.
RUN mkdir -p /opt/Pointcept/checkpoints && \
    wget -q --tries=3 -O /opt/Pointcept/checkpoints/s3dis-semseg-pt-v3m1-0-rpe.pth \
      https://huggingface.co/Pointcept/PointTransformerV3/resolve/main/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth && \
    ls -lh /opt/Pointcept/checkpoints

WORKDIR /opt/Pointcept

# Build-time smoke test: versions via importlib.metadata, then reach
# PointTransformerV3 — which triggers pointcept.models.__init__ and so
# transitively validates pointops / pointops2 / pointgroup_ops and the
# guarded-import audit. Confirms the S3DIS config + baked checkpoint.
# Cannot launch an sm_90 kernel here — that is the runtime test's job.
RUN python -c "import os, importlib.metadata as im; \
import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda); \
import spconv; print('spconv-cu121', im.version('spconv-cu121')); \
import pointops; print('pointops imported'); \
print('torch_geometric', im.version('torch_geometric'), '| numpy', im.version('numpy')); \
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3; \
print('reached PointTransformerV3'); \
assert os.path.isfile('configs/s3dis/semseg-pt-v3m1-0-rpe.py'), 'missing S3DIS PTv3 config'; \
ck = sorted(os.listdir('checkpoints')); print('baked checkpoints:', ck); \
assert any('s3dis' in c for c in ck), 'expected S3DIS checkpoint'"

# Verify the pointops CUDA extension carries sm_90 cubins (build-time
# guard for the no-GPU build; cuobjdump ships in the -devel base). Hard-
# fail only when cuobjdump reports arches but sm_90 is absent.
RUN POINTOPS_SO=$(python -c "import glob,os,pointops; d=os.path.dirname(pointops.__file__); c=glob.glob(os.path.join(d,'*.so')); print(c[0] if c else '')") && \
    if [ -n "$POINTOPS_SO" ] && command -v cuobjdump >/dev/null 2>&1; then \
      ARCHES=$(cuobjdump "$POINTOPS_SO" 2>/dev/null | grep -ioE "sm_[0-9]+" | sort -u | tr '\n' ' '); \
      echo "pointops ext: $POINTOPS_SO | arches: ${ARCHES:-none-detected}"; \
      if echo "$ARCHES" | grep -q "sm_90"; then echo "pointops: sm_90 cubins present"; \
      elif [ -n "$ARCHES" ]; then echo "ERROR: pointops built without sm_90 (has: $ARCHES)" && exit 1; \
      else echo "WARN: cuobjdump detected no arches; H100 runtime test must confirm sm_90"; fi; \
    else echo "WARN: pointops .so or cuobjdump not found; skipping build-time arch check"; fi

LABEL org.opencontainers.image.title="point-transformer-v3" \
      org.opencontainers.image.description="Point Transformer V3 (PTv3) point-cloud semantic segmentation, H100 sm_90, Pointcept v1.5.1, S3DIS checkpoint baked" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/point-transformer-v3/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/Pointcept/Pointcept" \
      bradleylab.model.paper="https://arxiv.org/abs/2312.10035" \
      bradleylab.model.task="point cloud semantic segmentation" \
      bradleylab.model.sensor="lidar:terrestrial/aerial" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:Pointcept/PointTransformerV3 (v1.5.1 S3DIS), baked at build"

# Inference: `python tools/test.py --config-file configs/s3dis/semseg-pt-v3m1-0-rpe.py
# --options weight=checkpoints/s3dis-semseg-pt-v3m1-0-rpe.pth ...` — see README.
CMD ["bash"]
