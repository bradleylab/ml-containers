# Superpoint Transformer (SPT + SuperCluster) for H100 — native sm_90 via CUDA 12.1
#
# Two models from one codebase (drprojects/superpoint_transformer):
#   - SPT          large-scale point-cloud SEMANTIC segmentation
#   - SuperCluster PANOPTIC segmentation (superpoint-graph clustering)
# Both run from the same install; they differ only in the Hydra experiment
# config and checkpoint. Bundling them is correct here — it is one
# repository, not two models (cf. the one-model-per-container rule's
# variant-tag clause).
#
# Upstream: https://github.com/drprojects/superpoint_transformer  (MIT)
# Papers:   SPT          arXiv:2306.08045 (ICCV 2023)
#           SuperCluster arXiv:2401.06704 (3DV 2024)
#
# ## Build path
#
# Base: pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel — same proven sm_90
# toolchain as segment-any-tree-h100 / forestformer3d. -devel (not
# -runtime) is REQUIRED: FRNN compiles a CUDA extension and needs nvcc +
# CUDA headers at build time.
#
# Upstream install.sh (master, read 2026-05-20) pins torch==2.2.0 and
# Python 3.8. We deliberately deviate on two points, both for build
# safety:
#   1. We do NOT reinstall torch. The base image already ships torch
#      2.2.2; reinstalling 2.2.0 on top would desync the bundled
#      torchvision and is the classic trigger for FRNN's
#      `_C undefined symbol` ABI failure (FRNN issues #21/#23 — torch
#      version skew between build and runtime). One torch version, frozen
#      before any extension compiles. PyG companion wheels for
#      torch-2.2.0+cu121 load fine under torch 2.2.2 (compatible within
#      the 2.2.x minor).
#   2. Python is 3.10 (base image). SPT supports "Python 3.8+"; pgeof and
#      the PyG wheels ship cp310 binaries, so 3.10 is fine.
#
# ## FRNN is the only source compile — and the sm_90 lever
#
# pgeof / pycut-pursuit / pygrid-graph install as prebuilt manylinux
# wheels (no nvcc). FRNN is the lone exception: it builds a CUDA
# extension, and its setup.py adds NO `-gencode` arch flags of its own —
# it leans on torch's cpp_extension, which auto-detects arch from GPUs
# visible at build time. CI builders have no GPU, so without an explicit
# arch list the FRNN .so would ship with no sm_90 cubin. Hence
# TORCH_CUDA_ARCH_LIST (incl. 9.0) + FORCE_CUDA=1 below, set before FRNN
# builds. (install.sh sets neither because it assumes a GPU build host.)
#
# ## Scope of this image (v1)
#
# This image delivers the buildable environment + the four baked
# checkpoints matching our use case + the repo's own `src/eval.py` Hydra
# inference path + a build-time import smoke test. SPT has NO turnkey
# "raw LAS in -> labeled LAS out" path: inference runs on data
# preprocessed into a dataset structure, and the superpoint partition is
# a cached pre-transform (see README "Running inference"). A custom
# BaseDataset reader for arbitrary clouds + the Compute2 mIoU/PQ
# validation gate are the documented follow-up, not part of v1.
#
# ## Validation note
#
# Build-time `import frnn` loads the .so but cannot launch an sm_90
# kernel (no GPU in CI). Per the catalog's run-time-test rule, arch
# correctness is only proven by a real forward pass on an H100 — see the
# README's Compute2 section. Build-smoke is necessary, not sufficient.

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
    PYTHONPATH=/opt/superpoint_transformer

# System deps:
#   git/wget/ca-certificates — clone repos, fetch checkpoints
#   build-essential/cmake/ninja-build — FRNN CUDA extension compile
#   libopenblas-dev — BLAS for numpy / FRNN
#   libgl1/libglib2.0-0/libgomp1/libx11-6/libxext6/libxrender1 — open3d
#     and numba runtime shared libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates \
        build-essential cmake ninja-build \
        libopenblas-dev \
        libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Patch std::to_address collision with CUDA 12 + g++11 (NVIDIA #596).
# Same defensive fix the other sm_90 images use; harmless if it no-ops.
RUN sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' \
    /usr/include/c++/11/bits/shared_ptr_base.h || true

# Build tooling. setuptools<70 keeps the legacy cpp_extension setup.py
# path FRNN uses working. numpy<2 is pinned up front and held throughout:
# this generation of the stack (torch 2.2, torch_geometric 2.3, numba,
# open3d, plyfile) predates the numpy-2 ABI break and silently crashes on
# import if numpy 2.x sneaks in transitively.
RUN python -m pip install --upgrade pip && \
    python -m pip install "setuptools<70" wheel ninja cython "numpy<2"

# PyG ecosystem — only the companions SPT's install.sh installs
# (pyg_lib + torch_scatter + torch_cluster; NOT torch_sparse /
# torch_spline_conv, which current SPT master no longer uses). Wheels are
# keyed to torch-2.2.0+cu121 and load under the base's torch 2.2.2.
RUN python -m pip install pyg_lib torch_scatter torch_cluster \
        -f https://data.pyg.org/whl/torch-2.2.0+cu121.html && \
    python -m pip install torch_geometric==2.3.0 torchmetrics==0.11.4

# SPT Python deps, mirroring install.sh minus the Jupyter stack (this is
# an inference image, not a notebook env). Explicit pins follow upstream:
# plotly==5.9.0, rich<=14.0, torchmetrics==0.11.4 (above). hydra-core and
# pytorch-lightning are unpinned upstream; we constrain them to the
# README-tested lines (Hydra 1.3, Lightning 2.2+) for reproducibility.
# torch-ransac3d (GPU ground-RANSAC) and torch-graph-components (EZ-SP
# GPU partition) are installed for import fidelity; we ship SPT +
# SuperCluster (not EZ-SP), so either may be dropped if it fights the
# build.
RUN python -m pip install \
        "numpy<2" \
        matplotlib \
        plotly==5.9.0 \
        plyfile \
        h5py \
        colorhash \
        seaborn \
        numba \
        pyrootutils \
        "hydra-core>=1.3,<1.4" \
        hydra-colorlog \
        hydra-submitit-launcher \
        "rich<=14.0" \
        torch_tb_profiler \
        wandb \
        open3d \
        gdown \
        torch-ransac3d \
        pgeof \
        pycut-pursuit \
        pygrid-graph \
        torch-graph-components \
        "pytorch-lightning>=2.2,<2.3"

# FRNN — fixed-radius nearest-neighbour CUDA extension (lone source
# compile). Pinned to a specific commit. prefix_sum (a git submodule)
# builds first, then FRNN. `python -m pip` (not bare pip) avoids the
# wrong-interpreter trap from FRNN issue #24. --no-build-isolation is
# required: both setup.py files import torch at build time, and PEP 517
# build isolation would hide the image's torch (ModuleNotFoundError on
# the first CI build). TORCH_CUDA_ARCH_LIST + FORCE_CUDA (set in ENV
# above) force sm_90 cubins despite no GPU at build time.
RUN git clone https://github.com/lxxue/FRNN.git /opt/FRNN && \
    cd /opt/FRNN && \
    git checkout efc6b378d45000c5de6234098b5092464eb92a24 && \
    git submodule update --init --recursive && \
    cd external/prefix_sum && python -m pip install --no-build-isolation . && \
    cd /opt/FRNN && python -m pip install --no-build-isolation . && \
    rm -rf /opt/FRNN/.git

# SPT source — pinned to the master commit whose install.sh this
# Dockerfile was written against. Run from src/ (no pip-install; the repo
# is a Hydra project, exposed via PYTHONPATH set above).
RUN git clone https://github.com/drprojects/superpoint_transformer.git /opt/superpoint_transformer && \
    cd /opt/superpoint_transformer && \
    git checkout eb959b61226f60e0c037cc105c56d51318650e8a && \
    rm -rf .git

# Bake the four (here five) checkpoints matching our demo goals — indoor
# (S3DIS Area 5 = fold5) and outdoor (DALES), for both SPT-semantic and
# SuperCluster-panoptic, plus the BIM-appropriate "with stuff" panoptic
# variant where wall/floor/ceiling are stuff classes. Each is 2-13 MB and
# lives on Zenodo (immutable records), so baking is deterministic and
# keeps the image self-contained / offline-smoke-testable.
RUN mkdir -p /opt/superpoint_transformer/checkpoints && \
    cd /opt/superpoint_transformer/checkpoints && \
    wget -q --tries=3 -O spt-2_dales.ckpt \
      https://zenodo.org/api/records/8042712/files/spt-2_dales.ckpt/content && \
    wget -q --tries=3 -O spt-2_s3dis_fold5.ckpt \
      https://zenodo.org/api/records/8042712/files/spt-2_s3dis_fold5.ckpt/content && \
    wget -q --tries=3 -O supercluster_dales.ckpt \
      https://zenodo.org/api/records/10689038/files/supercluster_dales.ckpt/content && \
    wget -q --tries=3 -O supercluster_s3dis_fold5.ckpt \
      https://zenodo.org/api/records/10689038/files/supercluster_s3dis_fold5.ckpt/content && \
    wget -q --tries=3 -O supercluster_s3dis_with_stuff_fold5.ckpt \
      https://zenodo.org/api/records/10689038/files/supercluster_s3dis_with_stuff_fold5.ckpt/content && \
    ls -lh

WORKDIR /opt/superpoint_transformer

# Build-time smoke test: print versions via importlib.metadata (per the
# ml-containers rule), load the FRNN .so (validates compile + link),
# reach the SPT model class, and confirm both task families (semantic +
# panoptic configs) and the baked checkpoints are present. Cannot launch
# an sm_90 kernel here — that is the runtime test's job (see README).
RUN python -c "import os, importlib.metadata as im; \
import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda); \
print('torch_geometric', im.version('torch_geometric')); \
print('pytorch_lightning', im.version('pytorch-lightning')); \
print('torchmetrics', im.version('torchmetrics')); \
print('pgeof', im.version('pgeof')); \
print('numpy', im.version('numpy')); \
import frnn; print('frnn loaded:', os.path.basename(os.path.dirname(frnn.__file__))); \
from src.models.semantic import SemanticSegmentationModule; print('reached SemanticSegmentationModule'); \
assert os.path.isdir('configs/experiment/semantic'), 'missing semantic experiment configs'; \
assert os.path.isdir('configs/experiment/panoptic'), 'missing panoptic experiment configs'; \
print('semantic + panoptic experiment configs present'); \
ck = sorted(os.listdir('checkpoints')); print('baked checkpoints:', ck); \
assert any('dales' in c for c in ck) and any('s3dis' in c for c in ck), 'expected DALES + S3DIS checkpoints'"

# Verify the FRNN CUDA extension actually carries sm_90 cubins — the
# build-time guard for the no-GPU-at-build risk that `import frnn` alone
# cannot catch. cuobjdump ships in the CUDA -devel base. Hard-fail only
# when cuobjdump reports arches but sm_90 is absent (the real failure
# mode); a tooling/path miss warns rather than blocking the build, since
# the H100 runtime test is the definitive arch check.
RUN FRNN_SO=$(python -c "import glob,os,frnn; c=glob.glob(os.path.join(os.path.dirname(frnn.__file__),'_C*.so')); print(c[0] if c else '')") && \
    if [ -n "$FRNN_SO" ] && command -v cuobjdump >/dev/null 2>&1; then \
      ARCHES=$(cuobjdump "$FRNN_SO" 2>/dev/null | grep -ioE "sm_[0-9]+" | sort -u | tr '\n' ' '); \
      echo "FRNN ext: $FRNN_SO | arches: ${ARCHES:-none-detected}"; \
      if echo "$ARCHES" | grep -q "sm_90"; then echo "FRNN: sm_90 cubins present"; \
      elif [ -n "$ARCHES" ]; then echo "ERROR: FRNN built without sm_90 (has: $ARCHES)" && exit 1; \
      else echo "WARN: cuobjdump detected no arches; H100 runtime test must confirm sm_90"; fi; \
    else echo "WARN: FRNN .so or cuobjdump not found; skipping build-time arch check"; fi

LABEL org.opencontainers.image.title="superpoint-transformer" \
      org.opencontainers.image.description="Superpoint Transformer (SPT, semantic) + SuperCluster (panoptic) point-cloud segmentation, H100 sm_90, DALES + S3DIS checkpoints baked" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/superpoint-transformer/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/drprojects/superpoint_transformer" \
      bradleylab.model.paper="https://arxiv.org/abs/2306.08045" \
      bradleylab.model.task="point cloud semantic + panoptic segmentation" \
      bradleylab.model.sensor="lidar:terrestrial/aerial" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="zenodo:8042712 (SPT) + zenodo:10689038 (SuperCluster), baked at build"

# No turnkey pipeline script upstream — inference is `python src/eval.py
# experiment=<semantic|panoptic>/<dataset> ckpt_path=checkpoints/<file>`.
# See README. Default to an interactive shell.
CMD ["bash"]
