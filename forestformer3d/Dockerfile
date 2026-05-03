# ForestFormer3D for H100 — Xiang et al. ICCV 2025 Oral
# arXiv:2506.16991 · https://github.com/SmartForest-no/ForestFormer3D
#
# Transformer-panoptic 3D forest instance segmentation built on
# OneFormer3D, fine-tuned on FOR-instanceV2 (extends FOR-instance with
# TU_WIEN deciduous alluvial leaf-off + BlueCat broadleaf temperate).
# Replaces PointGroup-style clustering with learned instance queries —
# no post-hoc clustering parameters to tune.
#
# Per-site F1 vs SegmentAnyTree on shared deciduous test sites
# (from the FF3D paper):
#   TU_WIEN  76.7%  (vs SAT 50.1%)
#   Wytham   75.0%  (vs SAT 61.4%)
#   BlueCat  61.7%  (no SAT comparable)
#
# Calibration caveat for Tyson UAV oak-hickory leaf-on (89 pts/m²):
# our density is below the FOR-instanceV2 training distribution and
# the paper's stable-performance floor (~75 pts/m²). Realistic F1
# floor is 60-70%, not 76%.
#
# ## Build path: Plan A (cu121 + torch 2.2.2 + CiSong10 MinkowskiEngine)
#
# Plan B (the original `pytorch:1.13.1-cuda11.6-cudnn8` + cu118 nvcc
# variant, faithfully mirroring upstream pins) failed in CI (run
# 25285693937, 2026-05-03) at the MinkowskiEngine compile step:
# `ValueError: Unknown CUDA arch (8.9) or GPU not supported`. PyTorch
# 1.13.1 predates Hopper / Ada by two years and its
# `_get_cuda_arch_flags()` rejects sm_89 outright, so 1.13.1 cannot
# emit native sm_90 kernels regardless of the underlying nvcc. We
# moved to Plan A, which is the same toolchain proven on
# `segment-any-tree-h100/Dockerfile` (also H100, also MinkowskiEngine).
#
# Stack:
#   Base                pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
#   PyTorch             2.2.2 (in base)
#   mmengine            0.10.3   (mmdet3d 1.4.0 accepts >=0.8.0,<1.0.0)
#   mmcv                2.1.0    (mmdet3d 1.4.0 accepts >=2.0.0rc4,<2.2.0;
#                                 cu121/torch2.1 prebuilt wheel —
#                                 ABI-compat with torch 2.2.x)
#   mmdet               3.3.0    (in [3.0.0rc5, 3.4.0))
#   mmsegmentation      1.2.2
#   mmdet3d             1.4.0    (released wheel)
#   MinkowskiEngine     CiSong10/cuda12-installation (proven on SAT v2)
#   spconv-cu121        2.3.8    (latest 2.3.x, cu121 wheels ship sm_90)
#   cumm-cu121          0.7.11   (matched to spconv-cu121 2.3.8 deps)
#   PyG ecosystem       torch-2.2.2+cu121 prebuilt wheels
#
# ## replace_mmdetection_files/ overlay
#
# FF3D upstream ships 3 full-file replacements (3,515 lines total)
# against `mmengine 0.7.3` + `mmdet3d @ 22aaa47`:
#   - mmengine/runner/loops.py            (466 L)
#   - mmengine/model/base_model/base_model.py  (348 L)
#   - mmdet3d/datasets/transforms/transforms_3d.py  (2,701 L)
#
# These were written against the older mm-stack. On Plan A we install
# mmengine 0.10.x and mmdet3d 1.4.0 from released wheels first, then
# COPY FF3D's frozen versions over the site-packages locations. The
# research-agent assessment in
# `tyson-forest-linkage/docs/METHOD_COMPARISON.md` §4.3 was "API stable,
# signature drift possible, not a hard blocker". The smoke test at the
# end of this Dockerfile imports each package after the overlay, so
# any signature drift surfaces at build time. If the smoke test fails
# on a renamed/relocated symbol in the FF3D overlay files, the next
# step is to forward-port FF3D's surgical customizations against the
# new mmengine 0.10.x / mmdet3d 1.4.0 sources — the deltas vs upstream
# are typically 50-200 lines against the FF3D copies (see git diff
# upstream main vs 0.10.x tags for mmengine and v1.4.0 tag for
# mmdet3d).

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
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# System deps mirror SAT v2 (also H100 + MinkowskiEngine + cu121).
# build-essential / cmake / ninja: MinkowskiEngine + segmentator csrc
# libopenblas-dev: MinkowskiEngine BLAS=openblas
# libsparsehash-dev: torch-points3d kdtree headers
# libgl1 / libglib2.0-0 / libglu1-mesa-dev / libosmesa6-dev: open3d
#   off-screen rendering at runtime
# ffmpeg / libsm6 / libxrender-dev / libxext6: open3d / matplotlib
# libsndfile1 / libusb-1.0-0-dev / libxcb-shm0: open3d transitive
# libtbb-dev: parallel STL paths used by some sparse-conv backends
# git-lfs: model-checkpoint helpers in upstream FF3D
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ninja-build cmake build-essential libopenblas-dev \
        libsparsehash-dev libtbb-dev libglib2.0-0 libglu1-mesa-dev \
        libosmesa6-dev libsm6 libusb-1.0-0-dev libxcb-shm0 libxext6 \
        libxrender-dev libsndfile1 libgl1 ffmpeg \
        wget unzip zip git-lfs ca-certificates \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Patch std::to_address collision with CUDA 12 + g++11 (NVIDIA #596).
# Same fix SAT v2 uses; required before MinkowskiEngine compiles.
RUN sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' \
    /usr/include/c++/11/bits/shared_ptr_base.h || true

RUN pip install 'setuptools<70' wheel ninja cython

# OpenMMLab stack — Plan A pins.
#
# Order matters:
#   1. Install mmengine + mmdet + mmsegmentation pure-Python wheels
#      (--no-deps to keep our pinned numpy/scipy stable).
#   2. Install mmcv 2.1.0 from the cu121/torch2.1 prebuilt index. The
#      cu121/torch2.2 index only ships mmcv 2.2.0, which mmdet3d 1.4.0
#      strictly forbids (`mmcv < 2.2.0` assertion in mmdet3d/__init__.py).
#      The torch-2.1 wheel is C++-ABI-compatible with torch 2.2 in the
#      2.x series.
#   3. Install mmdet3d 1.4.0 from PyPI wheel (no compile, pure Python).
#
# We DO NOT use --no-build-isolation here because everything is a
# pure-Python wheel install — there is no extension to compile.
RUN pip install --no-deps \
        mmengine==0.10.3 \
        mmdet==3.3.0 \
        mmsegmentation==1.2.2 \
    && pip install mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html \
        --no-deps \
    && pip install --no-deps mmdet3d==1.4.0

# MinkowskiEngine from CiSong10's CUDA 12 fork — same recipe SAT v2 uses.
# Native sm_90 kernels (PTX-JIT has hung on sparse convs in this
# container family before; we don't repeat that mistake).
RUN git clone https://github.com/CiSong10/MinkowskiEngine.git /tmp/me \
    && cd /tmp/me \
    && git checkout cuda12-installation \
    && python setup.py install --force_cuda --blas=openblas \
    && cd / && rm -rf /tmp/me

# spconv + cumm — cu121 wheels (sm_90 capable).
# spconv-cu121 2.3.8 requires cumm-cu121>=0.7.11,<0.8.0.
RUN pip install --no-deps \
        spconv-cu121==2.3.8 \
        cumm-cu121==0.7.11

# PyG ecosystem — torch-2.2.2+cu121 prebuilt wheels (matching SAT v2).
# torch-scatter / torch-cluster ship as wheels; torch-points-kernels
# does not, so we install it from PyPI (it builds against the host
# torch headers, so we keep the cython/ninja already installed above).
RUN pip install --no-cache-dir \
        torch-scatter torch-cluster \
        -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

# torch-points-kernels: SAT v2 patches its setup.py for numpy/cython
# pins and CUDA arch + at:: namespace fixes. We mirror that exactly.
RUN pip install --no-cache-dir "numpy>=1.24,<2.0" "cython<3.0" \
    && git clone https://github.com/torch-points3d/torch-points-kernels.git /tmp/tpk \
    && cd /tmp/tpk \
    && sed -i 's/"numpy<=1.21"/"numpy"/g' setup.py \
    && sed -i 's/-arch=sm_35/-arch=sm_50/g' setup.py \
    && sed -i 's/isIntegralType(/at::isIntegralType(/g' cuda/include/utils.h \
    && sed -i 's/isFloatingType(/at::isFloatingType(/g' cuda/include/utils.h \
    && pip install --no-cache-dir --no-build-isolation . \
    && cd / && rm -rf /tmp/tpk

# ScanNet superpoint segmentator (Karbo123). Upstream FF3D installs
# this from csrc; pin the same commit.
#
# Notes on adapting the upstream cmake invocation to torch 2.2:
#   1. `torch.utils.cmake_prefix_path` is a *string attribute* in
#      torch 2.x — not a callable. Upstream FF3D's `print(torch.utils.cmake_prefix_path())`
#      worked on torch 1.13.x (where it was a function) and raises
#      TypeError on 2.x. Drop the parens.
#   2. The python -c args use single-quote outer + double-quote inner
#      (no backslash escapes), which survives Dockerfile RUN parsing
#      cleanly and avoids the `\"` line-continuation gotcha (CI run
#      25286124144).
#   3. CMAKE_CXX_STANDARD bumped 14 -> 17. Torch 2.x C++ headers
#      (cloneable.h, conv.h) need C++17 to template-instantiate cleanly
#      under g++11 (CI run 25286954798).
#   4. Install location: segmentator's CMakeLists creates a SYMLINK at
#      $CMAKE_INSTALL_PREFIX/segmentator pointing into the source tree
#      (csrc/). If we delete the source tree after install, the symlink
#      target evaporates and `import segmentator.csrc.build.libsegmentator`
#      fails with ModuleNotFoundError (CI run 25287773931). Keep the
#      source tree under /opt/segmentator-src (a stable, non-tmp path)
#      so the in-place compiled .so survives container runtime.
RUN git clone https://github.com/Karbo123/segmentator.git /opt/segmentator-src \
    && cd /opt/segmentator-src/csrc \
    && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
    && sed -i 's/CMAKE_CXX_STANDARD 14/CMAKE_CXX_STANDARD 17/' CMakeLists.txt \
    && CMAKE_PREFIX=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
    && PY_INC=$(python -c 'from sysconfig import get_path; print(get_path("include"))') \
    && PY_LIB=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") + "/libpython3.10.so")') \
    && PY_PURELIB=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && mkdir -p build && cd build \
    && cmake .. \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX" \
        -DPYTHON_INCLUDE_DIR="$PY_INC" \
        -DPYTHON_LIBRARY="$PY_LIB" \
        -DCMAKE_INSTALL_PREFIX="$PY_PURELIB" \
    && make -j"$(nproc)" \
    && make install

# Remaining Python deps. Most of upstream FF3D's pinned versions are
# pure-Python and forward-compatible with the cu121/torch-2.2 stack,
# but a few of the older pins (numpy 1.24.1, scipy 1.10.1, scikit-learn
# 1.2.2) are kept because mmdet3d 1.4.0 / FF3D's data pipelines were
# tested against that line. We drop spconv/cumm here because they're
# already installed above with the cu121 variant.
RUN pip install --no-deps \
        addict==2.4.0 \
        yapf==0.33.0 \
        termcolor==2.3.0 \
        packaging==23.1 \
        rich==13.3.5 \
        opencv-python==4.7.0.72 \
        pycocotools==2.0.6 \
        Shapely==1.8.5 \
        scipy==1.10.1 \
        scikit-image==0.21.0 \
        prettytable==3.10.0 \
        terminaltables==3.1.10 \
        numba==0.57.0 \
        llvmlite==0.40.0 \
        pccm==0.4.16 \
        ccimport==0.4.4 \
        pybind11==2.10.4 \
        lark==1.1.5 \
        pyquaternion==0.9.9 \
        lyft-dataset-sdk==0.0.8 \
        pandas==2.0.1 \
        python-dateutil==2.8.2 \
        matplotlib==3.5.2 \
        pyparsing==3.0.9 \
        cycler==0.11.0 \
        kiwisolver==1.4.4 \
        scikit-learn==1.2.2 \
        joblib==1.2.0 \
        threadpoolctl==3.1.0 \
        cachetools==5.3.0 \
        nuscenes-devkit==1.1.10 \
        trimesh==3.21.6 \
        open3d==0.17.0 \
        plotly==5.18.0 \
        dash==2.14.2 \
        plyfile==1.0.2 \
        flask==3.0.0 \
        werkzeug==3.0.1 \
        click==8.1.7 \
        blinker==1.7.0 \
        itsdangerous==2.1.2 \
        importlib_metadata==2.1.2 \
        zipp==3.17.0 \
        tensorboard==2.15.1 \
        tensorboard-data-server==0.7.2 \
        protobuf \
        absl-py \
        future \
        MarkupSafe==2.0.1 \
        markdown \
        grpcio \
        google-auth-oauthlib \
        google-auth \
        requests-oauthlib \
        oauthlib \
        Pillow \
        PyYAML \
        typing-extensions \
        filelock \
        fsspec \
        regex \
        debugpy \
        "laspy[lazrs]"

# Clone FF3D into /workspace (upstream PYTHONPATH default).
RUN git clone https://github.com/SmartForest-no/ForestFormer3D.git /workspace
WORKDIR /workspace

ENV PYTHONPATH=/workspace

# Apply the 3,515-line replace_mmdetection_files/ overlay. These are
# full-file replacements that upstream README §4 documents as a manual
# `cp` step; we bake it in. The destination paths are derived at build
# time from the actually-installed mmengine 0.10.x and mmdet3d 1.4.0
# package locations, so this works regardless of the python prefix.
#
# WARNING: FF3D's overlay was authored against mmengine 0.7.3 and
# mmdet3d @ 22aaa47. On Plan A we have mmengine 0.10.x + mmdet3d 1.4.0,
# so signature drift in `loops.py`, `base_model.py`, or
# `transforms_3d.py` is possible. The smoke test below imports each
# subsystem; if it fails on an AttributeError or TypeError after the
# overlay, the fix is to forward-port the FF3D customizations against
# the new mmengine/mmdet3d source rather than blindly overwriting.
RUN MMENGINE_PATH=$(python -c "import mmengine, os; print(os.path.dirname(mmengine.__file__))") \
    && MMDET3D_PATH=$(python -c "import mmdet3d, os; print(os.path.dirname(mmdet3d.__file__))") \
    && echo "mmengine at $MMENGINE_PATH" \
    && echo "mmdet3d  at $MMDET3D_PATH" \
    && cp /workspace/replace_mmdetection_files/loops.py        "$MMENGINE_PATH/runner/loops.py" \
    && cp /workspace/replace_mmdetection_files/base_model.py   "$MMENGINE_PATH/model/base_model/base_model.py" \
    && cp /workspace/replace_mmdetection_files/transforms_3d.py "$MMDET3D_PATH/datasets/transforms/transforms_3d.py"

# Pretrained weights are NOT baked.
#
# Rationale: epoch_3000_fix.pth ships inside clean_forestformer.zip on
# Zenodo record 16742708 (~198 MB). Zenodo has been observed throttling
# or returning 5xx during high-traffic windows; coupling the image
# build to that endpoint risks transient CI failures. Weights download
# once on the target host (Compute2 scratch) into a bind-mounted dir
# and are reused across SLURM jobs.
COPY download_weights.sh /workspace/download_weights.sh
RUN chmod +x /workspace/download_weights.sh \
    && mkdir -p /workspace/work_dirs/clean_forestformer \
                /workspace/data/ForAINetV2/test_data \
                /workspace/data/ForAINetV2/train_val_data

# Optional thin inference wrapper for runtime.
COPY scripts/run_inference.sh /workspace/scripts/run_inference.sh
RUN chmod +x /workspace/scripts/run_inference.sh

# Smoke test: confirm every load-bearing dep imports + reports its
# version via importlib.metadata (per ml-containers rule —
# pkg.__version__ is convention, not contract).
RUN python -c "\
import importlib, importlib.metadata as im, sys; \
print('python', sys.version.split()[0]); \
import torch; print('torch', torch.__version__, 'cuda (build):', torch.version.cuda); \
print('mmengine        ', im.version('mmengine')); \
print('mmcv            ', im.version('mmcv')); \
print('mmdet           ', im.version('mmdet')); \
print('mmsegmentation  ', im.version('mmsegmentation')); \
print('mmdet3d         ', im.version('mmdet3d')); \
print('spconv-cu121    ', im.version('spconv-cu121')); \
print('cumm-cu121      ', im.version('cumm-cu121')); \
print('numpy           ', im.version('numpy')); \
print('open3d          ', im.version('open3d')); \
print('laspy           ', im.version('laspy')); \
import MinkowskiEngine as ME; \
print('MinkowskiEngine ', im.version('MinkowskiEngine')); \
import segmentator; print('segmentator     ok (csrc-installed, no PEP-621 metadata)'); \
import torch_scatter; print('torch_scatter   ', im.version('torch-scatter')); \
import torch_cluster; print('torch_cluster   ', im.version('torch-cluster')); \
from torch_points_kernels import instance_iou; print('torch-points-kernels', im.version('torch-points-kernels')); \
import oneformer3d; print('oneformer3d (FF3D in-tree) loaded from', oneformer3d.__file__); \
print('SMOKE TEST PASSED')"

LABEL org.opencontainers.image.title="forestformer3d" \
      org.opencontainers.image.description="ForestFormer3D (Xiang et al. ICCV 2025 Oral) transformer-panoptic 3D forest instance segmentation; H100 sm_90 via cu121 + torch 2.2.2 + CiSong10 MinkowskiEngine; Plan A build" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/forestformer3d/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="CC-BY-NC-4.0" \
      bradleylab.model.upstream="https://github.com/SmartForest-no/ForestFormer3D" \
      bradleylab.model.paper="https://arxiv.org/abs/2506.16991" \
      bradleylab.model.weights.source="zenodo:16742708 (clean_forestformer.zip -> epoch_3000_fix.pth, runtime fetch)" \
      bradleylab.model.task="panoptic instance segmentation, point cloud" \
      bradleylab.model.sensor="lidar:airborne" \
      bradleylab.model.status="experimental" \
      bradleylab.model.notes="CC BY-NC 4.0 inherited from OneFormer3D base; academic use OK, commercial requires upstream permission"

CMD ["bash"]
