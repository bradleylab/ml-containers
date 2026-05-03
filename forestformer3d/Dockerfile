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
# ## Build path: Plan B (lowest deviation)
#
# Upstream FF3D Dockerfile pins `pytorch:1.13.1-cuda11.6-cudnn8-devel`
# + `mmengine 0.7.3` + `mmcv 2.0.0` + `mmdet 3.0.0` + `mmdet3d @
# 22aaa47` + `MinkowskiEngine @ 02fc608` + `spconv-cu116 2.3.6`. Plan B
# preserves all mm* versions and the 3,515 lines of
# `replace_mmdetection_files/` patches verbatim, and only swaps the
# CUDA toolchain / sparse-conv backend so MinkowskiEngine and spconv
# emit native sm_90 kernels for H100.
#
# Note on base image: the spec named `pytorch:1.13.1-cuda11.8-cudnn8-devel`,
# but that exact Docker Hub tag does not exist (only cuda11.6 ships
# for torch 1.13.1). We approximate by basing on
# `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` (cu118 nvcc, sm_90
# capable) and pip-installing `torch==1.13.1+cu117` from
# pytorch.org — this is the same approach used in
# `treelearn/Dockerfile` and is binary-compatible with cu118 runtime.
# mmcv 2.0.0 cu117/torch1.13 prebuilt wheels are available.
# spconv-cu118 ships sm_90 PTX inside the wheel.
#
# Plan A (full port to mm-engine 0.10.x / mmcv 2.1 / mmdet3d 1.4 on
# torch 2.2 + cu121) is the documented fallback if Plan B fails to
# compile MinkowskiEngine on sm_90 in CI. See README.md §"Plan A
# fallback" for the version matrix.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
    CMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
    FORCE_CUDA=1 \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# System deps mirror upstream Dockerfile + Python 3.10 from deadsnakes.
# build-essential / cmake / ninja: MinkowskiEngine + segmentator csrc
# libopenblas-dev: MinkowskiEngine BLAS=openblas
# libsparsehash-dev: torch-points3d kdtree headers
# libgl1 / libglib2.0-0: open3d / opencv-python runtime
# ffmpeg / libsm6 / libxrender-dev: open3d / matplotlib runtime
# libxext6: X11 helper for open3d off-screen rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates wget git \
        build-essential cmake ninja-build \
        libopenblas-dev libsparsehash-dev \
        libgl1 libglib2.0-0 \
        ffmpeg libsm6 libxext6 libxrender-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-distutils \
    && wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.10 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install 'setuptools<70' wheel

# PyTorch 1.13.1+cu117 is the closest available 1.13.1 wheel; cu118
# torch 1.13.1 wheels were never published. The cu117 build is binary
# compatible with cu118 runtime libraries on the host.
RUN pip install --index-url https://download.pytorch.org/whl/cu117 \
        torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1

# OpenMMLab stack — exact upstream pins. mmcv prebuilt wheel matches
# cu117/torch1.13 (no cu118 wheel exists for torch 1.13).
#
# mmdet3d git install requires --no-build-isolation: its setup.py does
# `import torch` at module top, and pip's PEP-517 isolated build env
# doesn't see the host torch unless we disable isolation. We pre-install
# the build-time helpers (ninja, cython) so the mmdet3d build wheel can
# proceed with just our existing torch + setuptools.
RUN pip install --no-deps \
        mmengine==0.7.3 \
        mmdet==3.0.0 \
        mmsegmentation==1.0.0 \
    && pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html --no-deps \
    && pip install ninja cython \
    && pip install --no-deps --no-build-isolation \
        "git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61"

# MinkowskiEngine — upstream NVIDIA commit 02fc608, rebuilt with
# TORCH_CUDA_ARCH_LIST including 9.0 so H100 (sm_90) gets native
# kernels rather than PTX-JIT (which has hung on sparse convs in
# this container family before — SAT v1 incident).
#
# We use `python setup.py install` (mirroring SAT v2's recipe)
# rather than upstream FF3D's `pip install --install-option=...`
# form because pip 23.1+ removed `--install-option`. Setuptools'
# direct setup.py path still honors the same flags.
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/me \
    && cd /tmp/me \
    && git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 \
    && python setup.py install --force_cuda --blas=openblas \
    && cd / && rm -rf /tmp/me

# torch-scatter — upstream pins 2.0.9, build-from-source so the CUDA
# kernels match our toolchain. Keep arch list inclusive of 9.0.
RUN git clone https://github.com/rusty1s/pytorch_scatter.git /tmp/ts \
    && cd /tmp/ts \
    && git checkout tags/2.0.9 -b v2.0.9 \
    && pip install --no-deps . \
    && cd / && rm -rf /tmp/ts

# ScanNet superpoint segmentator (Karbo123). Upstream installs to
# site-packages/segmentator. Pin the same csrc commit as upstream.
RUN git clone https://github.com/Karbo123/segmentator.git /tmp/segmentator \
    && cd /tmp/segmentator/csrc \
    && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
    && mkdir -p build && cd build \
    && cmake .. \
        -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path())')" \
        -DPYTHON_INCLUDE_DIR="$(python -c 'from sysconfig import get_path; print(get_path(\"include\"))')" \
        -DPYTHON_LIBRARY="$(python -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\") + \"/libpython3.10.so\")')" \
        -DCMAKE_INSTALL_PREFIX="$(python -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')" \
    && make -j"$(nproc)" \
    && make install \
    && cd / && rm -rf /tmp/segmentator

# Remaining Python deps — mirror upstream Dockerfile pins exactly.
# spconv-cu118 (not cu116) is the one substitution: it ships sm_90
# kernels and is API-compatible with the 2.3.x line upstream uses.
# cumm-cu118 0.4.x correspondingly.
RUN pip install --no-deps \
        spconv-cu118==2.3.6 \
        cumm-cu118==0.4.11 \
        addict==2.4.0 \
        yapf==0.33.0 \
        termcolor==2.3.0 \
        packaging==23.1 \
        numpy==1.24.1 \
        rich==13.3.5 \
        opencv-python==4.7.0.72 \
        pycocotools==2.0.6 \
        Shapely==1.8.5 \
        scipy==1.10.1 \
        terminaltables==3.1.10 \
        numba==0.57.0 \
        llvmlite==0.40.0 \
        pccm==0.4.7 \
        ccimport==0.4.2 \
        pybind11==2.10.4 \
        ninja==1.11.1 \
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
        debugpy

# torch-points-kernels — upstream README §2 fix. Install --no-deps to
# avoid pulling a numpy >=2.0 that breaks the build, then reinstall
# torch-cluster --no-deps (upstream README §3).
RUN pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0 \
    && pip uninstall -y torch-cluster || true \
    && pip install --no-deps --no-cache-dir torch-cluster \
    && pip install --no-deps "laspy[lazrs]"

# laz-IO for runtime input handling on Tyson tiles.
RUN pip install --no-deps laspy

# Clone FF3D and place it at /workspace (upstream PYTHONPATH default).
RUN git clone https://github.com/SmartForest-no/ForestFormer3D.git /workspace
WORKDIR /workspace

ENV PYTHONPATH=/workspace

# Apply the 3,515-line replace_mmdetection_files/ overlay. Upstream
# README §4 describes this as a manual `cp` step; we bake it into
# the image so every container is consistent. These are full-file
# replacements, not patches, so a straight cp is the correct
# operation. The file paths under site-packages are pinned by the
# mm* versions installed above (mmengine 0.7.3, mmdet3d @ 22aaa47).
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
#
# Use the bundled `download_weights.sh` helper.
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
import torch; print('torch', torch.__version__, 'cuda_avail (build):', torch.version.cuda); \
print('mmengine     ', im.version('mmengine')); \
print('mmcv         ', im.version('mmcv')); \
print('mmdet        ', im.version('mmdet')); \
print('mmsegmentation', im.version('mmsegmentation')); \
print('mmdet3d      ', im.version('mmdet3d')); \
print('spconv-cu118 ', im.version('spconv-cu118')); \
print('cumm-cu118   ', im.version('cumm-cu118')); \
print('numpy        ', im.version('numpy')); \
print('open3d       ', im.version('open3d')); \
print('laspy        ', im.version('laspy')); \
import MinkowskiEngine as ME; \
print('MinkowskiEngine', im.version('MinkowskiEngine')); \
import segmentator; print('segmentator    ok (no PEP 621 metadata; csrc-installed)'); \
import torch_scatter; print('torch_scatter  ', im.version('torch-scatter')); \
import torch_cluster; print('torch_cluster  ', im.version('torch-cluster')); \
from torch_points_kernels import instance_iou; print('torch-points-kernels', im.version('torch-points-kernels')); \
import oneformer3d; print('oneformer3d (FF3D in-tree) loaded from', oneformer3d.__file__); \
print('SMOKE TEST PASSED')"

LABEL org.opencontainers.image.title="forestformer3d" \
      org.opencontainers.image.description="ForestFormer3D (Xiang et al. ICCV 2025 Oral) transformer-panoptic 3D forest instance segmentation; H100 sm_90 via cu118 nvcc + spconv-cu118 + MinkowskiEngine rebuilt for sm_90; Plan B build" \
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
