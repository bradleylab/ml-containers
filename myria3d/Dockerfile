# myria3d + FRACTAL-LidarHD_7cl_randlanet — airborne lidar semantic segmentation
#
# RandLA-Net trained by IGN on FRACTAL (French Lidar HD). Seven classes:
# other | ground | vegetation | building | water | bridge | permanent
# structure. This is the lab's first general-purpose ALS semantic
# segmenter — every other point-cloud image here is forestry-specific
# (tree instances, leaf/wood, crowns).
#
# Upstream code:  https://github.com/IGNF/myria3d          (BSD-3-Clause)
# Weights:        https://huggingface.co/IGNF/FRACTAL-LidarHD_7cl_randlanet
#                 (Etalab Open Licence 2.0, ungated, 13 MB — NOT baked, see below)
# Stack:          PyTorch 2.4.1 + CUDA 12.4 + PyTorch-Geometric + PDAL 2.10
#
# ## Base image: conda-forge, not NGC and not nvidia/cuda
#
# myria3d imports the PDAL *Python bindings* and GDAL's `osgeo.osr` at
# module load — `myria3d/pctl/dataset/utils.py` does `import pdal`,
# `myria3d/models/interpolation.py` does `import pdal` plus
# `from pdaltools import las_info`, and ign-pdal-tools' `las_info` does
# `import osgeo.osr`. Neither PDAL nor GDAL publishes a manylinux wheel:
# the PyPI `pdal` package is bindings only and needs libpdalcpp headers at
# build time. Upstream therefore ships a conda environment, and that is
# the supported install path. A conda-forge base is the one that gets
# PDAL 2.10 + python-pdal + GDAL into the same interpreter as torch
# without compiling PDAL from source.
#
# The two bases the rest of this repo reaches for do not fit here:
#   * NGC `nvcr.io/nvidia/pytorch:*` carries a global pip constraint
#     (`PIP_CONSTRAINT=/etc/pip/constraint.txt`) pinning numpy==1.26.4 and
#     a torch build we cannot match against the PyG companion wheels —
#     see esm/README.md for how that constraint broke an install outright.
#   * `nvidia/cuda:*` would still need the whole conda layer bolted on for
#     PDAL/GDAL, i.e. a second CUDA userspace stack for no benefit.
#
# Nothing in this image compiles a CUDA extension — torch and the three
# PyG companions all install as prebuilt wheels — so no nvcc, no CUDA
# headers, and no -devel base are needed. The cu124 torch wheels carry
# their own CUDA runtime; the host driver is injected by enroot/pyxis (or
# `docker run --gpus`) at run time.
#
# ## The three-way pin (torch / CUDA / PyG companions)
#
# torch-scatter, torch-cluster and torch-sparse are compiled against one
# exact (torch minor, CUDA minor, CPython) triple. Every version below was
# checked against the published wheel index before being written down:
#
#   torch          2.4.1+cu124   cp312   download.pytorch.org/whl/cu124
#   torch_scatter  2.1.2+pt24cu124 cp312 data.pyg.org/whl/torch-2.4.0+cu124
#   torch_cluster  1.6.3+pt24cu124 cp312 data.pyg.org/whl/torch-2.4.0+cu124
#   torch_sparse   0.6.18+pt24cu124 cp312 data.pyg.org/whl/torch-2.4.0+cu124
#
# PyG publishes one wheel index per torch *minor* (2.4.0), not per patch,
# so the 2.4.0 wheels are what a torch 2.4.1 install uses — same ABI
# within the minor. This is the same arrangement superpoint-transformer
# uses (torch 2.2.2 against the 2.2.0 index).
#
# torch 2.4.1 is upstream's own pin, and staying on it is deliberate
# rather than incidental: torch 2.6 flipped `torch.load` to
# `weights_only=True` by default, which is exactly the change that breaks
# loading Lightning checkpoints carrying non-tensor hyperparameters. The
# FRACTAL checkpoint is such a checkpoint. Do not bump torch past 2.5
# without testing a real `Model.load_from_checkpoint` first.
#
# torch_cluster is not optional despite nothing importing it by name: the
# inference config uses `torch_geometric.transforms.GridSampling`, which
# reaches torch_cluster's `grid_cluster`, and the model's `knn_graph`
# reaches its `knn`. torch_scatter is imported directly by
# `myria3d/models/interpolation.py`. torch_sparse is not imported anywhere
# in myria3d, but upstream's environment.yml installs it and the wheel is
# prebuilt, so it is kept for parity rather than trimmed.
#
# numpy is held at 1.26.4 — the line torch 2.4.1 and the pt24 companion
# wheels were built and tested against. conda-forge's current python-pdal
# and gdal builds declare `numpy >=1.25,<3`, so they run against 1.26
# even though they were compiled with numpy-2 headers. If the conda solve
# ever fails, relaxing this pin is the first thing to try (torch 2.4 added
# numpy-2 support), but do it deliberately and re-run the smoke test.

FROM condaforge/miniforge3:26.1.1-3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONNOUSERSITE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE is set here as well as in the job script because enroot
# bind-mounts $HOME into the container on Compute2; without it, a stray
# `pip install --user` on the login node shadows the image's site-packages.
# NVIDIA_* are the vars the container runtime's GPU hook reads; pyxis
# overrides them per allocation, so these only matter for a plain
# `docker run --gpus all`.

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Geo stack from conda-forge only (no `anaconda` channel: upstream's
# environment.yml pulls mkl from it, which carries Anaconda's commercial
# terms — not something an institutional build should depend on).
# `pdal` brings the CLI, which myria3d shells out to for tile metadata
# (`get_pdal_info_metadata` in pctl/dataset/utils.py runs `pdal info`).
RUN conda create -y -p /opt/env --override-channels -c conda-forge \
        python=3.12 \
        pip \
        "pdal=2.10.*" \
        python-pdal \
        gdal \
        numpy=1.26.4 \
    && conda clean -afy

ENV PATH=/opt/env/bin:$PATH \
    PROJ_LIB=/opt/env/share/proj \
    GDAL_DATA=/opt/env/share/gdal

# Runtime callers should use the absolute interpreter path
# `/opt/env/bin/python`, not a bare `python`: this base carries its own
# conda `base` env, and a login shell (`bash -lc`, which every Compute2
# recipe here uses) can activate it and put /opt/conda/bin ahead of
# /opt/env/bin on PATH. Build-time RUN steps are unaffected — no profile
# scripts run there — so they use `python` directly.
#
# PROJ_LIB/GDAL_DATA are normally set by conda's activation scripts, which
# never run here (the env is on PATH, not activated). Upstream's Dockerfile
# sets PROJ_LIB for the same reason. LD_LIBRARY_PATH is deliberately NOT
# pointed at /opt/env/lib the way upstream does — upstream needs it because
# their torch is a conda package linked against conda's CUDA; ours is a pip
# wheel with its own bundled runtime, and prepending conda's lib dir would
# only invite a libstdc++ mismatch.

# torch first and alone, so nothing later resolves a different build as a
# transitive dependency.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1

# PyG companions — prebuilt wheels keyed to torch-2.4.0+cu124 / cp312.
# `==2.1.2` matches the `2.1.2+pt24cu124` local version (PEP 440), so the
# wheels on the index are what get installed; no source build happens.
RUN pip install -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
        torch-scatter==2.1.2 \
        torch-cluster==1.6.3 \
        torch-sparse==0.6.18

# Remaining Python deps, mirroring upstream's environment.yml minus the
# training-only and dev tooling (comet_ml, pre-commit, black, sphinx, ...).
# comet_ml is intentionally absent: run.py guards its import in a
# try/except, and the FRACTAL inference config's `logger.comet` block is
# never instantiated on the predict path.
# ign-pdal-tools declares no dependencies of its own, so it cannot drag a
# PyPI `pdal` on top of the conda bindings.
RUN pip install \
        numpy==1.26.4 \
        torch-geometric==2.6.1 \
        pytorch-lightning==2.4.0 \
        torchmetrics==1.4.3 \
        hydra-core==1.3.2 \
        hydra-colorlog==1.2.0 \
        python-dotenv==1.0.1 \
        rich==13.9.4 \
        tqdm==4.67.1 \
        h5py==3.12.1 \
        pandas==2.2.3 \
        scipy==1.14.1 \
        matplotlib==3.9.2 \
        seaborn==0.13.2 \
        "laspy[lazrs]==2.5.4" \
        ign-pdal-tools==1.16.0 \
        requests==2.32.3 \
        huggingface_hub==1.28.0

# myria3d source, pinned to the main commit this recipe was written
# against (v3.9.0, merge of the FRACTAL RandLA-Net staging branch).
ARG MYRIA3D_REF=fcc22ac32f04380d971f99d6827edc4f0e376e58
ENV MYRIA3D_REF=${MYRIA3D_REF}

RUN git clone https://github.com/IGNF/myria3d.git /opt/myria3d && \
    cd /opt/myria3d && \
    git checkout ${MYRIA3D_REF} && \
    rm -rf .git

# Installed with --no-deps so the pinned set above stands: upstream's
# pyproject declares `dependencies = []` and defers to environment.yml, but
# installing it registers package metadata, which is what the smoke test
# reads the version from. run.py and configs/ still run from the source
# tree at /opt/myria3d.
RUN pip install --no-deps /opt/myria3d

# Weights are NOT baked.
#
# Upstream commits the 13 MB FRACTAL checkpoint into the repo at
# trained_model_assets/, so the clone above brings it along; it is deleted
# here. Two reasons: the lab's rule that model weights are staged at run
# time rather than shipped in the image, and licence hygiene — the code is
# BSD-3-Clause while the checkpoint is Etalab Open Licence 2.0, and keeping
# the artifact out of the GHCR image keeps those separate.
#
# The in-repo copy and the Hugging Face copy are byte-identical (sha256
# 58baca3f...8cd1, verified 2026-08-18), so either source is fine; the
# README documents the HF route because it is versioned and citable.
# The inference YAML stays: it is configuration, it lives in the BSD-3
# repo, and hydra's config_path in run.py points at this directory.
RUN rm -f /opt/myria3d/trained_model_assets/*.ckpt && \
    ls -la /opt/myria3d/trained_model_assets/

ENV MYRIA3D_CKPT_SHA256=58baca3fbc00af2fa4af2a26cea345c08decbbba0215d21d6640c412a42e8cd1 \
    HF_HOME=/root/.cache/huggingface

# Build-time smoke test 1 — versions from package metadata (never
# pkg.__version__), the model class reached, and a real RandLA-Net built on
# CPU at the FRACTAL geometry (9 input features, 7 classes). Offline: no
# checkpoint, no HF call, no GPU.
RUN python -c "import importlib.metadata as im, os, torch; \
print('torch', torch.__version__, '| cuda', torch.version.cuda); \
print(' | '.join(f'{p} {im.version(p)}' for p in ( \
    'myria3d', 'torch_geometric', 'torch_scatter', 'torch_cluster', 'torch_sparse', \
    'pytorch-lightning', 'torchmetrics', 'hydra-core', 'numpy', 'laspy', \
    'ign-pdal-tools', 'huggingface-hub'))); \
assert im.version('myria3d') == '3.9.0', im.version('myria3d'); \
from myria3d.models.model import Model; \
from myria3d.models.modules.pyg_randla_net import PyGRandLANet; \
net = PyGRandLANet(num_features=9, num_classes=7, num_neighbors=16, decimation=4, return_logits=True); \
n = sum(p.numel() for p in net.parameters()); \
print('PyGRandLANet built on CPU:', n, 'parameters'); \
assert n > 0, n; \
cfg = '/opt/myria3d/trained_model_assets/FRACTAL-LidarHD_7cl_randlanet-inference-Myria3DV3.8.yaml'; \
assert os.path.isfile(cfg), cfg; \
assert not [f for f in os.listdir('/opt/myria3d/trained_model_assets') if f.endswith('.ckpt')], \
    'a checkpoint is baked into the image; it must be staged at run time'; \
print('inference config present, no baked checkpoint')"

# Build-time smoke test 2 — the geo half of the stack, which is the part a
# pip-only image cannot provide. Both the Python bindings and the CLI are
# exercised, plus the LAS/LAZ readers the predict path depends on.
RUN python -c "import pdal, osgeo.osr, laspy; \
from pdaltools import las_info; \
from myria3d.pctl.dataset.utils import get_pdal_reader, get_pdal_info_metadata; \
print('pdal + osgeo.osr + laspy + pdaltools import OK')" && \
    pdal --version && \
    pdal --drivers 2>/dev/null | grep -q readers.las && echo "PDAL readers.las OK"

# Build-time smoke test 3 — hydra composes the FRACTAL inference config
# end to end. This is upstream's own build check (their Dockerfile runs the
# same command) and it catches config/omegaconf breakage that imports miss.
WORKDIR /opt/myria3d
RUN python run.py task.task_name=predict --help > /dev/null && \
    echo "hydra predict config composes"

LABEL org.opencontainers.image.title="myria3d" \
      org.opencontainers.image.description="myria3d RandLA-Net (IGN) semantic segmentation of airborne lidar into 7 classes — ground/vegetation/building/water/bridge/permanent structure; FRACTAL Lidar HD checkpoint staged at run time" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/myria3d/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="BSD-3-Clause" \
      bradleylab.model.upstream="https://github.com/IGNF/myria3d" \
      bradleylab.model.paper="https://huggingface.co/IGNF/FRACTAL-LidarHD_7cl_randlanet" \
      bradleylab.model.task="point cloud semantic segmentation (7 classes)" \
      bradleylab.model.sensor="lidar:airborne" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:IGNF/FRACTAL-LidarHD_7cl_randlanet (runtime fetch; Etalab Open Licence 2.0, ungated)" \
      bradleylab.build.myria3d_ref="${MYRIA3D_REF}"

CMD ["bash"]
