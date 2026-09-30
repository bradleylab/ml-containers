# aurora — Microsoft's Aurora foundation model for the Earth system.
#
# A 1.3-billion-parameter 3D Swin Transformer between Perceiver encoder and
# decoder layers, pretrained broadly and then fine-tuned per task; across its
# checkpoints the training data span ERA5, CMIP6, IFS HRES, GFS and CAMS. On
# the 0.25° global grid it takes the atmosphere at two times six hours apart
# (4 surface variables, and 5 atmospheric variables on 13 pressure levels) and
# predicts the state six hours later, rolled out autoregressively for longer
# leads. The same backbone carries released fine-tunes for 0.1° weather, CAMS
# air pollution and HRES-WAM ocean waves, and the Aurora 1.5 family with an
# expanded variable set.
#
# This image and its Compute2 check target the 0.25° weather checkpoints. The
# package also holds the classes for the other checkpoints; their input data
# are what limit them, as the README explains.
#
# ---------------------------------------------------------------------
# WEIGHTS ARE NOT BAKED.
# ---------------------------------------------------------------------
# Each full checkpoint is about 5 GB and the Hugging Face repository holds 41 GB
# across nine of them. They are MIT and ungated, so size is the only reason. The
# lab's copy is staged on Storage3 at Active/aurora by scripts/stage_weights.py,
# pinned to the Hugging Face revision the package itself pins, and is mounted
# at /weights.
#
# Upstream: https://github.com/microsoft/aurora @ v2.0.1 (0ad2d9a), MIT
# Weights:  https://huggingface.co/microsoft/aurora, MIT
# Paper:    Bodnar et al. 2025, Nature, https://doi.org/10.1038/s41586-025-09005-y

# Python 3.11: upstream's CI tests 3.10 and 3.11, and nothing newer.
FROM python:3.11-slim-bookworm

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the user's ~/.local off sys.path: enroot
# bind-mounts $HOME, and a host package shadowing an image one has broken
# images in this catalog before. HF_HOME is deliberately absent: pyxis lets
# image ENV beat srun --export, so jobs set it inside their own command.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# constraints.txt is the environment upstream's CI tested 2.0.1 in, with the
# torch build swapped for CUDA 12.9; its header has the details. torch gets a
# layer of its own, the multi-GB one, so a failure in the next step rebuilds
# without downloading it again. 12.9 rather than the 13.0 build CI used: it is
# the build nasa-ibm-lunar-fm runs on the same H100 nodes, and Compute2's
# driver (580.105.08) serves either.
COPY constraints.txt /opt/constraints.txt
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install -c /opt/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu129 torch

# xarray, zarr and gcsfs are for reading WeatherBench2 from public Google Cloud
# Storage without credentials, which is how SMOKE.md gets its initial states.
RUN python -m pip install -c /opt/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        microsoft-aurora xarray zarr gcsfs

COPY scripts/aurora-verify-onestep scripts/aurora-verify-small /usr/local/bin/
COPY scripts/stage_weights.py /opt/aurora/stage_weights.py

RUN mkdir -p /weights /work

# Build smoke test, offline and weight-free. torch imports in its own step
# under faulthandler so a compiled-extension crash is attributed to torch.
RUN python -X faulthandler -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda); assert torch.version.cuda and torch.version.cuda.startswith('12.9'), 'CUDA 12.9 torch was replaced'"
# Upstream's documented random-batch example, on the small architecture with
# random initialization: every variable must come back finite on the cropped
# grid (Aurora drops one latitude row to reach a multiple of its patch size).
RUN HF_HUB_OFFLINE=1 python -X faulthandler -c "\
import importlib.metadata as im; \
from datetime import datetime; \
import torch; \
import xarray as xr; \
import gcsfs, zarr; \
from aurora import AuroraSmallPretrained, Batch, Metadata; \
torch.manual_seed(0); \
batch = Batch( \
    surf_vars={k: torch.randn(1, 2, 17, 32) for k in ('2t', '10u', '10v', 'msl')}, \
    static_vars={k: torch.randn(17, 32) for k in ('lsm', 'z', 'slt')}, \
    atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ('z', 'u', 'v', 't', 'q')}, \
    metadata=Metadata(lat=torch.linspace(90, -90, 17), lon=torch.linspace(0, 360, 33)[:-1], \
                      time=(datetime(2020, 6, 1, 12, 0),), atmos_levels=(100, 250, 500, 850))); \
model = AuroraSmallPretrained().eval(); \
pred = model.forward(batch); \
surf = {k: tuple(v.shape) for k, v in pred.surf_vars.items()}; \
atmos = {k: tuple(v.shape) for k, v in pred.atmos_vars.items()}; \
assert all(s == (1, 1, 16, 32) for s in surf.values()), surf; \
assert all(s == (1, 1, 4, 16, 32) for s in atmos.values()), atmos; \
assert all(torch.isfinite(v).all() for v in [*pred.surf_vars.values(), *pred.atmos_vars.values()]), 'non-finite output'; \
assert pred.metadata.time == (datetime(2020, 6, 1, 18, 0),), pred.metadata.time; \
assert 'zarr' in xr.backends.list_engines(), 'xarray cannot see zarr'; \
print('microsoft-aurora', im.version('microsoft-aurora'), '| torch', torch.__version__, '| timm', im.version('timm'), '| numpy', im.version('numpy'), '| xarray', im.version('xarray'), '| zarr', im.version('zarr'), '| gcsfs', im.version('gcsfs'), '| huggingface-hub', im.version('huggingface-hub')); \
print('SMOKE OK')"
# The scripts need staged weights or network, so the build can only parse them.
RUN python -c "import ast, sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]" \
        /usr/local/bin/aurora-verify-onestep /usr/local/bin/aurora-verify-small /opt/aurora/stage_weights.py

LABEL org.opencontainers.image.title="aurora" \
      org.opencontainers.image.description="Microsoft Aurora — 1.3B-parameter foundation model for the Earth system; 0.25 deg global weather in 6 h steps (pretrained for ERA5, fine-tuned for IFS HRES T0); microsoft-aurora 2.0.1, torch 2.13.0 CUDA 12.9; weights mounted at runtime; H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/aurora/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/microsoft/aurora" \
      bradleylab.model.paper="https://doi.org/10.1038/s41586-025-09005-y" \
      bradleylab.model.task="global medium-range weather forecast, 0.25 deg, 6-hourly; released fine-tunes also cover 0.1 deg weather, air pollution and ocean waves" \
      bradleylab.model.sensor="ERA5 reanalysis / IFS HRES T0 gridded fields" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.weights.source="NOT baked — huggingface.co/microsoft/aurora @ 0be7e57: 0.25 deg pretrained, fine-tuned and small checkpoints, static fields and upstream's reference batch, 10.8 GB; staged lab-side on Storage3 at Active/aurora, mounted at /weights" \
      bradleylab.model.licence.note="MIT for the code and for all released checkpoints. Upstream asks commercial users to contact Microsoft and describes the release as research code, not an operational forecast service"

WORKDIR /work
CMD ["bash", "-c", "echo 'aurora ready: mount the staged weights at /weights, then run aurora-verify-onestep pretrained (see README.md)'"]
