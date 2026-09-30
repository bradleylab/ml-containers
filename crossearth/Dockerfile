# CrossEarth — vision foundation model for cross-domain remote sensing
# semantic segmentation (TPAMI 2025).
#
# CrossEarth uses a frozen DINOv2 backbone with an Earth-Style Injection
# pipeline + multi-task training to bridge domain gaps in spectral
# bands, region, resolution, and climate. It targets RSDG (Remote Sensing
# Domain Generalization) — i.e., training on one set of source domains
# and zero-shot inference on unseen target domains.
#
# This container vendors the upstream `VisionXLab/CrossEarth` repo at a
# pinned commit (no PyPI release; not pip-installable).
#
# v3 added one runnable task, `crossearth-land-cover` (the ENTRYPOINT): a
# six-class land-cover map of an RGB orthophoto from the Potsdam RGB source
# model, under the geospatial executor's entrypoint contract v1. That model's
# weights are baked in, converted at build time by scripts/convert_checkpoint.py
# from the published checkpoint at a pinned revision and checksum, so the task
# runs offline. The conversion reads the checkpoint's pickle through an
# allowlist and keeps only the inference tensors; see the script's docstring.
# Every other checkpoint in `Cusyoung/CrossEarth` still downloads from the Hub
# on first call. For a shell, run with `--entrypoint bash`. v4 lets the task
# run on the CPU when no GPU is visible (the entrypoint's docstring says how).
#
# ---------------------------------------------------------------------
# WHY THIS IS NOT ON UPSTREAM'S TORCH 2.0.1 / CUDA 11.7 ANY MORE.
# ---------------------------------------------------------------------
# Upstream's conda recipe is torch 2.0.1 + cuda 11.7 + xformers 0.0.20.
# CUDA 11.7 predates sm_90, so a torch built on it cannot target an H100 —
# and Compute2's GPUs are all H100. The 2026-09-01 harness probe of this
# image did not fail: it HUNG, warning that the device's capability was not
# in torch's compiled arch list, and burned its whole time limit. That is
# the worst kind of failure for a batch job.
#
# The move is torch 2.1.0 + cu121, and the two pins that had to move with it:
#   - mmcv 2.1.0, which openmim publishes for torch 2.1.0 on cu121 (checked
#     2026-09-04 at download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/).
#   - xformers 0.0.22.post7, the build paired with torch 2.1.0. Upstream's
#     exact 0.0.20 pin is an environment snapshot, not an API constraint:
#     CrossEarth uses memory_efficient_attention, SwiGLU, fmha,
#     scaled_index_add and index_select_cat (models/backbones/dino_layers/),
#     all present and unchanged through 0.0.22.
# mmsegmentation and mmdet keep their upstream ranges.
#
# The smoke test asserts the wheel's CUDA toolkit is >= 11.8, the version
# that introduced sm_90. It does NOT read torch.cuda.get_arch_list(): that is
# populated from the driver at runtime and is [] on a GPU-less runner. The
# real arch probe runs in ml-jobs/harness on an H100 and is the regression
# test for the hang.
#
# Upstream: https://github.com/VisionXLab/CrossEarth (MIT)
# Paper:    https://arxiv.org/abs/2410.22629
# Weights:  https://huggingface.co/Cusyoung/CrossEarth
# Stack:    pytorch/pytorch 2.1.0-cuda12.1 + mmcv 2.1.0 + mmseg 1.x +
#           mmdet 3.x + xformers 0.0.22.post7

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    PYTHONPATH=/opt/CrossEarth

# apt with retries. Two consecutive CI builds died here on "Connection failed"
# to three different Ubuntu mirror IPs over ten minutes -- the GitHub-runner to
# Canonical path, not the package list. The focal base (the only Ubuntu the
# torch 2.1.0 image ships on) cannot be swapped, so the step is made to
# survive a flaky mirror instead: Acquire::Retries plus a three-attempt loop.
RUN for attempt in 1 2 3; do \
        apt-get update -o Acquire::Retries=5 \
        && apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
            git ca-certificates \
            libgl1 libglib2.0-0 \
        && break; \
        echo "apt attempt $attempt failed; retrying in 60 s"; sleep 60; \
    done \
    && rm -rf /var/lib/apt/lists/*

# Pin to upstream HEAD as of 2026-04-02. No PyPI release; vendor the
# repo directly. CrossEarth/__init__.py imports models/heads/segmentors,
# which side-effect-registers them with mmseg's MODELS registry.
ARG CROSSEARTH_SHA=644a5a1b3c01b2e5531820b5291d4397597f75de
RUN git clone https://github.com/VisionXLab/CrossEarth.git /opt/CrossEarth \
    && cd /opt/CrossEarth \
    && git checkout ${CROSSEARTH_SHA} \
    && rm -rf /opt/CrossEarth/.git

# The mm* stack, resolved for torch 2.1.0 + cu121 (see the header):
#   - openmim for mmcv/mmengine (mim resolves the cu121/torch2.1.0 wheel)
#   - mmcv ==2.1.0 — the openmim wheel that exists for this torch
#   - mmsegmentation >=1.0.0,<1.3 and mmdet >=3.0.0,<3.4, upstream's ranges
#   - xformers ==0.0.22.post7 — the build paired with torch 2.1.0
# requirements.txt: numpy ftfy scipy prettytable matplotlib regex timm einops
#
# numpy<2 stays load-bearing: torch 2.1.0 + torchvision 0.16.0 were compiled
# against the numpy 1.x ABI, and CrossEarth's unpinned `numpy` resolves to
# 2.x otherwise — surfacing as `RuntimeError: Numpy is not available` the
# first time torchvision touches a tensor.
RUN pip install --upgrade pip wheel \
    && pip install -U openmim \
    && mim install mmengine \
    && mim install "mmcv==2.1.0" \
    && pip install \
        "mmsegmentation>=1.0.0,<1.3" \
        "mmdet>=3.0.0,<3.4" \
        "xformers==0.0.22.post7" \
        "numpy<2" \
    && pip install -r /opt/CrossEarth/requirements.txt \
    && pip install future tensorboard "huggingface_hub>=0.20.0" "numpy<2"

# GeoTIFF reading and writing for the land-cover entrypoint. The rasterio wheel
# bundles its own GDAL; numpy<2 again, for the torch 2.1 ABI above. rasterio
# 1.4.3 leaves affine unpinned, and under this base image's Python 3.10 affine
# 3.0.1 cannot iterate an Affine, so reading any dataset's bounds fails ("No
# '__dict__' attribute on 'Affine' instance to cache '_astuple' property", v3's
# entrypoint on Compute2); affine 2.4.0 works.
RUN pip install "rasterio==1.4.3" "affine==2.4.0" "safetensors==0.8.0" "numpy<2"

RUN mkdir -p /opt/hf-cache

# The Potsdam RGB source model, converted to safetensors in one layer so the
# 1.7 GB training checkpoint never lands in the image. The revision pins the
# Hub commit and the checksum pins the file (the Hub's LFS oid for it).
ARG WEIGHTS_REVISION=fd54d96bd12765e73f8c228d5bec6c25012a1845
ARG WEIGHTS_FILE="Potsdam(r)-source.pth"
ARG WEIGHTS_SHA256=2bdf5bfe2e25dbb1c1a339212183a31c4a5e9979ce93111f4dbdd4d9f73755b2
COPY scripts/convert_checkpoint.py /opt/crossearth-tools/convert_checkpoint.py
RUN mkdir -p /opt/weights /tmp/checkpoint \
    && python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('Cusyoung/CrossEarth', '${WEIGHTS_FILE}', revision='${WEIGHTS_REVISION}', local_dir='/tmp/checkpoint')" \
    && python /opt/crossearth-tools/convert_checkpoint.py \
        "/tmp/checkpoint/${WEIGHTS_FILE}" /opt/weights/crossearth_potsdam_rgb.safetensors \
        --expect-sha256 "${WEIGHTS_SHA256}" \
        --source "huggingface.co/Cusyoung/CrossEarth@${WEIGHTS_REVISION}/${WEIGHTS_FILE}" \
    && rm -rf /tmp/checkpoint

COPY bin/crossearth-land-cover /usr/local/bin/crossearth-land-cover

# Smoke test — imports clean, registry side-effects fire, the
# CrossEarth-specific classes (Reins, LoRAReins, ReinMask2FormerHead,
# PEFTEncoderDecoder, ReinsDinoVisionTransformer) land in mmseg's
# MODELS / BACKBONES registries. Per `~/.claude/rules/ml-containers.md`,
# version printout uses importlib.metadata, not __version__, for every
# package. No weight fetch at build time (HF cache empty; first
# runtime call downloads).
RUN python -c "import torch; \
import torchvision; \
import torchvision.transforms.functional as TF; \
import numpy as np; \
TF.to_tensor(np.zeros((4, 4, 3), dtype=np.uint8)); \
import importlib.metadata as im; \
import sys; sys.path.insert(0, '/opt/CrossEarth'); \
import CrossEarth; \
from mmseg.registry import MODELS; \
expected = ['Reins', 'LoRAReins', 'ReinMask2FormerHead', 'PEFTEncoderDecoder', 'ReinsDinoVisionTransformer']; \
all_registered = set(MODELS.module_dict).union(*(set(c.module_dict) for c in MODELS.children.values())); \
found = [n for n in expected if n in all_registered]; \
assert len(found) >= 3, f'CrossEarth registry side-effects missing: found={found}, expected at least 3 of {expected}'; \
assert int(np.__version__.split('.')[0]) < 2, f'numpy must be <2 for the torch 2.1 ABI; got {np.__version__}'; \
assert tuple(map(int, torch.version.cuda.split('.')[:2])) >= (11, 8), f'sm_90 (H100) needs CUDA >= 11.8 — v1 was cu117 and hung; this wheel is cu{torch.version.cuda}'; \
from xformers.ops import memory_efficient_attention, SwiGLU, fmha; \
print('mmcv', im.version('mmcv'), \
'| mmsegmentation', im.version('mmsegmentation'), \
'| mmengine', im.version('mmengine'), \
'| mmdet', im.version('mmdet'), \
'| xformers', im.version('xformers'), \
'| torch', torch.__version__, \
'| torchvision', torchvision.__version__, \
'| numpy', np.__version__, \
'| cuda-build', torch.version.cuda); \
print('CrossEarth-specific classes registered:', found)"

# The land-cover model builds from upstream's config and takes the baked
# weights with no key missing or left over, on CPU (no GPU on the runner), and
# the entrypoint's resampling writes a GeoTIFF on a new grid.
RUN python -c "import importlib.machinery, importlib.util, torch, rasterio; \
import importlib.metadata as im; import numpy as np; from pathlib import Path; \
from rasterio.transform import from_origin; \
loader = importlib.machinery.SourceFileLoader('land_cover', '/usr/local/bin/crossearth-land-cover'); \
spec = importlib.util.spec_from_loader('land_cover', loader); \
import sys; land_cover = importlib.util.module_from_spec(spec); sys.modules['land_cover'] = land_cover; \
loader.exec_module(land_cover); \
model = land_cover.build_model(torch.device('cpu')); \
params = sum(p.numel() for p in model.parameters()); \
print('land-cover model:', params, 'parameters;', land_cover.weights_provenance()); \
site = rasterio.open('/tmp/site.tif', 'w', driver='GTiff', width=8, height=8, count=3, dtype='uint8', crs='EPSG:32615', transform=from_origin(0, 80, 10, 10)); \
site.write(np.ones((3, 8, 8), dtype='uint8')); site.close(); \
land_cover.resample(Path('/tmp/site.tif'), Path('/tmp/resampled.tif'), 5.0); \
print('resample check:', rasterio.open('/tmp/resampled.tif').shape); \
print('rasterio', im.version('rasterio'), '| affine', im.version('affine'), \
'| gdal', rasterio.__gdal_version__, '| safetensors', im.version('safetensors'))" \
    && rm -f /tmp/site.tif /tmp/resampled.tif

# gpu=required: torch 2.1.0 cu121 with sm_90; v1 (cu117) could not target the H100 and hung
LABEL org.opencontainers.image.title="crossearth" \
      org.opencontainers.image.description="CrossEarth (TPAMI 2025) — vision foundation model for cross-domain remote sensing semantic segmentation. Frozen DINOv2 backbone + Earth-Style Injection + multi-task training; targets Remote Sensing Domain Generalization (RSDG). Container vendors the upstream repo + mmcv/mmseg stack; ENTRYPOINT: a land-cover map of an RGB orthophoto (Potsdam RGB model, weights baked in, geospatial executor contract v1); other checkpoints via Hugging Face Hub at runtime." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/crossearth/README.md" \
      org.opencontainers.image.version="v4" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/VisionXLab/CrossEarth" \
      bradleylab.model.paper="https://arxiv.org/abs/2410.22629" \
      bradleylab.model.task="cross-domain semantic segmentation — domain-generalizable RSDG; frozen DINOv2 backbone + Mask2Former head + Earth-Style Injection" \
      bradleylab.model.sensor="image:multi (optical RS, region/resolution/spectral domain transfer)" \
      bradleylab.model.status="utility" \
      bradleylab.model.gpu="required" \
      bradleylab.model.weights.source="huggingface-hub:Cusyoung/CrossEarth; Potsdam(r)-source.pth baked at /opt/weights (converted to safetensors), others fetched at runtime into HF_HOME=/opt/hf-cache"

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/crossearth-land-cover"]
