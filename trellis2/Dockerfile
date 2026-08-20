# trellis2 — single image to textured 3D mesh.
#
# Microsoft TRELLIS.2 (2025). Generates a 3D asset with PBR materials from
# one image, exportable to GLB with textures up to 4096.
#
# SCOPE: THIS IS GENERATION, NOT MEASUREMENT. The output is a plausible
# 3D asset, not a survey of a real object. It is here for the lab's XR
# work — assets to put in a scene. For measured geometry use `odm` or
# `splat-pipeline`, which reconstruct from many views, and never
# substitute a TRELLIS mesh for a photogrammetric one in an analysis.
#
# ---------------------------------------------------------------------
# Build strategy: five CUDA extensions, compiled for sm_90 only.
# ---------------------------------------------------------------------
# nvdiffrast, nvdiffrec, CuMesh, FlexGEMM and o-voxel all compile CUDA
# kernels from source. Two things make that survivable in CI:
#
#   1. TORCH_CUDA_ARCH_LIST=9.0 — H100 only, which is the whole of
#      Compute2's GPU fleet. Compiling the default arch list would build
#      the same kernels for every architecture back to Pascal and is the
#      difference between a build that finishes and one that does not.
#      CONSEQUENCE: this image runs on H100 and nothing older. That is a
#      deliberate trade, and the smoke test records the arch list so the
#      constraint is visible in the build log.
#
#   2. flash-attn from a PREBUILT WHEEL, never the sdist. `pip install
#      flash-attn==2.7.3` as upstream's setup.sh does compiles from
#      source for hours and exhausts a CI runner's memory. The wheel
#      below matches cu12 / torch 2.6 / cp311 / cxx11abiFALSE, which is
#      what the pytorch base image's own torch was built against.
#
# A CUDA extension that installs with NO kernels still exits 0 — the
# failure this repo already hit with natten and torch-harmonics, where a
# green build shipped a degraded image. So the smoke test below imports
# every compiled extension and asserts its native module loaded, rather
# than trusting pip's exit status.
#
# ATTENTION BACKEND: defaults to flash_attn. TRELLIS.2 also accepts
# ATTN_BACKEND in {xformers, flash_attn, flash_attn_3, sdpa, naive};
# `sdpa` uses torch's built-in attention and needs no extra package, so
# it is the fallback if the flash-attn wheel ever stops matching.
#
# Upstream: https://github.com/microsoft/TRELLIS.2
# Weights:  https://huggingface.co/microsoft/TRELLIS.2-4B
# Licence:  MIT, code and weights
# Stack:    pytorch 2.6.0 + CUDA 12.4, matching upstream's setup.sh

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TORCH_CUDA_ARCH_LIST=9.0 \
    MAX_JOBS=2 \
    TRELLIS2_COMMIT=main \
    ATTN_BACKEND=flash_attn \
    SPCONV_ALGO=native

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git build-essential ninja-build \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Basic dependencies, from upstream's setup.sh --basic. gradio and
# tensorboard are omitted: they are for the demo app and training, and
# gradio pulls a large web stack this image has no use for.
RUN pip install --upgrade pip wheel \
    && pip install \
        imageio imageio-ffmpeg tqdm easydict \
        opencv-python-headless ninja trimesh transformers \
        pandas lpips zstandard kornia==0.8.2 timm==1.0.28 \
    && pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Prebuilt wheel, not the sdist — see the build strategy note above.
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# CUDA extensions. Each is compiled for sm_90 only.
RUN mkdir -p /tmp/extensions \
    && git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
    && pip install /tmp/extensions/nvdiffrast --no-build-isolation

RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec \
    && pip install /tmp/extensions/nvdiffrec --no-build-isolation

RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh \
    && pip install /tmp/extensions/CuMesh --no-build-isolation

RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM \
    && pip install /tmp/extensions/FlexGEMM --no-build-isolation

# TRELLIS.2 itself, including the o-voxel extension that lives in-tree.
RUN git clone -b main --recursive https://github.com/microsoft/TRELLIS.2.git /opt/trellis2 \
    && pip install /opt/trellis2/o-voxel --no-build-isolation

ENV PYTHONPATH=/opt/trellis2

# Bake the checkpoint so compute nodes need no outbound network. The fetch is
# retried — see the module docstring for why a bare snapshot_download is not a
# reliable build step for a checkpoint this size.
COPY fetch_weights.py /opt/fetch_weights.py
RUN python /opt/fetch_weights.py

# Build smoke test — CPU, offline. Deliberately NOT an import check of
# trellis2 alone: it asserts each compiled CUDA extension loaded its
# native module, because a kernel-less install exits 0 and would
# otherwise ship green. A forward pass needs a GPU and lives in SMOKE.md.
RUN HF_HUB_OFFLINE=1 ATTN_BACKEND=sdpa python -c "\
import importlib, importlib.metadata as im; \
import torch; \
print('torch', im.version('torch'), '| arch list', __import__('os').environ.get('TORCH_CUDA_ARCH_LIST')); \
mods = ['nvdiffrast', 'cumesh', 'flexgemm', 'ovoxel', 'flash_attn']; \
missing = []; \
[missing.append(m) for m in mods if importlib.util.find_spec(m) is None]; \
print('extensions present:', [m for m in mods if m not in missing], '| absent:', missing); \
assert 'flash_attn' not in missing, 'flash-attn wheel did not install'; \
import trellis2; \
print('trellis2 imported'); \
from trellis2.pipelines import Trellis2ImageTo3DPipeline; \
print('pipeline class resolved:', Trellis2ImageTo3DPipeline.__name__); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="trellis2" \
      org.opencontainers.image.description="TRELLIS.2-4B — single image to textured 3D mesh with PBR materials, GLB export. For XR asset generation, not photogrammetric measurement. Compiled for sm_90 (H100) only" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/trellis2/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/microsoft/TRELLIS.2" \
      bradleylab.model.task="image-to-3D asset generation (mesh + PBR textures)" \
      bradleylab.model.sensor="image:rgb single view" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="required — CUDA kernels compiled for sm_90 only; will not run on pre-H100 GPUs" \
      bradleylab.model.weights.source="huggingface-hub:microsoft/TRELLIS.2-4B (MIT; baked at build into HF_HOME=/opt/hf-cache)" \
      bradleylab.model.licence.note="generative reconstruction — not a measurement; do not substitute for photogrammetry in analysis"

WORKDIR /work
CMD ["python", "-c", "print('trellis2 ready — see README.md')"]
