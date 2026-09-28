# syntax=docker/dockerfile:1
# anysat — AnySat, one Earth-observation encoder for many resolutions, scales
# and sensors (Astruc, Gonthier, Mallet, Landrieu; CVPR 2025).
#
# A scale-adaptive JEPA encoder pretrained self-supervised on GeoPlex, five
# multimodal datasets spanning 11 sensors. Any subset of those sensors over
# one extent goes in as a dict of tensors: single-date VHR (aerial,
# aerial-flair, spot, naip) and time series (s2, s1-asc, s1, alos, l7, l8,
# modis) with their day-of-year dates. Out come features — one vector per
# tile, per patch, or per sub-patch — not labels; a task head, even a linear
# one, has to be trained on top.
#
# The image serves upstream's torch.hub entrypoint from a pinned clone, fully
# offline: the checkpoint is baked where the entrypoint's
# torch.hub.load_state_dict_from_url looks first. Upstream's training stack
# (Lightning, Hydra, wandb, GeoPlex loaders) is not installed; the image
# extracts features and the helper scripts fit linear probes on them.
#
# Upstream: https://github.com/gastruc/AnySat @ 5f6f475 (MIT)
# Weights:  https://huggingface.co/g-astruc/AnySat @ 63f2521 (MIT)
# Paper:    https://arxiv.org/abs/2412.14123

FROM python:3.12-slim-bookworm

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the user's ~/.local off sys.path: enroot
# bind-mounts $HOME, and a host package shadowing an image one has broken
# images in this catalog before. TORCH_HOME is where the baked checkpoint
# lives; it is image-owned, so pyxis keeping the image's value over --export
# is what we want here.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch-hub

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Upstream asks only for torch>=2.0. 2.12.1 on CUDA 12.9 is the build the
# `nasa-ibm-lunar-fm` image already runs on Compute2's H100 nodes, whose
# driver (580.105.08) supports CUDA up to 13.0. torchvision is left out: the
# inference path never imports it.
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu129 torch==2.12.1

# constraints.txt is the full resolved set, so a rebuild does not drift and
# the resolver cannot swap the CUDA 12.9 torch above for another build.
COPY constraints.txt /opt/constraints.txt
RUN python -m pip install -c /opt/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        numpy rasterio pyproj scikit-learn pystac-client planetary-computer

# Not pip-installable: upstream loads through torch.hub, which puts the clone
# on sys.path and imports hubconf.py from it.
ARG ANYSAT_REF=5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a
RUN git clone https://github.com/gastruc/AnySat.git /opt/anysat \
    && git -C /opt/anysat checkout "$ANYSAT_REF" \
    && rm -rf /opt/anysat/.git \
    && echo "$ANYSAT_REF" > /opt/anysat/UPSTREAM_REF

# hubconf.py fetches models/AnySat.pth (504 MB) with
# torch.hub.load_state_dict_from_url, which returns the file from
# $TORCH_HOME/hub/checkpoints/<URL basename> without touching the network if
# it is already there. The repository's other file, models/AnySat_full.pth
# (1.1 GB), is not referenced by the hub entrypoint and is not baked.
ARG ANYSAT_HF_REV=63f252119babfe896d200f88631496f86eddacd3
ARG ANYSAT_SHA256=ce1246527a28bade624048c448a1fa473ca85791b998f562ce2bcc3dcac93d4f
RUN mkdir -p /opt/torch-hub/hub/checkpoints \
    && curl -fsSL --retry 5 -o /opt/torch-hub/hub/checkpoints/AnySat.pth \
        "https://huggingface.co/g-astruc/AnySat/resolve/${ANYSAT_HF_REV}/models/AnySat.pth" \
    && echo "${ANYSAT_SHA256}  /opt/torch-hub/hub/checkpoints/AnySat.pth" | sha256sum -c -

COPY scripts/anysat_features.py scripts/fetch_aoi.py scripts/nlcd_probe.py /usr/local/bin/
RUN chmod +x /usr/local/bin/anysat_features.py /usr/local/bin/fetch_aoi.py /usr/local/bin/nlcd_probe.py \
    && mkdir -p /work

# Build smoke test, with the network cut so a missing bake fails here rather
# than downloading. Imports are split under faulthandler: a compiled-extension
# segfault has been silent in other images in this catalog until they were.
RUN --network=none python -X faulthandler -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda); assert torch.version.cuda and torch.version.cuda.startswith('12.9'), 'CUDA 12.9 torch was replaced'"
RUN --network=none python -X faulthandler -c "import importlib.metadata as im, numpy, rasterio, pyproj, sklearn, pystac_client, planetary_computer; print(*(f'{p} {im.version(p)}' for p in ('numpy', 'rasterio', 'pyproj', 'scikit-learn', 'pystac-client', 'planetary-computer')), sep=' | ')"
RUN --network=none python -X faulthandler /usr/local/bin/anysat_features.py --help > /dev/null
# 320 m extent: NAIP 256 px at 1.25 m, Sentinel-2 32 px at 10 m. With 40 m
# patches that is an 8 x 8 patch grid.
RUN --network=none python -X faulthandler -c "\
import importlib.metadata as im, torch; \
torch.manual_seed(0); \
m = torch.hub.load('/opt/anysat', 'anysat', source='local', pretrained=True, flash_attn=False).eval(); \
x = {'naip': torch.randn(1, 4, 256, 256), 's2': torch.randn(1, 6, 10, 32, 32), \
     's2_dates': torch.tensor([[15, 75, 135, 195, 255, 315]])}; \
torch.set_grad_enabled(False); \
tile = m(x, patch_size=40, output='tile'); \
patch = m(x, patch_size=40, output='patch'); \
tokens = m(x, patch_size=40, output='all'); \
assert tile.shape == (1, 768), tile.shape; \
assert patch.shape == (1, 8, 8, 768), patch.shape; \
assert torch.isfinite(tile).all() and torch.isfinite(patch).all(), 'non-finite features'; \
assert torch.allclose(tokens[:, 0], tile) and torch.allclose(tokens[:, 1:].reshape(1, 8, 8, 768), patch), \"'all' disagrees with 'tile'/'patch'\"; \
print('torch', torch.__version__, '| numpy', im.version('numpy')); \
print('tile', tuple(tile.shape), '| patch', tuple(patch.shape)); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="anysat" \
      org.opencontainers.image.description="AnySat — JEPA Earth-observation encoder over any combination of 11 sensors (aerial 0.2 m, SPOT, NAIP, Sentinel-1/2, ALOS-2, Landsat 7/8, MODIS time series); tile, patch and dense features. Weights baked" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/anysat/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/gastruc/AnySat" \
      bradleylab.model.paper="https://arxiv.org/abs/2412.14123" \
      bradleylab.model.task="multimodal, multi-resolution Earth-observation feature extraction (tile, patch, dense); heads such as linear probes are trained separately" \
      bradleylab.model.sensor="image:multi (aerial RGBN 0.2 m, aerial RGBN+elevation 0.2 m, SPOT 1 m, NAIP 1.25 m; time series: Sentinel-2, Sentinel-1, ALOS-2, Landsat 7, Landsat 8, MODIS)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.weights.source="baked — huggingface.co/g-astruc/AnySat @ 63f2521, models/AnySat.pth (504 MB, sha256 ce124652…), at /opt/torch-hub/hub/checkpoints" \
      bradleylab.model.licence.note="MIT for the code (gastruc/AnySat LICENSE, whose copyright line reads Damien Robert, 2023) and MIT for the weights (Hugging Face model card). The GeoPlex pretraining datasets are not redistributed here"

WORKDIR /work
CMD ["bash", "-c", "echo 'anysat ready: torch.hub.load(\"/opt/anysat\", \"anysat\", source=\"local\", pretrained=True, flash_attn=False), or anysat_features.py --help'"]
