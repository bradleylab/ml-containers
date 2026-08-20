# depth-anything-3 — monocular depth, multi-view geometry and camera pose
# from one model.
#
# Lin, Yang et al. (2025) "Depth Anything 3". Given one image it predicts
# metric-relative depth; given several it additionally recovers camera
# poses and intrinsics, which is the part that replaces a separate
# structure-from-motion step for quick reconstructions.
#
# Outputs per input image:
#   depth      [N, H, W]  float32
#   confidence [N, H, W]  float32
#   pose (w2c) [N, 3, 4]  float32
#   intrinsics [N, 3, 3]  float32
#
# WHY THIS AND NOT Depth-Anything-V2 OR VGGT. DA3 covers both — monocular
# depth (V2's job) and multi-view geometry plus pose (VGGT's job) — and it
# is Apache-2.0, where both of those are CC-BY-NC-4.0. One permissive
# image replaces two restricted ones.
#
# CHECKPOINT LICENCE IS PER-CHECKPOINT, NOT PER-REPO. The code is
# Apache-2.0, but the released weights are not uniformly so:
#
#   DA3-LARGE-1.1              Apache-2.0   <- baked here
#   DA3MONO-LARGE              Apache-2.0
#   DA3METRIC-LARGE            Apache-2.0
#   DA3-BASE                   Apache-2.0
#   DA3-LARGE                  CC-BY-NC-4.0
#   DA3NESTED-GIANT-LARGE-1.1  CC-BY-NC-4.0
#
# Upstream's own README example loads `depth-anything/da3-large`, which
# redirects to DA3-LARGE — a non-commercial checkpoint. Following that
# example verbatim would put NC weights inside an Apache image. This
# container deliberately bakes DA3-LARGE-1.1 instead, and the smoke test
# asserts which checkpoint is present so a later swap cannot quietly
# change the licence of the image.
#
# Upstream: https://github.com/ByteDance-Seed/Depth-Anything-3
# Weights:  https://huggingface.co/depth-anything/DA3-LARGE-1.1
# Licence:  Apache-2.0 (code and the baked weights)
# Stack:    pytorch 2.5.1 + CUDA 12.1, matching dinov3/dinov3-sat/sam2
#
# numpy is held below 2 because depth-anything-3 declares numpy<2; open3d
# and pycolmap in its dependency set are the usual reason that constraint
# cannot simply be relaxed.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

# libgl1/libglib2.0-0: open3d and opencv both dlopen these at import.
# libgomp1: pycolmap's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pinned exactly: the package pulls 32 dependencies including open3d,
# pycolmap and moviepy, and an unpinned install of that surface is how a
# working image stops building three months later.
RUN pip install --upgrade pip wheel \
    && pip install "depth-anything-3==0.1.1" "numpy<2"

# Bake the Apache-2.0 checkpoint so compute nodes need no outbound network.
RUN python -c "\
from depth_anything_3.api import DepthAnything3; \
DepthAnything3.from_pretrained('depth-anything/DA3-LARGE-1.1')"

# Build smoke test — offline, CPU. Proves the package imports, the baked
# checkpoint loads without network, and the model carries the parameter
# count we expect. The real forward pass needs a GPU and lives in
# SMOKE.md, which runs on Compute2.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im; \
import torch; \
from depth_anything_3.api import DepthAnything3; \
print('depth-anything-3', im.version('depth-anything-3'), '| torch', im.version('torch'), '| numpy', im.version('numpy')); \
m = DepthAnything3.from_pretrained('depth-anything/DA3-LARGE-1.1'); \
n = sum(p.numel() for p in m.model.parameters()) if hasattr(m, 'model') else sum(p.numel() for p in m.parameters()); \
print('parameters', f'{n:,}'); \
assert n > 100_000_000, n; \
assert im.version('numpy').startswith('1.'), im.version('numpy'); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="depth-anything-3" \
      org.opencontainers.image.description="Depth Anything 3 — monocular depth, multi-view geometry and camera pose from a single model; Apache-2.0 code and weights, replacing the non-commercial Depth-Anything-V2 and VGGT" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/depth-anything-3/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/ByteDance-Seed/Depth-Anything-3" \
      bradleylab.model.paper="https://arxiv.org/abs/2511.10647" \
      bradleylab.model.task="monocular depth estimation, multi-view geometry, camera pose and intrinsics recovery" \
      bradleylab.model.sensor="image:rgb general (single or multi-view)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface-hub:depth-anything/DA3-LARGE-1.1 (Apache-2.0; baked at build into HF_HOME=/opt/hf-cache)" \
      bradleylab.model.licence.note="checkpoint licences differ within this family — DA3-LARGE and DA3NESTED-GIANT-LARGE-1.1 are CC-BY-NC-4.0; this image bakes the Apache-2.0 DA3-LARGE-1.1"

WORKDIR /work
CMD ["python", "-c", "from depth_anything_3.api import DepthAnything3; print('depth-anything-3 ready')"]
