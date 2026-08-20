# birefnet — high-resolution foreground segmentation.
#
# Zheng et al. (2024) "BiRefNet". Bilateral reference network for
# dichotomous image segmentation: one high-quality foreground mask per
# image, at full resolution, with clean edges on thin structures — hair,
# twigs, antennae, grain boundaries — that coarser segmenters smear.
#
# HOW IT DIFFERS FROM sam2. SAM2 segments what you prompt it with and
# returns many candidate masks. BiRefNet is unprompted and returns one
# mask: the salient foreground. For "cut this specimen out of its
# background" it needs no interaction; for "segment this particular
# thing" use sam2.
#
# TRUST_REMOTE_CODE, AND WHY THE REVISION IS PINNED. This checkpoint is
# `custom_code`: loading it executes birefnet.py and BiRefNet_config.py
# fetched from the model repository. Pinning the version of transformers
# does not pin that code. So the revision is pinned to an exact commit,
# and the build fetches only that revision. Bumping it means reading the
# diff of those two files first — this is executable content, not data.
#
# Upstream: https://github.com/ZhengPeng7/BiRefNet
# Weights:  https://huggingface.co/ZhengPeng7/BiRefNet
# Licence:  MIT, code and weights
# Stack:    pytorch 2.5.1 + CUDA 12.1, matching the other vision images

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    BIREFNET_REVISION=e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel \
    && pip install \
        "transformers==5.15.1" \
        "timm==1.0.28" \
        "einops>=0.7" \
        "kornia==0.8.2" \
        "Pillow>=10.0" \
        "numpy>=1.26" \
        "opencv-python-headless>=4.9"

# kornia is pinned to 0.8.2 deliberately. 0.8.3 runs torch.jit.script at
# module scope and segfaults against torch 2.5.x — the failure that broke
# prithvi-eo and terramind in the 2026-08 build-rot sweep. Do not float it.

# Bake the checkpoint at the pinned revision. The revision governs the
# executable birefnet.py as well as the weights.
RUN python -c "\
import os; \
from transformers import AutoModelForImageSegmentation; \
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True, revision=os.environ['BIREFNET_REVISION'])"

# Build smoke test — offline, CPU. A real forward pass; asserts the mask
# comes back at input resolution and single-channel, which is what any
# downstream compositing step assumes.
#
# .float() is required, not tidiness: this checkpoint loads in half
# precision, and half-precision convolution is not implemented on CPU, so
# a float32 input against the fp16 bias raises
#   RuntimeError: Input type (float) and bias type (c10::Half) should be the same
# On a GPU either cast the model up or cast the input to half; on CPU the
# model must be cast up.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im, os; \
import numpy as np, torch; \
from transformers import AutoModelForImageSegmentation; \
print('transformers', im.version('transformers'), '| torch', im.version('torch'), '| kornia', im.version('kornia')); \
m = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True, revision=os.environ['BIREFNET_REVISION']).eval().float(); \
torch.set_grad_enabled(False); \
x = torch.zeros(1, 3, 1024, 1024); \
out = m(x); \
pred = out[-1] if isinstance(out, (list, tuple)) else out; \
pred = pred[-1] if isinstance(pred, (list, tuple)) else pred; \
pred = pred.sigmoid(); \
print('mask', tuple(pred.shape)); \
assert pred.shape[0] == 1 and pred.shape[1] == 1, pred.shape; \
assert pred.shape[-2:] == (1024, 1024), pred.shape; \
assert bool(torch.isfinite(pred).all()); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="birefnet" \
      org.opencontainers.image.description="BiRefNet — high-resolution dichotomous foreground segmentation with clean edges on thin structures; unprompted, one mask per image" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/birefnet/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/ZhengPeng7/BiRefNet" \
      bradleylab.model.paper="https://arxiv.org/abs/2401.03407" \
      bradleylab.model.task="dichotomous image segmentation (high-resolution foreground matting)" \
      bradleylab.model.sensor="image:rgb general" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:ZhengPeng7/BiRefNet at revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 (baked at build into HF_HOME=/opt/hf-cache)" \
      bradleylab.model.licence.note="trust_remote_code checkpoint — the pinned revision governs executable model code, not only weights"

WORKDIR /work
CMD ["python", "-c", "print('birefnet ready — see README.md')"]
