# siglip2 — general image-text scoring with a sigmoid-loss CLIP.
#
# Tschannen et al. (2025) "SigLIP 2". Encodes images and text into one
# space, so an image can be scored against arbitrary text prompts with no
# training. This is the general-purpose counterpart to `remoteclip` and
# `dofa-clip` in this repo, both of which are remote-sensing specific and
# score poorly on anything that is not overhead imagery.
#
# so400m-patch14-384: 1.14 B parameters, 384x384 input, patch 14.
#
# WHAT IT IS FOR HERE. Zero-shot triage of image collections — sorting
# field photographs, flagging frames that contain a given feature,
# building a text-searchable index over a photo archive. It gives a
# similarity score, not a detection; for "where in the image" use
# `grounding-dino`, and for a mask use `sam2`.
#
# Upstream: https://github.com/google-research/big_vision
# Weights:  https://huggingface.co/google/siglip2-so400m-patch14-384
# Licence:  Apache-2.0, code and weights
# Stack:    pytorch 2.5.1 + CUDA 12.1, matching dinov3 and depth-anything-3
#
# Weights are baked so compute nodes need no outbound network.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# transformers pinned: SigLIP2 support is a model-registry detail, and an
# unpinned major bump is how this class of image stops loading.
RUN pip install --upgrade pip wheel \
    && pip install \
        "transformers==5.15.1" \
        "sentencepiece>=0.2" \
        "Pillow>=10.0" \
        "numpy>=1.26"

RUN python -c "\
from transformers import AutoModel, AutoProcessor; \
AutoModel.from_pretrained('google/siglip2-so400m-patch14-384'); \
AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')"

# Build smoke test — offline, CPU. A real forward pass, not just an import:
# an image of noise scored against two prompts must return one logit per
# prompt and finite values. A transformers bump that breaks the model
# fails the build here rather than on a compute node.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im; \
import numpy as np, torch; \
from PIL import Image; \
from transformers import AutoModel, AutoProcessor; \
print('transformers', im.version('transformers'), '| torch', im.version('torch')); \
mid = 'google/siglip2-so400m-patch14-384'; \
m = AutoModel.from_pretrained(mid).eval(); \
p = AutoProcessor.from_pretrained(mid); \
img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (384, 384, 3), dtype=np.uint8)); \
x = p(text=['a photograph of rock', 'a photograph of water'], images=img, padding='max_length', return_tensors='pt'); \
out = m(**x); \
print('logits_per_image', tuple(out.logits_per_image.shape)); \
assert tuple(out.logits_per_image.shape) == (1, 2), out.logits_per_image.shape; \
assert bool(torch.isfinite(out.logits_per_image).all()); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="siglip2" \
      org.opencontainers.image.description="SigLIP 2 so400m-patch14-384 — general image-text scoring; zero-shot triage and text search over image collections. General-purpose counterpart to remoteclip and dofa-clip" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/siglip2/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/google-research/big_vision" \
      bradleylab.model.paper="https://arxiv.org/abs/2502.14786" \
      bradleylab.model.task="zero-shot image-text scoring and retrieval" \
      bradleylab.model.sensor="image:rgb general" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:google/siglip2-so400m-patch14-384 (baked at build into HF_HOME=/opt/hf-cache)"

WORKDIR /work
CMD ["python", "-c", "print('siglip2 ready — see README.md')"]
