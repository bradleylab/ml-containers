# sam3 — promptable concept segmentation: segment everything you *name*.
#
# Meta (2025) "SAM 3". The successor to SAM 2, and the difference is not
# incremental. SAM 2 segments what you point at — a click, a box, a mask.
# SAM 3 segments what you describe: give it "tree", and it finds and
# masks every tree in the image or video, with no exemplar and no
# training. For "every crown", "every crater", "every fracture" that is
# the difference between a prompt and a labelling campaign.
#
# `sam2` stays in the catalog. It is not deprecated by this: point-prompt
# segmentation is still the right tool when you know where the object is
# and want one mask, and existing work references the published image.
#
# ---------------------------------------------------------------------
# WEIGHTS ARE NOT BAKED. This is deliberate and is the whole design.
# ---------------------------------------------------------------------
# facebook/sam3 is a MANUALLY GATED repository — access is granted per
# account by Meta. The SAM License does permit redistribution provided a
# copy of the Agreement travels with the materials, so baking the weights
# would be licence-compliant. We do not, because publishing them in an
# image on a public registry would hand them to anyone who pulls it,
# bypassing the gate Meta deliberately put in front of them. Compliant
# is not the same as appropriate.
#
# So this image carries the environment and no SAM Materials. At run time
# the caller supplies their own HF token, having accepted the gate
# themselves, and the weights are fetched to a mounted cache:
#
#   HF_TOKEN=...  HF_HOME=/mounted/cache  python your_script.py
#
# Consequences to know about:
#   - The container needs outbound network on first use. On Compute2 that
#     means staging the cache from a login node, then running offline with
#     HF_HUB_OFFLINE=1. SMOKE.md shows both steps.
#   - The build smoke test cannot load weights, so it verifies the
#     architecture is registered and the class resolves. Weight loading is
#     proven in SMOKE.md instead, not here.
#
# The Agreement is shipped at /opt/licenses/LICENSE.sam3.md for reference
# even though no SAM Materials are distributed — anyone who fetches
# weights into this container is bound by it, and it should be readable
# without leaving the node.
#
# Upstream: https://github.com/facebookresearch/sam3
# Weights:  https://huggingface.co/facebook/sam3  (gated, runtime fetch)
# Licence:  SAM License — redistribution permitted with the Agreement;
#           publications must acknowledge SAM Materials; trade-control
#           terms forbid military, weapons and ITAR uses.
# Stack:    pytorch 2.5.1 + CUDA 12.1, matching the other vision images

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    SAM3_MODEL_ID=facebook/sam3

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel \
    && pip install \
        "transformers==5.15.1" \
        "Pillow>=10.0" \
        "numpy>=1.26" \
        "opencv-python-headless>=4.9" \
        "decord>=0.6"

COPY LICENSE.sam3.md /opt/licenses/LICENSE.sam3.md

# Build smoke test — no weights, no network, no token. Verifies that this
# transformers build actually knows the sam3_video architecture, which is
# the thing most likely to break on a transformers bump. Resolving the
# class from the model type does not touch the gated repository.
RUN python -c "\
import importlib.metadata as im; \
import transformers; \
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES; \
print('transformers', im.version('transformers'), '| torch', im.version('torch')); \
assert 'sam3_video' in CONFIG_MAPPING_NAMES, sorted(k for k in CONFIG_MAPPING_NAMES if 'sam' in k); \
from transformers import Sam3VideoModel; \
print('Sam3VideoModel resolved:', Sam3VideoModel.__name__); \
print('weights are NOT baked — fetch at runtime with your own HF token'); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="sam3" \
      org.opencontainers.image.description="SAM 3 — promptable concept segmentation: masks every instance of a named concept in image or video, no exemplar and no training. Weights are NOT baked; fetched at runtime with the caller's own HF token because the upstream repo is manually gated" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/sam3/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="LicenseRef-SAM" \
      bradleylab.model.upstream="https://github.com/facebookresearch/sam3" \
      bradleylab.model.task="promptable concept segmentation (text-prompted instance masks) in images and video" \
      bradleylab.model.sensor="image:rgb general; video" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface-hub:facebook/sam3 — GATED, NOT baked; runtime fetch with the caller's own HF_TOKEN" \
      bradleylab.model.licence.note="SAM License at /opt/licenses/LICENSE.sam3.md — publications must acknowledge SAM Materials; trade-control terms forbid military, weapons and ITAR uses"

WORKDIR /work
CMD ["python", "-c", "print('sam3 ready — weights fetch at runtime, see README.md')"]
