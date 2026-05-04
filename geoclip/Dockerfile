# geoclip — worldwide image geolocalization via CLIP-aligned GPS.
#
# Vivanco Cepeda, Nayak, Shah (NeurIPS 2023) — CLIP-style alignment
# between RGB photo embeddings and a learned location encoder over
# MP-16 (~4.7M global geo-tagged photos). Given an image, returns the
# top-k predicted (lat, lon) locations and probabilities. SOTA on
# Im2GPS3k, YFCC26k, GWS15k, NUS-Wide.
#
# Backbone: CLIP ViT-L/14 image encoder + small MLP location encoder
# + a 100K-point GPS gallery (lat/lon coordinates_100K.csv) bundled
# with the pip package. Top-k retrieval against the gallery returns
# discrete candidate locations.
#
# Upstream: https://github.com/VicenteVivan/geo-clip
# Paper:    https://arxiv.org/abs/2309.16020 (Vivanco Cepeda et al. 2023)
# License:  MIT
#
# Multi-arch: linux/amd64 + linux/arm64. Every dependency publishes
# aarch64 wheels (torch 2.5.1 CPU, torchvision 0.20.1, Pillow,
# huggingface_hub, geoclip is pure Python). No C toolchain in image.
#
# Weights are baked at build time. The geoclip package does NOT ship
# its own weights — `GeoCLIP(from_pretrained=True)` fetches from HF
# Hub on instantiation. To make the container offline-capable, we
# instantiate during build, which populates HF_HOME=/opt/hf-cache.
# This adds ~900 MB of CLIP-L/14 weights to the image but means
# `model.predict(...)` runs without network access at runtime.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU wheels — matches seisbench / remoteclip / neuralhydrology
# for ABI consistency across the catalog.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1

RUN pip install \
        "geoclip>=1.2" \
        "transformers>=4.40,<5" \
        "Pillow>=10.0" \
        "huggingface_hub>=0.25,<1.0"
# transformers<5 is required: geoclip's ImageEncoder calls CLIP and
# expects a tensor, not the BaseModelOutputWithPooling structured
# output returned by transformers 5.x. huggingface_hub<1.0 keeps the
# ABI compatible with transformers 4.x (the 1.x release is paired
# with transformers 5.x).

RUN mkdir -p /opt/hf-cache /opt/scripts
COPY scripts/geoclip_predict.py /opt/scripts/geoclip_predict.py

# Build-time smoke test #1: package + classes reachable.
# importlib.metadata.version per ml-containers rules.
RUN python -c "import torch, geoclip, huggingface_hub; \
import importlib.metadata as im; \
print('geoclip', im.version('geoclip'), '| torch', torch.__version__, '| huggingface_hub', huggingface_hub.__version__); \
from geoclip import GeoCLIP; \
assert callable(GeoCLIP)"

# Build-time smoke test #2 + weight bake: instantiate GeoCLIP, which
# downloads the CLIP-L/14 backbone + location encoder + gallery
# weights into HF_HOME. After this layer, the container is
# offline-capable for predict() calls.
RUN python -c "from geoclip import GeoCLIP; \
m = GeoCLIP(); m.eval(); \
print('image_encoder:', type(m.image_encoder).__name__); \
print('location_encoder:', type(m.location_encoder).__name__); \
print('gps_gallery:', tuple(m.gps_gallery.shape)); \
assert tuple(m.gps_gallery.shape) == (100000, 2)" \
    && du -sh /opt/hf-cache

LABEL org.opencontainers.image.title="geoclip" \
      org.opencontainers.image.description="GeoCLIP — worldwide image geolocalization via CLIP-aligned GPS coordinates (Vivanco Cepeda et al., NeurIPS 2023). CPU runtime. Weights baked at build time." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/geoclip/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/VicenteVivan/geo-clip" \
      bradleylab.model.paper="https://arxiv.org/abs/2309.16020" \
      bradleylab.model.task="worldwide image geolocalization (top-k GPS prediction from RGB photo)" \
      bradleylab.model.sensor="image:rgb" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="HuggingFace Hub via geoclip package; baked into image at /opt/hf-cache during build (offline runtime)"

WORKDIR /work
CMD ["bash"]
