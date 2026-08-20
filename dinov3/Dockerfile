# dinov3 — DINOv3 web-image ViT as a frozen feature extractor.
#
# Siméoni, Vo, Oquab et al. (2025) "DINOv3". Self-supervised vision
# transformer; this container ships the LVD-1689M variant, pretrained on
# 1.689 billion curated web images. It is the general-purpose counterpart
# to `dinov3-sat` in this repo, which carries the satellite weights.
#
# WHICH ONE TO USE. The two images are the same architecture with
# different priors, and picking the wrong one degrades features silently:
#
#   dinov3-sat  nadir RGB aerial/satellite orthoimagery
#   dinov3      everything else — field photographs, thin sections,
#               microscopy, close-range and oblique drone, lab imagery
#
# Before this image existed, `dinov3-sat` was the only DINOv3 in the
# catalog, so non-satellite imagery was being fed satellite priors.
#
# ViT-L/16, 1024-D patch tokens. Variable input size (the position
# embedding interpolates), so the caller chooses the trade between patch
# ground-size and token count.
#
# NORMALIZATION: these weights use ImageNet statistics — mean
# (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) — which are NOT the
# SAT weights' statistics. Feeding one model the other's constants
# produces no error, just worse embeddings, so both images resolve them
# from timm's pretrained_cfg and assert them in the build smoke test. A
# checkpoint swap that changes them fails the build rather than the
# science.
#
# Upstream: https://github.com/facebookresearch/dinov3
# Weights:  https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m
#           (the facebook/ repo of the same weights is manually gated;
#           the timm mirror is not, and carries the same licence)
# Licence:  DINOv3 License — NOT open source. Redistribution is permitted
#           only with a copy of the Agreement, which is why
#           LICENSE.dinov3.md is shipped inside this image. Publications
#           using it MUST acknowledge DINO Materials. See README.md.
#
#           Note the `*_qkvb.eupe_lvd1689m` variants on the same timm
#           mirror carry fair-noncommercial-research-license instead —
#           a different and stricter licence. This image uses the plain
#           lvd1689m weights.
# Stack:    pytorch 2.5.1 + CUDA 12.1 (matches `dinov3-sat` and `sam2`)
#
# Weights are baked at build time so the enroot .sqsh is self-contained on
# compute nodes with no outbound network. The smoke test runs offline
# against the baked cache.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# timm 1.0.28 matches `dinov3-sat`; the model entrypoint name is a timm
# registry detail, not a stable upstream API, so it is pinned.
RUN pip install --upgrade pip wheel \
    && pip install \
        "timm==1.0.28" \
        "safetensors>=0.4" \
        "Pillow>=10.0" \
        "numpy>=1.26"

COPY LICENSE.dinov3.md /opt/licenses/LICENSE.dinov3.md
COPY extract_features.py /opt/dinov3/extract_features.py

# Bake the checkpoint into the image (~1.2 GB) so compute nodes need no
# network. This is a redistribution of DINO Materials, permitted by the
# licence copied above and shipped with it.
RUN python -c "import timm; timm.create_model('hf-hub:timm/vit_large_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)"

# Smoke test — offline (HF_HUB_OFFLINE=1), no network, no fetch. Loads the
# baked weights, asserts the ImageNet normalization statistics and the
# token geometry the downstream head depends on.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im; \
import timm, torch; \
print('timm', im.version('timm'), '| torch', im.version('torch')); \
m = timm.create_model('hf-hub:timm/vit_large_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0).eval(); \
cfg = timm.data.resolve_model_data_config(m); \
print('mean', cfg['mean'], '| std', cfg['std'], '| input', cfg['input_size']); \
assert m.num_features == 1024, m.num_features; \
assert tuple(m.patch_embed.patch_size) == (16, 16), m.patch_embed.patch_size; \
assert all(abs(a - b) < 1e-6 for a, b in zip(cfg['mean'], (0.485, 0.456, 0.406))), cfg['mean']; \
assert all(abs(a - b) < 1e-6 for a, b in zip(cfg['std'], (0.229, 0.224, 0.225))), cfg['std']; \
f = m.forward_features(torch.zeros(1, 3, 256, 256)); \
print('forward_features', tuple(f.shape)); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="dinov3" \
      org.opencontainers.image.description="DINOv3 LVD-1689M ViT-L/16 frozen feature extractor — general web-image pretraining; dense 1024-D patch tokens for a downstream head. General-purpose counterpart to dinov3-sat" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/dinov3/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="LicenseRef-DINOv3" \
      bradleylab.model.upstream="https://github.com/facebookresearch/dinov3" \
      bradleylab.model.task="dense feature extraction (frozen backbone for segmentation/classification heads)" \
      bradleylab.model.sensor="image:rgb general — field photography, microscopy, thin section, close-range and oblique drone" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:timm/vit_large_patch16_dinov3.lvd1689m (baked at build into HF_HOME=/opt/hf-cache)" \
      bradleylab.model.licence.note="DINOv3 License at /opt/licenses/LICENSE.dinov3.md — redistribute only with this file; publications must acknowledge DINO Materials"

WORKDIR /work
CMD ["python", "/opt/dinov3/extract_features.py", "--help"]
