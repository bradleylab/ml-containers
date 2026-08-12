# dinov3-sat — DINOv3 satellite-pretrained ViT as a frozen feature extractor.
#
# Siméoni, Vo, Oquab et al. (2025) "DINOv3". Self-supervised vision
# transformer; this container ships the SAT-493M variant, pretrained on
# 493 million 512x512 Maxar RGB ortho-rectified tiles at 0.6 m GSD —
# not the web-image weights. That makes it the right prior for nadir
# RGB aerial/satellite orthoimagery, which is what the lab feeds it.
#
# ViT-L/16, 1024-D patch tokens. Variable input size (the position
# embedding interpolates), so the caller chooses the trade between
# patch ground-size and token count: at a 78 m chip, 512 px input gives
# 2.44 m per patch, 1024 px gives 1.22 m.
#
# Frozen encoder only — no task head. It emits dense patch features for
# a downstream head to consume (cf. `remoteclip`, `croma`, `clay`).
#
# NORMALIZATION: the SAT weights use their own mean/std, NOT ImageNet's.
# Feeding ImageNet statistics degrades features silently — no error, just
# worse embeddings. Always resolve them from timm's pretrained_cfg; the
# smoke test below asserts the expected values so a checkpoint swap that
# changes them fails the build rather than the science.
#
# Upstream: https://github.com/facebookresearch/dinov3
# Weights:  https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m
#           (the facebook/ repo of the same weights is manually gated;
#           the timm mirror is not, and carries the same licence)
# Licence:  DINOv3 License — NOT open source. Redistribution is permitted
#           only with a copy of the Agreement, which is why LICENSE.dinov3.md
#           is shipped inside this image. Publications using it MUST
#           acknowledge DINO Materials. See README.md.
# Stack:    pytorch 2.5.1 + CUDA 12.1 (matches `sam2`, the other H100 image)
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

# timm 1.0.28 is the version the DINOv3-SAT architecture + weights were
# verified against (2026-08-11); pinned because the model entrypoint name
# is a timm registry detail, not a stable upstream API.
RUN pip install --upgrade pip wheel \
    && pip install \
        "timm==1.0.28" \
        "safetensors>=0.4" \
        "Pillow>=10.0" \
        "numpy>=1.26"

COPY LICENSE.dinov3.md /opt/licenses/LICENSE.dinov3.md
COPY extract_features.py /opt/dinov3-sat/extract_features.py

# Bake the checkpoint into the image (~1.2 GB) so compute nodes need no
# network. This is a redistribution of DINO Materials, permitted by the
# licence copied above and shipped with it.
RUN python -c "import timm; timm.create_model('hf-hub:timm/vit_large_patch16_dinov3.sat493m', pretrained=True, num_classes=0)"

# Smoke test — offline (HF_HUB_OFFLINE=1), no network, no fetch. Loads the
# baked weights, asserts the SAT normalization statistics and the token
# geometry the downstream head depends on.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im; \
import timm, torch; \
print('timm', im.version('timm'), '| torch', im.version('torch')); \
m = timm.create_model('hf-hub:timm/vit_large_patch16_dinov3.sat493m', pretrained=True, num_classes=0).eval(); \
cfg = timm.data.resolve_model_data_config(m); \
print('mean', cfg['mean'], '| std', cfg['std'], '| input', cfg['input_size']); \
assert m.num_features == 1024, m.num_features; \
assert tuple(m.patch_embed.patch_size) == (16, 16), m.patch_embed.patch_size; \
assert all(abs(a - b) < 1e-6 for a, b in zip(cfg['mean'], (0.430, 0.411, 0.296))), cfg['mean']; \
assert all(abs(a - b) < 1e-6 for a, b in zip(cfg['std'], (0.213, 0.156, 0.143))), cfg['std']; \
f = m.forward_features(torch.zeros(1, 3, 256, 256)); \
print('forward_features', tuple(f.shape)); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="dinov3-sat" \
      org.opencontainers.image.description="DINOv3 SAT-493M ViT-L/16 frozen feature extractor — satellite/aerial RGB ortho pretraining (493M Maxar tiles at 0.6 m); dense 1024-D patch tokens for a downstream head" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/dinov3-sat/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="LicenseRef-DINOv3" \
      bradleylab.model.upstream="https://github.com/facebookresearch/dinov3" \
      bradleylab.model.task="dense feature extraction (frozen backbone for segmentation/classification heads)" \
      bradleylab.model.sensor="image:rgb-aerial/satellite ortho (pretrained at 0.6 m; runs at any GSD)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:timm/vit_large_patch16_dinov3.sat493m (baked at build into HF_HOME=/opt/hf-cache)" \
      bradleylab.model.licence.note="DINOv3 License at /opt/licenses/LICENSE.dinov3.md — redistribute only with this file; publications must acknowledge DINO Materials"

WORKDIR /work
CMD ["python", "/opt/dinov3-sat/extract_features.py", "--help"]
