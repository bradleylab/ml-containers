# rf-detr — trainable object detector, Apache-2.0.
#
# Roboflow RF-DETR (ICLR 2026). Real-time detection on a DINOv2 backbone,
# built for fine-tuning on your own annotations — which is the gap it fills
# here. `deepforest` detects one class it was trained for; `grounding-dino`
# finds what you can describe but cannot learn from your labels. Neither
# answers "I annotated 500 drone frames, give me a detector".
#
# ---------------------------------------------------------------------
# WHY NOT YOLO.
# ---------------------------------------------------------------------
# Ultralytics YOLO is the obvious choice and is AGPL-3.0. Images in this
# repo are published to a PUBLIC registry, so an AGPL model would make the
# image AGPL-encumbered along with anything built from it, and the AGPL
# network clause would reach any service that ever served its predictions.
# RF-DETR does the same job under Apache-2.0, so the question does not
# arise. This is a deliberate substitution, not an oversight.
#
# ---------------------------------------------------------------------
# THE LICENCE SPLIT INSIDE RF-DETR.
# ---------------------------------------------------------------------
# Upstream ships two packages, and only one is Apache-2.0:
#
#   rfdetr       Apache-2.0   <- installed here
#   rfdetr_plus  PML 1.0      <- NOT installed; RF-DETR-XL and -2XL live here
#
# Installing rfdetr_plus would put PML-licensed weights inside an image
# labelled Apache-2.0 — the same trap depth-anything-3 has, where the
# family's checkpoints do not share one licence. The smoke test asserts
# rfdetr_plus is absent, so a later "let's use the bigger model" cannot
# quietly change what this image is.
#
# Upstream: https://github.com/roboflow/rf-detr
# Paper:    https://arxiv.org/abs/2511.09554
# Licence:  Apache-2.0 (package and the baked RFDETRMedium checkpoint)
# Stack:    pytorch 2.5.1 + CUDA 12.1, matching the other vision images

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TORCH_HOME=/opt/torch-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# rfdetr only. Pinned: it moves quickly and pulls transformers, which is the
# dependency most likely to break a vision model on a major bump.
RUN pip install --upgrade pip wheel \
    && pip install "rfdetr==1.9.4"

# Bake the COCO-pretrained checkpoint so compute nodes need no network.
# Instantiating the class is what triggers the download.
RUN python -c "\
from rfdetr import RFDETRMedium; \
RFDETRMedium(); \
print('checkpoint cached')"

# Build smoke test — offline, CPU. Loads the baked checkpoint and runs a real
# forward pass, and asserts the Apache/PML boundary holds. Whether it detects
# anything meaningful needs a real photograph and a GPU, which is SMOKE.md.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im, importlib.util; \
import numpy as np, torch; \
from PIL import Image; \
print('rfdetr', im.version('rfdetr'), '| torch', im.version('torch'), '| transformers', im.version('transformers')); \
assert importlib.util.find_spec('rfdetr_plus') is None, 'rfdetr_plus is PML 1.0 and must not be in an Apache-2.0 image'; \
from rfdetr import RFDETRMedium; \
m = RFDETRMedium(); \
img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)); \
det = m.predict(img, threshold=0.5); \
print('detections on noise:', len(det)); \
assert hasattr(det, 'xyxy') and hasattr(det, 'confidence'), dir(det)[:8]; \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="rf-detr" \
      org.opencontainers.image.description="RF-DETR (Roboflow) — real-time object detection on a DINOv2 backbone, designed for fine-tuning on your own annotations. Apache-2.0 alternative to the AGPL-licensed YOLO" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/rf-detr/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/roboflow/rf-detr" \
      bradleylab.model.paper="https://arxiv.org/abs/2511.09554" \
      bradleylab.model.task="object detection, fine-tunable on custom annotations" \
      bradleylab.model.sensor="image:rgb general" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="rfdetr RFDETRMedium COCO checkpoint (Apache-2.0; baked at build)" \
      bradleylab.model.licence.note="rfdetr_plus (RF-DETR-XL/2XL) is PML 1.0 and is deliberately NOT installed; the build asserts its absence"

WORKDIR /work
CMD ["python", "-c", "print('rf-detr ready — see README.md')"]
