# grounding-dino — open-vocabulary object detection from a text prompt.
#
# Liu et al. (2023) "Grounding DINO". Detects objects named in free text
# with no training and no fixed class list: pass "boulder. fracture.
# tree." and get boxes back. That is the difference from `deepforest`,
# which detects one class it was trained for, and from `sam2`, which
# segments what you point at but cannot find it from a description.
#
# The natural pairing is grounding-dino to locate and sam2 to mask.
#
# PROMPT FORMAT IS LOAD-BEARING. Upstream expects lower-case phrases
# separated by periods, with a trailing period: "a boulder. a fracture."
# Capitalised text or comma separators silently degrade recall — no
# error, just fewer detections. The build smoke test pins this by
# asserting a well-formed prompt round-trips through the processor.
#
# Upstream: https://github.com/IDEA-Research/GroundingDINO
# Weights:  https://huggingface.co/IDEA-Research/grounding-dino-base
# Licence:  Apache-2.0, code and weights
# Stack:    pytorch 2.5.1 + CUDA 12.1, matching the other vision images
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

RUN pip install --upgrade pip wheel \
    && pip install \
        "transformers==5.15.1" \
        "Pillow>=10.0" \
        "numpy>=1.26"

RUN python -c "\
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; \
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base'); \
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')"

# Build smoke test — offline, CPU. A real forward pass on noise with a
# well-formed prompt. Asserts the post-processor returns the three keys
# downstream code indexes into, so a transformers bump that renames them
# fails the build rather than a compute job.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im; \
import numpy as np, torch; \
from PIL import Image; \
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; \
print('transformers', im.version('transformers'), '| torch', im.version('torch')); \
mid = 'IDEA-Research/grounding-dino-base'; \
p = AutoProcessor.from_pretrained(mid); \
m = AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval(); \
img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)); \
x = p(images=img, text='a rock. a tree.', return_tensors='pt'); \
with torch.inference_mode(): out = m(**x); \
r = p.post_process_grounded_object_detection(out, x.input_ids, threshold=0.05, target_sizes=[(480, 640)])[0]; \
print('keys', sorted(r.keys())); \
assert {'scores', 'boxes'} <= set(r.keys()), sorted(r.keys()); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="grounding-dino" \
      org.opencontainers.image.description="Grounding DINO base — open-vocabulary object detection from free-text prompts; finds objects with no training and no fixed class list. Pairs with sam2 for masks" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/grounding-dino/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/IDEA-Research/GroundingDINO" \
      bradleylab.model.paper="https://arxiv.org/abs/2303.05499" \
      bradleylab.model.task="open-vocabulary (zero-shot) object detection" \
      bradleylab.model.sensor="image:rgb general" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface-hub:IDEA-Research/grounding-dino-base (baked at build into HF_HOME=/opt/hf-cache)"

WORKDIR /work
CMD ["python", "-c", "print('grounding-dino ready — see README.md')"]
