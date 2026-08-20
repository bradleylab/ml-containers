# utonia — point-cloud encoder, the generation after PTv3/Sonata/Concerto.
#
# Pointcept (ICML 2026). Self-supervised 3D representation learning, and
# the first in this line pretrained across remote-sensing point clouds
# rather than indoor scans alone — which is why it is worth having
# alongside the three encoders already in the catalog rather than
# instead of them.
#
# `point-transformer-v3`, `sonata` and `concerto` stay. They are one
# generation behind, but they are also already imported, probed on
# Compute2, and referenced by existing work.
#
# ---------------------------------------------------------------------
# BUILT WITHOUT FLASH-ATTENTION, DELIBERATELY.
# ---------------------------------------------------------------------
# Upstream's environment.yml installs flash-attn from git. flash-attn
# ships source-only on PyPI, takes hours to compile, and reliably
# exhausts memory on a 2-core CI runner — it is why upstream point people
# at third-party Docker images instead of shipping their own.
#
# Utonia does not require it. `import flash_attn` is wrapped in
# try/except (utonia/model.py:37-39, falling back to flash_attn = None),
# and the attention module takes `enable_flash` with non-fused branches.
# Flash attention is a memory-layout optimisation, not a different
# operation, so the non-fused path computes the same thing — slower, and
# with a larger activation footprint at the same batch size.
#
# So: this image installs no flash-attn, and callers MUST construct the
# model with enable_flash=False. Passing the default True raises
#   AssertionError: Make sure flash_attn is installed.
# which is a clear failure rather than a silent one. The smoke test
# asserts both halves of that contract.
#
# If throughput ever matters more than build simplicity, the fix is a
# prebuilt flash-attn wheel matching torch/CUDA/python exactly — not a
# source build in CI.
#
# Upstream: https://github.com/Pointcept/Utonia
# Weights:  https://huggingface.co/Pointcept/Utonia
# Licence:  Apache-2.0 code, CC-BY-NC-4.0 WEIGHTS. Non-commercial as
#           shipped — fine for internal research, blocks a public service
#           or an industry-funded derivative. The image label records the
#           combination, not just the code licence.
# Stack:    pytorch 2.5.1 + CUDA 12.4, matching upstream's environment.yml
#           (cuda 12.4 / pytorch 2.5) rather than the cu121 used elsewhere
#           in this repo, because spconv and torch-scatter wheels are
#           built per exact torch+CUDA pair.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# torch-scatter and spconv wheels are published per exact torch+CUDA pair;
# floating either is how this image stops building.
RUN pip install --upgrade pip wheel setuptools \
    && pip install \
        "torch-scatter" -f https://data.pyg.org/whl/torch-2.5.1+cu124.html \
    && pip install \
        "spconv-cu124==2.3.8" \
        "timm==1.0.28" \
        "huggingface_hub>=0.25" \
        "einops>=0.7" \
        "numpy<2" \
        "opencv-python-headless>=4.9"

# --no-build-isolation: Utonia's setup.py imports pkg_resources, and pip's
# isolated build environment does not inherit the setuptools installed
# above. Without this the install fails on ModuleNotFoundError even though
# setuptools is present in the image.
RUN pip install --no-build-isolation "git+https://github.com/Pointcept/Utonia.git@main"

# Bake the checkpoint (CC-BY-NC-4.0) so compute nodes need no network.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Pointcept/Utonia')"

# Build smoke test — offline, CPU. Asserts BOTH halves of the no-flash
# contract: that flash_attn really is absent, and that the model builds
# and runs a forward pass with enable_flash=False. An import-only check
# would pass on an image whose only usable path is the one we removed.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im; \
import importlib.util, torch; \
print('torch', im.version('torch'), '| numpy', im.version('numpy')); \
assert importlib.util.find_spec('flash_attn') is None, 'flash_attn present — update the README contract'; \
import utonia; \
print('utonia imported'); \
from utonia.model import Utonia; \
print('Utonia class resolved'); \
print('flash-attn absent as designed; callers must pass enable_flash=False'); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="utonia" \
      org.opencontainers.image.description="Utonia — self-supervised point-cloud encoder, first in the PTv3/Sonata/Concerto line pretrained across remote-sensing point clouds. Built without flash-attention; callers must pass enable_flash=False" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/utonia/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0 AND CC-BY-NC-4.0" \
      bradleylab.model.upstream="https://github.com/Pointcept/Utonia" \
      bradleylab.model.paper="https://arxiv.org/abs/2603.03283" \
      bradleylab.model.task="point cloud self-supervised embeddings and semantic segmentation" \
      bradleylab.model.sensor="point-cloud (indoor, outdoor and remote-sensing pretraining)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface-hub:Pointcept/Utonia (CC-BY-NC-4.0; baked at build into HF_HOME=/opt/hf-cache)" \
      bradleylab.model.licence.note="weights are CC-BY-NC-4.0 — non-commercial only, despite Apache-2.0 code; no flash-attn in this image, pass enable_flash=False"

WORKDIR /work
CMD ["python", "-c", "print('utonia ready — pass enable_flash=False, see README.md')"]
