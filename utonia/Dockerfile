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
# ---------------------------------------------------------------------
# THE CALLING CONTRACT: use utonia.load(), NOT from_pretrained().
# ---------------------------------------------------------------------
# PointTransformerV3 subclasses PyTorchModelHubMixin, so from_pretrained()
# exists and is the obvious thing to reach for. It is WRONG here.
# Pointcept/Utonia publishes no config.json — only .pth files — so
# from_pretrained() has nothing to configure from and silently falls back
# to the constructor defaults, enc_channels (48, 96, 192, 384, 512) over
# enc_num_head (3, 6, 12, 24, 32). That is head dimension 16 at every
# stage, and the model dies during construction with
#   AssertionError: Head dimension must be divisible by 3 for 3D RoPE, 16
#
# The real architecture lives INSIDE the checkpoint, under ckpt["config"]:
# channels (54, 108, 216, 432, 576) over heads (3, 6, 12, 24, 32), i.e.
# head dimension 18 throughout. utonia.load() reads it. So:
#
#   model = utonia.load("/opt/weights/utonia.pth",
#                       custom_config=dict(enable_flash=False,
#                                          enc_patch_size=[1024] * 5))
#
# custom_config overlays ckpt["config"] rather than replacing it, which is
# how enable_flash gets turned off without discarding the architecture.
# Both keys are needed: upstream's own no-flash path sets the patch size
# too (demo/3_batch_forward.py), because the non-fused attention holds a
# larger activation footprint. The smoke test below asserts the recovered
# architecture, so a checkpoint swap that changes it fails the build.
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
# setuptools held below 81: Utonia's setup.py imports pkg_resources, which
# modern setuptools has removed. Upgrading unpinned installs a version
# without it and the build fails inside setuptools' own build_meta.
RUN pip install --upgrade pip wheel "setuptools<81" \
    && pip install \
        "torch-scatter" -f https://data.pyg.org/whl/torch-2.5.1+cu124.html \
    && pip install \
        "spconv-cu124==2.3.8" \
        "timm==1.0.28" \
        "huggingface_hub>=0.25" \
        "einops>=0.7" \
        "addict>=2.4" \
        "scipy==1.14.1" \
        "packaging" \
        "numpy<2" \
        "opencv-python-headless>=4.9"

# --no-build-isolation: Utonia's setup.py imports pkg_resources, and pip's
# isolated build environment does not inherit the setuptools installed
# above. Without this the install fails on ModuleNotFoundError even though
# setuptools is present in the image.
RUN pip install --no-build-isolation "git+https://github.com/Pointcept/Utonia.git@main"

# Bake the checkpoint (CC-BY-NC-4.0) at a FIXED PATH, not in the HF cache.
#
# utonia.load()'s Hugging Face branch cannot reach a baked cache: its
# repo_id defaults to "Pointcept/utonia" (lowercase u) while the real repo
# is "Pointcept/Utonia", and it passes local_dir=~/.cache/utonia/ckpt,
# which bypasses HF_HOME entirely. Either alone would miss; together they
# guarantee a network fetch. load() also accepts a plain file path, which
# resolves nothing and needs no network — so that is what this image
# offers, and what the README documents.
RUN mkdir -p /opt/weights && python -c "\
import shutil; \
from huggingface_hub import hf_hub_download; \
shutil.copyfile(hf_hub_download('Pointcept/Utonia', 'utonia.pth'), '/opt/weights/utonia.pth'); \
print('checkpoint staged at /opt/weights/utonia.pth')"

# Build smoke test — offline, CPU.
#
# The previous version of this test resolved the model CLASS and stopped.
# It passed on an image whose only documented calling path
# (from_pretrained) could not construct a model at all — the failure only
# surfaced on an H100 weeks later. So this one builds the real model from
# the staged checkpoint and asserts the architecture it recovered.
#
# The asserted values are the checkpoint's own, read back on
# c2-gpu-005 (job 2950544, 2026-08-31). They are here to catch a
# checkpoint swap that silently changes the architecture, which is the
# failure that would otherwise reach a caller as a shape error.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im, importlib.util, torch, utonia; \
print('torch', im.version('torch'), '| numpy', im.version('numpy')); \
assert importlib.util.find_spec('flash_attn') is None, 'flash_attn present — update the README contract'; \
deps = ['addict', 'huggingface_hub', 'numpy', 'packaging', 'scipy', 'spconv', 'timm', 'torch', 'torch_scatter']; \
absent = [d for d in deps if importlib.util.find_spec(d) is None]; \
print('runtime deps absent:', absent); \
assert not absent, absent; \
cfg = utonia.load('/opt/weights/utonia.pth', ckpt_only=True)['config']; \
print('in_channels', cfg['in_channels'], '| enc_channels', tuple(cfg['enc_channels']), '| enc_num_head', tuple(cfg['enc_num_head'])); \
assert cfg['in_channels'] == 9, cfg['in_channels']; \
assert tuple(cfg['enc_channels']) == (54, 108, 216, 432, 576), cfg['enc_channels']; \
assert tuple(cfg['enc_depths']) == (3, 3, 3, 12, 3), cfg['enc_depths']; \
assert tuple(cfg['enc_num_head']) == (3, 6, 12, 24, 32), cfg['enc_num_head']; \
heads = [c // h for c, h in zip(cfg['enc_channels'], cfg['enc_num_head'])]; \
print('head dims', heads); \
assert all(d % 3 == 0 for d in heads), heads; \
model = utonia.load('/opt/weights/utonia.pth', custom_config=dict(enable_flash=False, enc_patch_size=[1024] * 5)).eval(); \
n = sum(p.numel() for p in model.parameters()); \
print('params', n); \
assert n == 137253744, n; \
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
      bradleylab.model.api="utonia.load('/opt/weights/utonia.pth', custom_config=dict(enable_flash=False, enc_patch_size=[1024]*5)) — NOT from_pretrained(), which builds the wrong architecture from defaults" \
      bradleylab.model.weights.source="huggingface-hub:Pointcept/Utonia (CC-BY-NC-4.0; utonia.pth staged at build to /opt/weights/utonia.pth)" \
      bradleylab.model.licence.note="weights are CC-BY-NC-4.0 — non-commercial only, despite Apache-2.0 code"

WORKDIR /work
CMD ["python", "-c", "print('utonia ready — load with utonia.load(\'/opt/weights/utonia.pth\', custom_config=dict(enable_flash=False, enc_patch_size=[1024]*5)); see README.md')"]
