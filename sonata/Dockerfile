# Sonata for H100 — self-supervised PTv3 point-cloud encoder (Meta)
#
# Sonata (Meta / facebookresearch) is the self-supervised Point
# Transformer V3 encoder that Concerto was derived from. This image runs
# its two released capabilities:
#   - per-point feature EMBEDDINGS (encoder output)
#   - closed-set ScanNet-20 SEMANTIC segmentation via the shipped
#     linear-probe head (sonata_linear_prob_head_sc.pth)
#
# Upstream code:    https://github.com/facebookresearch/sonata  (Apache-2.0)
# Upstream weights: https://huggingface.co/facebook/sonata  (CC-BY-NC-4.0)
# Paper:            arXiv:2503.16429 (CVPR 2025)
#
# ## Build path — no compilation (same shape as the concerto image)
#
# Sonata is a STANDALONE package (does not import pointcept.*), so there
# is no ocnn / pointops / Pointcept import chain. Native deps are prebuilt
# wheels only: spconv-cu121 and torch_scatter. FlashAttention is OPTIONAL
# (model.py guards `import flash_attn`; the encoder runs enable_flash=False
# with upcast attention). We skip it, so nothing compiles.
#
# ## Base image
#
# Proven pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel + spconv-cu121, for
# parity with the sibling point-cloud images. Upstream ships/tests torch
# 2.5.0/cu124; nothing in Sonata's code is 2.5-only (the loader branches
# for torch<2.4), so 2.2.2/cu121 runs it. Documented fallback if a
# numerical issue surfaces: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel +
# spconv-cu124 + torch_scatter from the torch-2.5.0+cu124 wheel index.
#
# ## Scope (v1)
#
# Buildable environment + sonata weights + the ScanNet-20 probe head + a
# build-time smoke test that builds the model from the baked checkpoint.
# Ships experimental pending a Compute2 runtime check (embedding
# extraction + ScanNet mIoU via the probe head).

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/Sonata

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# numpy<2 pinned (torch 2.2 ABI; upstream pins numpy<=1.26.4).
RUN python -m pip install --upgrade pip && \
    python -m pip install "numpy==1.26.4"

# Native deps — both prebuilt wheels (no compile). spconv-cu121 ships
# sm_90 kernels; torch_scatter from the torch-2.2.x+cu121 PyG index.
RUN python -m pip install spconv-cu121 && \
    python -m pip install torch-scatter \
        -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Sonata's pure-Python deps (no FlashAttention, no demo viz stack).
RUN python -m pip install huggingface_hub timm addict scipy einops natsort

# Sonata source, pinned. Run from source via PYTHONPATH (package is
# /opt/Sonata/sonata).
RUN git clone https://github.com/facebookresearch/sonata.git /opt/Sonata && \
    cd /opt/Sonata && \
    git checkout 18c09ff8d713494f78a8213792262b910977a65d && \
    rm -rf .git

# Bake weights: sonata (434 MB encoder) + the ScanNet-20 linear-probe
# head (0.10 MB). Each .pth is a {"config","state_dict"} dict.
RUN python -c "from huggingface_hub import hf_hub_download; \
[hf_hub_download('facebook/sonata', f, local_dir='/opt/Sonata/checkpoints') \
 for f in ['sonata.pth', 'sonata_linear_prob_head_sc.pth']]" && \
    ls -lh /opt/Sonata/checkpoints

WORKDIR /opt/Sonata

# Build-time smoke test: versions, then BUILD the model from the baked
# checkpoint on CPU (config baked into the .pth; enable_flash forced off
# since flash_attn is absent) and load the state dict strictly. Also load
# the ScanNet probe head. No GPU needed for construction + load.
RUN python -c "import importlib.metadata as im, torch; \
print('torch', torch.__version__, '| cuda', torch.version.cuda); \
print('spconv-cu121', im.version('spconv-cu121'), '| torch_scatter', im.version('torch-scatter')); \
print('numpy', im.version('numpy'), '| timm', im.version('timm')); \
import torch_scatter, spconv.pytorch; \
from sonata.model import PointTransformerV3; print('imported sonata PointTransformerV3'); \
ck = torch.load('/opt/Sonata/checkpoints/sonata.pth', map_location='cpu'); \
ck['config']['enable_flash'] = False; \
m = PointTransformerV3(**ck['config']); \
mk, un = m.load_state_dict(ck['state_dict'], strict=False); \
assert len(mk) == 0 and len(un) == 0, f'state_dict mismatch: {len(mk)} missing / {len(un)} unexpected'; \
print('sonata built; params:', sum(p.numel() for p in m.parameters())); \
hk = torch.load('/opt/Sonata/checkpoints/sonata_linear_prob_head_sc.pth', map_location='cpu'); \
print('ScanNet-20 probe head config:', hk['config'])"

LABEL org.opencontainers.image.title="sonata" \
      org.opencontainers.image.description="Sonata self-supervised PTv3 encoder (Meta), per-point embeddings + ScanNet-20 linear-probe semseg, H100 sm_90" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/sonata/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0 AND CC-BY-NC-4.0" \
      bradleylab.model.upstream="https://github.com/facebookresearch/sonata" \
      bradleylab.model.paper="https://arxiv.org/abs/2503.16429" \
      bradleylab.model.task="point cloud self-supervised embeddings + semantic segmentation" \
      bradleylab.model.sensor="lidar:terrestrial/aerial" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:facebook/sonata (sonata + ScanNet probe), baked at build; CC-BY-NC-4.0"

# Encoder-only backbone — no turnkey CLI. See README for the embedding-
# extraction and ScanNet-probe-segmentation snippets.
CMD ["bash"]
