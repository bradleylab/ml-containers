# Concerto for H100 — joint 2D-3D self-supervised PTv3 backbone (encoder-only)
#
# Concerto (Pointcept group, Sonata-derived) is a self-supervised Point
# Transformer V3 encoder. This image runs the two RELEASED capabilities:
#   - per-point feature EMBEDDINGS (the encoder output)
#   - closed-set SEMANTIC segmentation via the shipped ScanNet-20
#     linear-probe head (concerto_large_linear_prob_head_sc.pth)
#
# NOT included: the paper's open-world / CLIP-language ("zero-shot")
# path. It is absent from the released code and weights (no clip /
# open_clip / text-translator anywhere in the repo), so it is not
# reproducible from public artifacts and is not part of this image.
#
# Upstream code:    https://github.com/Pointcept/Concerto  (Apache-2.0)
# Upstream weights: https://huggingface.co/Pointcept/Concerto  (CC-BY-NC-4.0)
# Paper:            arXiv:2510.23607
#
# ## Build path — no compilation
#
# Unlike point-transformer-v3, Concerto is a STANDALONE package (it does
# not import pointcept.*), so there is no ocnn / pointops / pointgroup_ops
# and no Pointcept import-chain to satisfy. The only native deps are
# prebuilt wheels: spconv-cu121 and torch_scatter. FlashAttention (the one
# thing that would compile) is OPTIONAL — model.py guards `import
# flash_attn` and the encoder runs with enable_flash=False (upcast
# attention; slower, numerically fine). We skip it, so nothing compiles.
#
# ## Base image
#
# Reuses the proven pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel +
# spconv-cu121 base for parity with superpoint-transformer / PTv3.
# Upstream ships/tests torch 2.5.0/cu124; nothing in Concerto's code is
# 2.5-only (the loader explicitly branches for torch<2.4), so 2.2.2/cu121
# runs it. If a numerical issue ever surfaces, the documented fallback is
# pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel + spconv-cu124 +
# torch_scatter from the torch-2.5.0+cu124 wheel index.
#
# ## Scope (v1)
#
# Buildable environment + concerto_large weights + the ScanNet-20 probe
# head + a build-time smoke test that builds the model from the baked
# checkpoint. Ships experimental pending a Compute2 runtime check
# (embedding extraction + ScanNet mIoU via the probe head).

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/Concerto

# Minimal system deps — nothing compiles, so no build toolchain needed.
# libgomp1 covers scipy/torch OpenMP; git/wget for clone + checkpoint fetch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# numpy<2 pinned (open3d / torch 2.2 ABI); upstream pins numpy<=1.26.4.
RUN python -m pip install --upgrade pip && \
    python -m pip install "numpy==1.26.4"

# Native deps — both prebuilt wheels (no compile). spconv-cu121 ships
# sm_90 kernels (proven in the sibling images); torch_scatter from the
# torch-2.2.x+cu121 PyG index (the only PyG companion Concerto uses).
RUN python -m pip install spconv-cu121 && \
    python -m pip install torch-scatter \
        -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Concerto's pure-Python deps (no FlashAttention, no open3d/gradio demo
# stack — this is an inference image, not the visualization demos).
RUN python -m pip install huggingface_hub timm addict scipy einops natsort

# Concerto source, pinned. Run from source via PYTHONPATH (the package is
# /opt/Concerto/concerto); upstream offers "copy the concerto folder" as
# an install path, which avoids the deprecated setup.py install.
RUN git clone https://github.com/Pointcept/Concerto.git /opt/Concerto && \
    cd /opt/Concerto && \
    git checkout 10a7d17cff4dddff028f1522c2e72de4c4515df7 && \
    rm -rf .git

# Bake weights: concerto_large (834 MB encoder) + the ScanNet-20 linear-
# probe head (0.14 MB). Pulled from HF at build so the image is offline-
# usable. Each .pth is a {"config","state_dict"} dict.
RUN python -c "from huggingface_hub import hf_hub_download; \
[hf_hub_download('Pointcept/Concerto', f, local_dir='/opt/Concerto/checkpoints') \
 for f in ['concerto_large.pth', 'concerto_large_linear_prob_head_sc.pth']]" && \
    ls -lh /opt/Concerto/checkpoints

WORKDIR /opt/Concerto

# Build-time smoke test: versions via importlib.metadata, then BUILD the
# model from the baked checkpoint on CPU (config is baked into the .pth;
# enable_flash forced off since flash_attn is not installed) and load the
# state dict strictly. Also load the ScanNet probe head. No GPU needed —
# construction + load_state_dict run on CPU; the sm_90 path is the
# runtime test's job (spconv/torch_scatter are trusted prebuilt wheels).
RUN python -c "import importlib.metadata as im, torch; \
print('torch', torch.__version__, '| cuda', torch.version.cuda); \
print('spconv-cu121', im.version('spconv-cu121'), '| torch_scatter', im.version('torch-scatter')); \
print('numpy', im.version('numpy'), '| timm', im.version('timm')); \
import torch_scatter, spconv.pytorch; \
from concerto.model import PointTransformerV3; print('imported concerto PointTransformerV3'); \
ck = torch.load('/opt/Concerto/checkpoints/concerto_large.pth', map_location='cpu'); \
ck['config']['enable_flash'] = False; \
m = PointTransformerV3(**ck['config']); \
mk, un = m.load_state_dict(ck['state_dict'], strict=False); \
assert len(mk) == 0 and len(un) == 0, f'state_dict mismatch: {len(mk)} missing / {len(un)} unexpected'; \
print('concerto_large built; params:', sum(p.numel() for p in m.parameters())); \
hk = torch.load('/opt/Concerto/checkpoints/concerto_large_linear_prob_head_sc.pth', map_location='cpu'); \
print('ScanNet-20 probe head config:', hk['config'])"

LABEL org.opencontainers.image.title="concerto" \
      org.opencontainers.image.description="Concerto joint 2D-3D self-supervised PTv3 encoder (Pointcept), per-point embeddings + ScanNet-20 linear-probe semseg, H100 sm_90" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/concerto/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0 AND CC-BY-NC-4.0" \
      bradleylab.model.upstream="https://github.com/Pointcept/Concerto" \
      bradleylab.model.paper="https://arxiv.org/abs/2510.23607" \
      bradleylab.model.task="point cloud self-supervised embeddings + semantic segmentation" \
      bradleylab.model.sensor="lidar:terrestrial/aerial" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:Pointcept/Concerto (concerto_large + ScanNet probe), baked at build; CC-BY-NC-4.0"

# Encoder-only backbone — no turnkey CLI. See README for the embedding-
# extraction and ScanNet-probe-segmentation snippets.
CMD ["bash"]
