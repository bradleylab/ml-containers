# RemoteCLIP — vision-language foundation model for remote sensing.
#
# Liu, Chen et al. (2024, IEEE TGRS) — CLIP architecture fine-tuned
# on a 12× larger remote-sensing pre-training corpus assembled via
# Box-to-Caption + Mask-to-Box conversion (RET-3, SEG-4, DET-10) plus
# UAV imagery. Three OpenCLIP-format checkpoints are distributed via
# Hugging Face Hub: RN50, ViT-B-32, ViT-L-14.
#
# RemoteCLIP is a checkpoint, not a package. This container bundles
# OpenCLIP (the model architecture) and huggingface_hub (the weight
# fetcher); checkpoints are pulled at runtime into HF_HOME so the
# build is deterministic and offline.
#
# Upstream: https://github.com/ChenDelong1999/RemoteCLIP
# Paper:    https://arxiv.org/abs/2306.11029  (Liu, Chen et al. 2024)
# Weights:  https://huggingface.co/chendelong/RemoteCLIP
# Stack:    python:3.11-slim + PyTorch 2.5 (cu121 on amd64, CPU on arm64) + open-clip-torch
#
# Multi-arch: linux/amd64 + linux/arm64. Every dependency publishes an
# aarch64 wheel (torch 2.5.1, torchvision 0.20.1, open-clip-torch is
# pure Python, Pillow + huggingface_hub all ship aarch64 wheels), so
# no C toolchain is required in the image (verified against PyPI +
# download.pytorch.org/whl/cpu, 2026-05-02).
#
# Bind-mount the HF cache to /opt/hf-cache so the checkpoint only
# downloads once per host (CLIP-sized: RN50 ~400 MB, ViT-L-14 ~1.7 GB).

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Torch per architecture. The amd64 image carries CUDA torch so batch
# embedding on Compute2 uses the H100 -- the 2026-09-01 harness found this
# image executing a pure-GPU workload on CPU while holding a GPU allocation.
# The arm64 image keeps CPU torch: the cu121 index has no aarch64 wheels, and
# the laptop use this multi-arch build exists for is CPU anyway. Both wheels
# are the same torch version, so the model code is identical across arches.
ARG TARGETARCH
RUN pip install --upgrade pip wheel \
    && if [ "$TARGETARCH" = "amd64" ]; then \
         pip install --index-url https://download.pytorch.org/whl/cu121 \
             torch==2.5.1 torchvision==0.20.1; \
       else \
         pip install --index-url https://download.pytorch.org/whl/cpu \
             torch==2.5.1 torchvision==0.20.1; \
       fi

RUN pip install \
        "open-clip-torch>=2.20" \
        "huggingface_hub>=0.25.0" \
        "Pillow>=10.0"

RUN mkdir -p /opt/hf-cache

# Smoke test: import + reach the public OpenCLIP API + huggingface_hub.
# No weight fetch at build time. importlib.metadata for portable
# version checks (see ~/.claude/rules/ml-containers.md).
ARG TARGETARCH
RUN python -c "import torch, open_clip, huggingface_hub, platform; \
import importlib.metadata as im; \
print('open-clip-torch', im.version('open-clip-torch'), '| torch', torch.__version__, '| cuda-build', torch.version.cuda, '| huggingface_hub', huggingface_hub.__version__); \
assert (torch.version.cuda is not None) == (platform.machine() == 'x86_64'), f'amd64 must carry CUDA torch and arm64 must not: cuda={torch.version.cuda} machine={platform.machine()}'; \
assert 'sm_90' in torch.cuda.get_arch_list() if torch.version.cuda else True, torch.cuda.get_arch_list(); \
assert callable(open_clip.create_model_and_transforms); \
assert callable(open_clip.get_tokenizer); \
assert callable(huggingface_hub.hf_hub_download)"

LABEL org.opencontainers.image.title="remoteclip" \
      org.opencontainers.image.description="RemoteCLIP — CLIP architecture fine-tuned on remote-sensing imagery (RN50 / ViT-B-32 / ViT-L-14); CUDA torch on amd64, CPU torch on arm64; weights via Hugging Face Hub at chendelong/RemoteCLIP" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/remoteclip/README.md" \
      org.opencontainers.image.version="v2" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/ChenDelong1999/RemoteCLIP" \
      bradleylab.model.paper="https://arxiv.org/abs/2306.11029" \
      bradleylab.model.task="remote-sensing image-text retrieval / zero-shot scene classification" \
      bradleylab.model.sensor="image:rgb-satellite (Sentinel-2, NAIP, UAV — anything CLIP ingests as RGB)" \
      bradleylab.model.status="utility" \
      bradleylab.model.gpu="optional — amd64 image carries CUDA torch (sm_90); arm64 image is CPU-only by design" \
      bradleylab.model.weights.source="huggingface-hub:chendelong/RemoteCLIP (RemoteCLIP-{RN50,ViT-B-32,ViT-L-14}.pt; runtime fetch into HF_HOME=/opt/hf-cache)"

WORKDIR /work
CMD ["bash"]
