# dnabert-s — DNABERT-S species-aware DNA language model (MAGICS-LAB).
#
# DNA LM tuned for metagenomic binning: it produces 768-dimensional
# sequence embeddings in which contigs from the same species cluster
# together. Same architecture as DNABERT-2 (MosaicBERT with ALiBi, BPE
# vocab 4096, hidden 768, ~117M params), contrastively fine-tuned.
#
# Upstream:  https://github.com/MAGICS-LAB/DNABERT_S
#            Released 2024, effectively frozen (last push 2026-01-01).
#            The repo has NO LICENSE file — formally all rights reserved.
#            Nothing from it is vendored here; see README.md.
# Weights:   https://huggingface.co/zhihan1996/DNABERT-S
#            Apache-2.0 and ungated. 468 MB pytorch_model.bin (fp32).
#            The custom modelling code (bert_layers.py, flash_attn_triton.py)
#            lives in that same Apache-2.0 HF repo and is fetched at load
#            time via trust_remote_code=True — it is not baked here either.
# Stack:     nvidia/cuda 12.1 + Python 3.11 + PyTorch 2.5.1 cu121 +
#            transformers 4.27 (upstream's own pin).
#
# # Why this base, and why it is not the NGC image the other sequence LMs use
#
# transformers 4.27 is a hard requirement from upstream, and it drags the
# whole environment back to 2023:
#
#   - It constrains tokenizers to <0.14. tokenizers 0.13.x predates Python
#     3.12 and ships no cp312 wheel, so on a Python 3.12 base pip falls back
#     to a Rust source build. NGC PyTorch 25.04 (the base used by `esm` and
#     `evo2`) is Python 3.12, so it is out. Python 3.11 also still ships
#     distutils, which 3.12 removed and code of this vintage still reaches
#     for.
#   - torch 2.6 flipped the default of torch.load() to weights_only=True.
#     transformers 4.27 calls torch.load() with no such argument when it
#     reads a .bin checkpoint, and DNABERT-S ships a .bin, not safetensors.
#     Pinning torch 2.5.1 keeps that question from arising at all.
#
# So: the CUDA 12.1 + Python 3.11 + torch 2.5.1 stack already used by
# prithvi-eo / satlas / clay / terramind, which is H100-capable (sm_90) and
# old enough to keep transformers 4.27 happy. A CPU-only base would run this
# model fine — it is 0.5 GB of weights — but the reason to containerise it
# is batch-embedding metagenome contigs, and that is a GPU job.

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
        wget ca-certificates \
    && wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.11 /tmp/get-pip.py && rm /tmp/get-pip.py \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.5.1 cu121 — H100 sm_90; matches prithvi-eo / satlas / terramind.
# Version pinned deliberately, not for consistency alone: see the torch.load
# note in the header.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1

# transformers 4.27 is upstream's pin (DNABERT_S requirements). The 4.27
# series has had no release since April 2023, so `4.27.*` is effectively
# frozen; it is written as a wildcard rather than a patch number so pip
# takes the last patch of a dead series.
#
# huggingface_hub <0.26 is not upstream's pin — it is ours, and it is
# load-bearing. huggingface_hub 0.26 removed `cached_download`, which
# transformers 4.27 imports at package-import time; with a floating hub
# version `import transformers` raises ImportError before any model is
# touched.
#
# einops is imported by the MosaicBERT modelling code that
# trust_remote_code fetches from the checkpoint repo. numpy is held below
# 2.0 because everything pinned here predates the numpy 2 API break.
RUN pip install \
        "transformers==4.27.*" \
        "huggingface_hub>=0.16,<0.26" \
        "einops>=0.6" \
        "numpy<2"

# Remove triton. Upstream advises `pip uninstall triton` on anything other
# than an A100: the repo's triton flash-attention kernel misbehaves on other
# GPUs, and without triton the remote code falls back to standard attention.
# We run on H100, so triton comes out.
#
# This has to be the LAST pip step in the file. torch declares triton as a
# Linux dependency, so any later `pip install` can pull it straight back in.
# The smoke test below asserts it is really gone.
#
# Cost of removing it: torch.compile / TorchInductor are unavailable in this
# image. Nothing in the DNABERT-S inference path uses them.
RUN pip uninstall -y triton

# Weights are NOT baked — fetched from zhihan1996/DNABERT-S on first
# from_pretrained() call. Point HF_HOME at a bind-mounted scratch dir at
# runtime so the checkpoint and its remote code persist across jobs.
ENV HF_HOME=/root/.cache/huggingface

# Build-time smoke test. Offline: no weight fetch, no network, no GPU.
# Versions come from package metadata per the ml-containers rule, never
# pkg.__version__. Unlike esm / evo2 this image CAN import its stack on a
# CPU runner — nothing here needs a device at import time — so the test
# resolves the real Auto* classes rather than only reading metadata.
#
# The triton assertion is the point of this test as much as the imports
# are: a silent triton reinstall would put the model back on the kernel
# upstream tells us not to use on non-A100 hardware, and would show up as
# wrong numbers rather than as an error.
RUN python -c "import importlib.metadata as im, importlib.util, sys, torch; \
print('python', sys.version.split()[0], \
'| transformers', im.version('transformers'), \
'| tokenizers', im.version('tokenizers'), \
'| huggingface_hub', im.version('huggingface_hub'), \
'| einops', im.version('einops'), \
'| numpy', im.version('numpy'), \
'| torch', torch.__version__, '| cuda-build', torch.version.cuda); \
assert im.version('transformers').startswith('4.27.'), im.version('transformers'); \
assert importlib.util.find_spec('triton') is None, 'triton is installed — remote code would take the triton flash-attn kernel'; \
import transformers, einops; \
from transformers import AutoConfig, AutoModel, AutoTokenizer; \
assert callable(AutoModel.from_pretrained); \
assert callable(AutoTokenizer.from_pretrained); \
assert callable(AutoConfig.from_pretrained); \
print('SMOKE OK')"

WORKDIR /workspace

LABEL org.opencontainers.image.title="dnabert-s" \
      org.opencontainers.image.description="DNABERT-S (MAGICS-LAB) species-aware DNA language model — 768-d contig embeddings for metagenomic binning; CUDA 12.1 + torch 2.5.1 + transformers 4.27 (upstream pin), triton removed for non-A100 GPUs; H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/dnabert-s/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/MAGICS-LAB/DNABERT_S" \
      bradleylab.model.task="DNA language model, species-aware sequence embedding (768-d, mean pooling) for metagenomic binning" \
      bradleylab.model.sensor="dna-sequence" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:zhihan1996/DNABERT-S (runtime fetch; Apache-2.0, ungated)" \
      bradleylab.model.weights.license="Apache-2.0 (HF model card)" \
      bradleylab.model.code.license="MAGICS-LAB/DNABERT_S ships no LICENSE file (formally all rights reserved); no code from that repo is vendored in this image" \
      bradleylab.build.note="triton uninstalled — upstream advises removing it on non-A100 GPUs; torch.compile unavailable"
