# ntv3 — Nucleotide Transformer v3 (InstaDeep).
#
# Genomic language model reading up to 1 Mb at nucleotide resolution. The
# post-trained checkpoints emit ~16,000 functional genomic tracks across 24
# species (the BigWig-style signal set) plus base-resolution annotation, on
# top of the usual sequence embeddings.
#
# Upstream:  https://github.com/instadeepai/nucleotide-transformer
#            (docs/nucleotide_transformer_v3.md). Released Dec 2025,
#            actively maintained. That repo is JAX; this image takes the
#            HF PyTorch route instead — see the note below.
# Weights:   https://huggingface.co/InstaDeepAI/NTv3_650M_post — 650M
#            params, 2.72 GB fp32 safetensors, embedding dim 1536.
# Stack:     NGC PyTorch 25.04 (torch 2.7.0a0, CUDA 12.9, Python 3.12) +
#            transformers >= 4.55.
#
# # IMPORTANT — LICENCE AND GATING
#
# Weights: **InstaDeep non-commercial licence.** No commercial use, and no
# training of competing models on the outputs. Code: CC BY-NC-SA 4.0.
# University research is in scope; this image must not be used for
# commercial work. See the use_restriction label at the bottom.
#
# The HF repo is **gated** — an account must accept the terms, and the
# download needs an HF token. That token is NEVER baked into this image and
# never appears in this file. Stage the checkpoint from a login node with
# the token supplied from the environment at that moment, then run jobs
# offline against the cache. README.md has the procedure; the smoke test
# below asserts no HF_TOKEN was captured into the image.
#
# # Why this base
#
# The same NGC PyTorch 25.04 image as `esm` and `evo2`, the lab's other
# large sequence models: torch 2.7 with CUDA 12.9 and flash-attn already
# built, Python 3.12, and sm_90 in TORCH_CUDA_ARCH_LIST. Nothing in the NTv3
# stack pulls the environment backwards the way DNABERT-S's transformers 4.27
# pin does, so there is no reason to deviate. bf16 on H100 is upstream's
# recommendation and this base supports it natively.

FROM nvcr.io/nvidia/pytorch:25.04-py3

# transformers >= 4.55 is upstream's floor: the checkpoint declares a custom
# `ntv3_posttrained` architecture and ships its own pipeline modules
# (ntv3_tracks_pipeline.py, ntv3_interpret_pipeline.py) that are loaded via
# trust_remote_code. Written as a floor-plus-ceiling range rather than an
# exact pin because the remote code tracks recent transformers and the
# checkpoint is only months old; the ceiling stops a transformers 5.x from
# arriving unannounced. To make a build byte-reproducible, pin an exact
# version here and commit it, so the pin lives in git history.
RUN pip install --no-cache-dir "transformers>=4.55,<5"

# Weights are NOT baked — the repo is gated, so a build-time fetch would
# require a token in the image. Point HF_HOME at a bind-mounted scratch dir
# holding a cache staged from a login node (see README.md).
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

# Build-time smoke test. Offline: no weight fetch (it would need a token
# anyway), no network, no GPU. Versions come from package metadata per the
# ml-containers rule, never pkg.__version__.
#
# Unlike esm / evo2 this does import its stack for real — transformers and
# torch both import fine on a CPU runner. What it cannot reach is the model
# class itself, which lives in remote code inside a gated repo and only
# materialises at from_pretrained() time on a node with the staged cache.
#
# The HF_TOKEN assertion is a guard, not a formality: it fails the build if
# a token ever gets introduced as a build-arg or ENV, which is the way a
# credential would end up published to GHCR.
RUN python -c "import importlib.metadata as im, os, torch; \
from packaging.version import Version; \
tv = im.version('transformers'); \
print('transformers', tv, \
'| tokenizers', im.version('tokenizers'), \
'| huggingface_hub', im.version('huggingface_hub'), \
'| safetensors', im.version('safetensors'), \
'| torch', torch.__version__, '| cuda-build', torch.version.cuda); \
assert Version(tv) >= Version('4.55'), tv; \
import transformers; \
from transformers import AutoConfig, AutoModel, AutoTokenizer; \
assert callable(AutoModel.from_pretrained); \
assert callable(AutoTokenizer.from_pretrained); \
assert callable(AutoConfig.from_pretrained); \
assert not os.environ.get('HF_TOKEN'), 'HF_TOKEN is set at build time — a gated-repo credential must never be baked into this image'; \
assert not os.environ.get('HUGGING_FACE_HUB_TOKEN'), 'HUGGING_FACE_HUB_TOKEN is set at build time — see above'; \
print('SMOKE OK')"

WORKDIR /workspace

LABEL org.opencontainers.image.title="ntv3" \
      org.opencontainers.image.description="Nucleotide Transformer v3 (InstaDeep) — genomic LM to 1 Mb at nucleotide resolution; embeddings, ~16k functional tracks (24 species), base-resolution annotation. NGC pytorch 25.04 + transformers>=4.55, HF PyTorch route. WEIGHTS NON-COMMERCIAL and HF-GATED (token needed to stage weights; never baked). H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/ntv3/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="LicenseRef-InstaDeep-NTv3-NonCommercial" \
      bradleylab.model.upstream="https://github.com/instadeepai/nucleotide-transformer" \
      bradleylab.model.task="genomic language model — sequence embeddings (1536-d), ~16,000 functional genomic tracks across 24 species, base-resolution annotation" \
      bradleylab.model.sensor="dna-sequence" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:InstaDeepAI/NTv3_650M_post (GATED — accept terms + HF token at staging time; runtime fetch only, never baked)" \
      bradleylab.model.weights.license="InstaDeep NTv3 non-commercial licence (weights); CC-BY-NC-SA-4.0 (upstream code)" \
      bradleylab.model.use_restriction="NON-COMMERCIAL RESEARCH ONLY — no commercial use; no training of competing models on this model's outputs" \
      bradleylab.model.input.constraint="input length must be a multiple of 128 bp and padded with N (not [PAD]); post-trained track outputs are cropped to the middle 62.5% of the window"
