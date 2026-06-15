# Evo2 — Arc Institute genomic foundation model
# DNA language model (StripedHyena 2) for variant-effect prediction,
# sequence scoring, and genome generation across all domains of life.
#
# Upstream:  https://github.com/ArcInstitute/evo2  (code Apache-2.0)
# Weights:   https://huggingface.co/arcinstitute  (evo2_7b, evo2_40b, ...)
# Stack:     NGC PyTorch 25.04 (torch 2.7.0a0, CUDA 12.9, Transformer
#            Engine 2.2, Python 3.12) + the evo2 pip package.
#
# GPU notes: the 7B models run in bf16 on any recent GPU; 20B/40B require
# FP8 via Transformer Engine on a Hopper GPU (H100), and 40B needs
# multiple H100s. C2 general-gpu nodes are single H100 80GB, so 7B / 1B /
# *_base fit; the 40B model does not (multi-GPU). sm_90 is covered by the
# base image's TORCH_CUDA_ARCH_LIST.
#
# Backfilled recipe: reproduces the previously ad-hoc
# ghcr.io/bradleylab/evo2 image (NGC pytorch:25.04 + the two pip installs
# below), reconstructed from the published image's build history.

FROM nvcr.io/nvidia/pytorch:25.04-py3

# Evo2 + biopython. NGC 25.04 already ships PyTorch 2.7, CUDA 12.9,
# Transformer Engine 2.2 and flash-attn, which evo2 builds against — so
# no separate transformer-engine / flash-attn install is needed here.
RUN pip install --no-cache-dir evo2 biopython

# Weights are NOT baked — fetched from the arcinstitute HF org on first
# use (e.g. Evo2('evo2_7b')). Point HF_HOME at a bind-mounted scratch dir
# at runtime so checkpoints persist across jobs.
ENV HF_HOME=/root/.cache/huggingface

# Build-time smoke test: confirm the install. Version is read from package
# metadata only — evo2 / transformer_engine require a GPU at import time,
# which the CPU build runner lacks, so we do NOT `import evo2` here.
RUN python -c "import importlib.metadata as im, torch; print('evo2', im.version('evo2'), '| biopython', im.version('biopython'), '| torch', torch.__version__)"

WORKDIR /workspace

LABEL org.opencontainers.image.title="evo2" \
      org.opencontainers.image.description="Evo2 (Arc Institute) genomic foundation model — DNA LM for variant scoring and generation; NGC pytorch 25.04 + evo2 pip; H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/evo2/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/ArcInstitute/evo2" \
      bradleylab.model.task="genomic foundation model, DNA sequence scoring + generation" \
      bradleylab.model.sensor="dna-sequence" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:arcinstitute (runtime fetch)"
