# poseidon — foundation models for solving partial differential equations.
#
# Herde, Raonić, Rohner et al. (NeurIPS 2024), ETH Zurich CAMLab. A scalable
# Operator Transformer (scOT) with time-conditioned layer normalization,
# pretrained on families of PDEs -- compressible Euler, Navier-Stokes, wave,
# Poisson, Helmholtz -- and intended to be finetuned onto a downstream
# operator with far fewer samples than training from scratch.
#
# What it predicts is a solution field, not a scalar: given an input state
# (and a time, for time-dependent problems) it returns the solved field.
#
# ---------------------------------------------------------------------
# THE LICENSE SITUATION, WHICH IS UNUSUAL AND MUST NOT BE SMOOTHED OVER.
# ---------------------------------------------------------------------
# The WEIGHTS are clearly licensed: CC-BY-NC-4.0, ungated, on the Hugging
# Face Hub. Non-commercial, like sonata / concerto / utonia here.
#
# The CODE is not licensed at all. There is no LICENSE file anywhere in
# camlab-ethz/poseidon (whole tree checked at b8fa28f, 2026-09-21) and the
# README states no terms. Absent a license, no permission to use, modify or
# redistribute has formally been granted, and this image necessarily
# redistributes the code because scOT has to be installed to run.
#
# Against that: the project page states, under "Usage", "We encourage using
# our pretrained models on your own datasets. To that end, you can directly
# plug your dataset into our code and then finetune using our scripts." The
# paper describes the models and datasets as open sourced, and the weights
# carry a deliberate license tag. The missing file reads as an oversight
# rather than a withholding. That is a judgment this lab made, written down
# here so the next reader inherits the reasoning and not just the outcome.
#
# Consequences, standing until upstream adds a license:
#   - Non-commercial only, from the weights.
#   - This image is NOT listed on the public model catalog, the same
#     treatment forainet and backman-thermal-deer get for the same reason.
#   - If upstream adds a license, revisit both the label and that decision.
#
# ---------------------------------------------------------------------
# torch 2.0.1 IS PINNED UPSTREAM, AND THAT IS AN H100 TRAP.
# ---------------------------------------------------------------------
# scOT's pyproject pins torch == 2.0.1. The default wheel for that version
# is cu117, which predates sm_90: crossearth shipped exactly that way and
# HUNG on an H100 rather than failing. CUDA 11.8 is the first release with
# Hopper support, and torch 2.0.1+cu118 exists -- so the pin is honoured and
# the GPU is still reachable. The build asserts the CUDA version rather than
# torch.cuda.get_arch_list(), which is empty on a driver-less CI runner.
#
# scOT is installed --no-deps for the same reason: its bare `torch == 2.0.1`
# requirement would otherwise resolve to the cu117 wheel and silently replace
# the one installed above. Its dependencies are installed explicitly here.
#
# Upstream: https://github.com/camlab-ethz/poseidon @ b8fa28f (NO LICENSE)
# Weights:  https://huggingface.co/camlab-ethz/Poseidon-{T,B,L}
#           CC-BY-NC-4.0, ungated. T (83 MB) is baked; B (1.26 GB) and
#           L (5.03 GB) are fetched at runtime into a mounted cache.
# Paper:    https://arxiv.org/abs/2405.19101

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# enroot exposes the GPU only when the image asks for it. On a CPU partition
# override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the operator's ~/.local off sys.path: enroot
# bind-mounts $HOME, and a host package shadowing an image one has broken
# images in this catalog before.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    POSEIDON_MODEL_ID=camlab-ethz/Poseidon-T \
    WANDB_MODE=disabled

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# cu118, not the default cu117 that torch 2.0.1 otherwise resolves to.
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
        "torch==2.0.1+cu118" \
        "torchvision==0.15.2+cu118"

# scOT's own dependency pins, installed before the package so that
# --no-deps below leaves nothing missing. numpy is held under 2 for the
# torch 2.0 ABI.
RUN python -m pip install \
        "transformers==4.29.2" \
        "accelerate==0.31.0" \
        "wandb==0.14.2" \
        "numpy<2" \
        "h5py" \
        "pandas" \
        "pyyaml" \
        "matplotlib" \
        "safetensors>=0.4"

ARG POSEIDON_REF=b8fa28f59bd7f7673323f28d11a12c6f3a215c61
RUN git clone https://github.com/camlab-ethz/poseidon.git /opt/poseidon \
    && cd /opt/poseidon \
    && git checkout "$POSEIDON_REF" \
    && python -m pip install --no-deps -e .

# Poseidon-T baked: small enough to make the image self-sufficient and to let
# the build smoke test load a real checkpoint offline. B and L stay out --
# together they are 6.3 GB, and both fetch at runtime from the same ungated
# repository into whatever cache HF_HOME points at.
RUN python -c "\
from huggingface_hub import snapshot_download; \
p = snapshot_download('camlab-ethz/Poseidon-T', allow_patterns=['*.json', '*.safetensors']); \
print('baked:', p)"

# Build smoke test — offline, no fetch. Loads the baked checkpoint so an
# install that imports but cannot build the model fails here rather than on a
# compute node.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im; \
import torch; \
from scOT.model import ScOT; \
print('scOT', im.version('scOT'), '| torch', im.version('torch'), '| transformers', im.version('transformers')); \
print('cuda build:', torch.version.cuda); \
assert torch.version.cuda is not None, 'CUDA torch was replaced by a CPU wheel'; \
assert tuple(int(p) for p in torch.version.cuda.split('.')[:2]) >= (11, 8), \
    f'sm_90 needs CUDA >= 11.8, got cu{torch.version.cuda} -- the cu117 default cannot target an H100'; \
m = ScOT.from_pretrained('camlab-ethz/Poseidon-T'); \
n = sum(p.numel() for p in m.parameters()); \
print('Poseidon-T parameters:', f'{n:,}'); \
assert n > 1_000_000, n; \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="poseidon" \
      org.opencontainers.image.description="Poseidon — ETH CAMLab foundation models for PDEs (scalable Operator Transformer); predicts solution fields for Euler/Navier-Stokes/wave/Poisson/Helmholtz families and finetunes onto downstream operators. Poseidon-T baked; B and L fetched at runtime" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/poseidon/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="NOASSERTION" \
      bradleylab.model.upstream="https://github.com/camlab-ethz/poseidon" \
      bradleylab.model.paper="https://arxiv.org/abs/2405.19101" \
      bradleylab.model.task="PDE solution-operator learning — predicts solution fields, finetunable onto a downstream operator" \
      bradleylab.model.sensor="simulation — gridded PDE solution fields" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="required" \
      bradleylab.model.weights.source="huggingface-hub:camlab-ethz/Poseidon-T baked (CC-BY-NC-4.0); Poseidon-B and -L fetched at runtime from the same ungated repository" \
      bradleylab.model.licence.note="CODE HAS NO LICENSE — no LICENSE file anywhere in camlab-ethz/poseidon at b8fa28f, and the README states no terms, so no permission is formally granted. WEIGHTS are CC-BY-NC-4.0, non-commercial. The project page encourages using the pretrained models on your own datasets, which is why this was built; it is not a license. Not listed on the public catalog. Revisit if upstream adds a LICENSE"

WORKDIR /work
CMD ["python", "-c", "print('poseidon ready — see README.md; scOT.inference is the CLI')"]
