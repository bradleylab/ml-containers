# syntax=docker/dockerfile:1
# forpt — ForPT, a self-supervised foundation model for forest lidar point clouds.
#
# ForPT pretrains a LitePT-S backbone (12.4 M parameters) on unlabeled airborne,
# UAV and mobile laser scans of forests, then fine-tunes it for semantic
# segmentation (ground / wood / leaf, FOR-instanceV2), tree species
# classification (33 species, FOR-species20K) and tree age regression (FORage).
# This image holds the pretrained encoder and the three fully fine-tuned models,
# with two commands: forpt-semseg labels every point of a LAS/LAZ file, and
# forpt-species names the species of single-tree clouds.
#
# Upstream code:    https://github.com/prs-eth/ForPT @ 8304f57 (MIT)
# Upstream weights: https://huggingface.co/prs-eth/ForPT @ 6b9047e (no license
#                   stated; treated as covered by the code's MIT license)
# Paper:            Yue et al., Toward a foundation model for forest point
#                   clouds, arXiv:2609.24787
#
# LitePT's attention runs only through FlashAttention (no fallback in
# litept_v1m2_pretrain.py), so inference needs the GPU. The devel base is for
# compiling upstream's two CUDA extensions, pointops and PointROPE; torch 2.4.1
# with CUDA 12.4 is the stack LitePT's install instructions name.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_MODE=disabled \
    TORCH_CUDA_ARCH_LIST="9.0" \
    PYTHONPATH=/opt/forpt/src:/opt/forpt/lib

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# --no-deps: the lock file lists every package, so pip cannot swap the base
# image's torch for another build. pip check then proves the set is complete.
COPY requirements.lock.txt /opt/forpt/requirements.lock.txt
RUN pip install --no-deps -r /opt/forpt/requirements.lock.txt \
        --find-links https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    && pip check

# Prebuilt FlashAttention for torch 2.4, CUDA 12, CPython 3.11; building it from
# source does not fit a GitHub-hosted runner.
ARG FLASH_ATTN_WHEEL=flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
ARG FLASH_ATTN_SHA256=92cf49e6f66795b6934cec0cba526ed6e45d3313de3f905d45df8773f19092a9
RUN curl -fsSL -o "/tmp/${FLASH_ATTN_WHEEL}" \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$(echo "${FLASH_ATTN_WHEEL}" | sed 's/+/%2B/')" \
    && echo "${FLASH_ATTN_SHA256}  /tmp/${FLASH_ATTN_WHEEL}" | sha256sum -c - \
    && pip install --no-deps "/tmp/${FLASH_ATTN_WHEEL}" \
    && rm "/tmp/${FLASH_ATTN_WHEEL}"

ARG FORPT_SHA=8304f57620fdffc35ac9a203ee844a015a875869
RUN git clone https://github.com/prs-eth/ForPT.git /opt/forpt/src \
    && cd /opt/forpt/src \
    && git checkout "${FORPT_SHA}" \
    && rm -rf .git libs/pointops/dist

# Both extensions compile for sm_90 from TORCH_CUDA_ARCH_LIST; there is no GPU
# on the build runner. forpt.datasets imports pointops at import time, and
# PointROPE replaces upstream's slower pure-PyTorch rotary embedding.
RUN cd /opt/forpt/src/libs/pointops && pip install --no-deps --no-build-isolation . \
    && cd /opt/forpt/src/libs/pointrope && pip install --no-deps --no-build-isolation . \
    && rm -rf /opt/forpt/src/libs/pointops/build /opt/forpt/src/libs/pointrope/build

COPY scripts/forpt_lib.py /opt/forpt/lib/
COPY scripts/forpt-fetch-weights scripts/forpt-semseg scripts/forpt-species scripts/forpt-selftest /usr/local/bin/

RUN forpt-fetch-weights && rm -rf /root/.cache/huggingface
ENV HF_HUB_OFFLINE=1

RUN --network=none python -c "import importlib.metadata as im, torch; \
import spconv.pytorch, torch_scatter, flash_attn, pointops, pointrope, laspy, forpt.models, forpt.datasets; \
print('torch', torch.__version__, '| cuda-build', torch.version.cuda); \
print(' | '.join(f'{p} {im.version(p)}' for p in ['spconv-cu124', 'torch-scatter', 'flash-attn', 'timm', 'numpy', 'laspy', 'lazrs', 'pointops', 'pointrope']))"

# Builds every model from upstream's configs, loads the baked weights strictly
# and runs the test transforms on a synthetic tree, CPU only, network disabled.
RUN --network=none NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES="" forpt-selftest

WORKDIR /work

LABEL org.opencontainers.image.title="forpt" \
      org.opencontainers.image.description="ForPT self-supervised LitePT-S foundation model for forest lidar: ground/wood/leaf semantic segmentation of LAS/LAZ and 33-species tree classification, plus the pretrained encoder and the age regressor" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/forpt/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/prs-eth/ForPT" \
      bradleylab.model.paper="https://arxiv.org/abs/2609.24787" \
      bradleylab.model.task="forest point cloud semantic segmentation (ground, wood, leaf), tree species classification (33 species), tree age regression, and self-supervised point features" \
      bradleylab.model.sensor="lidar:airborne/uav/mobile forest scans" \
      bradleylab.model.status="experimental" \
      bradleylab.model.domain="forestry" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.weights.source="baked: huggingface:prs-eth/ForPT @ 6b9047e — pretrain-forpt epoch 30 (student backbone), semseg-forpt-ft-p100, cls-forpt-ft-allpts, age-forpt-ft-p100; SHA-256-checked at build, reduced to state dicts" \
      bradleylab.model.weights.license="MIT (code repository license; the Hugging Face repository states none)" \
      bradleylab.model.licence.note="The Hugging Face weights repository states no license. The weights are treated as covered by the MIT license of the upstream code repository."

CMD ["bash"]
