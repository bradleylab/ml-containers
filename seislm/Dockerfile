# syntax=docker/dockerfile:1
# seislm — SeisLM, a self-supervised foundation model for seismic waveforms
# (Liu, Münchmeyer, Laurenti, Marone, de Hoop and Dokmanić, 2024).
#
# A Wav2Vec2-style encoder for single-station, 3-component waveforms at 100 Hz:
# two strided convolutions, a convolutional position embedding and pre-norm
# transformer blocks, pretrained with a masked contrastive loss against
# quantized features on the unlabeled training folds of eight SeisBench datasets
# (ETHZ, INSTANCE, Iquique, STEAD, GEOFON, MLAAPDE, PNW, OBST2024).
# SeisLM-base has 6 blocks and 11.4 M parameters, SeisLM-large 12 blocks and
# 90.7 M. Fine-tuned, it is evaluated on event detection, phase identification
# and onset regression (ETHZ, GEOFON, STEAD) and on foreshock-aftershock
# classification at station NRCA in the 2016 Norcia sequence.
#
# ---------------------------------------------------------------------
# WASHU-INTERNAL. WEIGHTS ARE NOT BAKED.
# ---------------------------------------------------------------------
# The upstream repository has no LICENSE file (GitHub reports none); its
# setup.py metadata says MIT, but no license text accompanies the code, and the
# checkpoints, shared from a Google Drive folder, carry no license at all. The
# image carries the code for WashU-internal research use. The checkpoints are
# held privately by the lab (NAS ml_model_weights/seislm, staged on Storage3 at
# Active/seislm/weights), mounted at /weights at run time, and never fetched
# into the image. The model may be listed publicly, marked WashU-only; the
# weights are never published.
#
# Upstream: https://github.com/liutianlin0121/seisLM @ ae1d27a (no license file)
# Weights:  Google Drive folder linked from upstream's README (no license)
# Paper:    https://arxiv.org/abs/2410.15765

FROM python:3.12-slim-bookworm

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the user's ~/.local off sys.path: enroot bind-mounts
# $HOME, and a host package shadowing an image one has broken images in this
# catalog before. HF_HOME and SEISBENCH_CACHE_ROOT are deliberately absent:
# pyxis lets image ENV beat srun --export, so jobs set them in their command.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# git: upstream's seisLM.utils.project_path locates its data and results
# directories with `git rev-parse --show-toplevel` when it is imported.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# Upstream's README targets Python 3.9 with pytorch-cuda 11.8 and pins no
# versions. Nothing in the code needs either: its imports are torch, Lightning,
# ml_collections, SeisBench, einops, scikit-learn, torchmetrics, GitPython and
# torchtune (transformers appears only in its unit tests, wandb only in its
# training scripts), and both checkpoints load and run on torch 2.14 with
# Lightning 2.6. torch 2.14.0 is the CUDA 13.0 build satclip uses; 13.0 is the
# newest CUDA Compute2's driver (580.105.08) supports, and the release build
# compiles sm_90 kernels for the H100. torch gets a layer of its own, the
# multi-GB one, so a failure in the next step rebuilds without downloading it.
COPY constraints.txt /opt/constraints.txt
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install -c /opt/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 torch

# torchtune is here for one symbol: upstream imports torchtune.modules.RMSNorm
# at module level, although neither released checkpoint enables RMSNorm
# (use_rms_norm is false in both). Installing it keeps upstream's code
# unmodified; it brings Hugging Face datasets and torchao with it, which
# constraints.txt pins to versions that import. gdown downloads the NRCA data
# from Google Drive, as upstream's demo notebook does.
RUN python -m pip install -c /opt/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        torchtune torchao lightning ml-collections seisbench einops \
        gitpython scikit-learn torchmetrics pandas gdown

# Upstream's setup.py declares no package data, so a pip install would leave
# the configs behind; the tree goes to /opt/seislm and onto PYTHONPATH instead,
# as `pip install -e .` in upstream's README would. The tarball, not a clone,
# keeps upstream's history out of the image.
ARG SEISLM_GIT_SHA=ae1d27ac498ec7ae106e56eeaac386a0a5dd124f
RUN mkdir -p /opt/seislm \
    && curl -fsSL "https://github.com/liutianlin0121/seisLM/archive/${SEISLM_GIT_SHA}.tar.gz" \
        | tar -xz --strip-components=1 -C /opt/seislm \
    && rm -rf /opt/seislm/img /opt/seislm/notebooks \
    && echo "${SEISLM_GIT_SHA}" > /opt/seislm/UPSTREAM_SHA \
    && python -m compileall -q /opt/seislm/seisLM
ENV SEISLM_HOME=/opt/seislm \
    PYTHONPATH=/opt/seislm:/opt/seislm-lab

COPY scripts/seislm_checkpoints.py /opt/seislm-lab/seislm_checkpoints.py
COPY scripts/seislm-selftest scripts/seislm-verify-shock /usr/local/bin/
RUN mkdir -p /weights /work

# Build smoke test, with no network so nothing can be fetched. torch imports on
# its own under faulthandler so a compiled-extension crash is attributed to it.
RUN --network=none python -X faulthandler -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda); assert torch.version.cuda and torch.version.cuda.startswith('13.0'), 'CUDA 13.0 torch was replaced'"
RUN --network=none python -X faulthandler -c "\
import importlib.metadata as im; \
import torchtune.modules, lightning, seisbench, ml_collections; \
import seisLM.model.foundation.pretrained_models, seisLM.model.task_specific.foreshock_aftershock_models; \
print(' | '.join(f'{p} {im.version(p)}' for p in ('torchtune', 'torchao', 'datasets', 'lightning', 'seisbench', 'ml-collections', 'pandas', 'gdown')))"
# Both architectures, randomly initialized from the upstream pretraining configs
# they were trained with: parameter counts, feature shapes and finite output on
# a synthetic 30 s window.
RUN --network=none HF_HOME=/tmp/hf python -X faulthandler /usr/local/bin/seislm-selftest --random-init
# The foreshock-aftershock pipeline end to end on synthetic data: upstream's
# data loaders, classifier, fine-tuning for one epoch on the CPU, test on
# last.ckpt and the confusion matrix, from a randomly initialized SeisLM-base.
RUN --network=none HF_HOME=/tmp/hf python -X faulthandler /usr/local/bin/seislm-verify-shock selfcheck \
    && rm -rf /tmp/hf

LABEL org.opencontainers.image.title="seislm" \
      org.opencontainers.image.description="SeisLM — self-supervised Wav2Vec2-style foundation model for 3-component seismic waveforms (base 11.4 M, large 90.7 M parameters), fine-tunable for detection, phase picking and foreshock-aftershock classification. WashU-internal: weights NOT baked, mounted at /weights" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/seislm/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="NOASSERTION" \
      bradleylab.model.upstream="https://github.com/liutianlin0121/seisLM" \
      bradleylab.model.paper="https://arxiv.org/abs/2410.15765" \
      bradleylab.model.task="seismic waveform encoder (self-supervised pretraining) for fine-tuning: event detection, P/S phase identification and onset picking, foreshock-aftershock classification" \
      bradleylab.model.sensor="waveform:seismic (single station, 3-component ZNE, 100 Hz)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.domain="seismology" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.use_restriction="WashU-internal research use only — upstream publishes no license for code or weights; weights are held privately by the lab and mounted at run time" \
      bradleylab.model.weights.source="NOT baked — SeisLM-base (136 MB) and SeisLM-large (1.09 GB) Lightning checkpoints from upstream's Google Drive folder, held privately by the lab (NAS ml_model_weights/seislm; Storage3 Active/seislm/weights), mounted at /weights and SHA-256-checked before loading" \
      bradleylab.model.licence.note="No LICENSE file upstream (GitHub: none); setup.py metadata says MIT but ships no license text; the checkpoints carry no license. Code included for WashU-internal research use; weights not redistributed; listed publicly as WashU-only" \
      bradleylab.upstream.git_sha="${SEISLM_GIT_SHA}"

WORKDIR /work
CMD ["bash", "-c", "echo 'seislm ready: mount the lab weights at /weights, then run seislm-selftest --device cuda (see README.md)'"]
