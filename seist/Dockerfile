# seist — SeisT, multi-task seismogram transformer.
#
# Li et al., "SeisT: A Foundational Deep-Learning Model for Earthquake
# Monitoring Tasks" (IEEE TGRS 2024). One backbone, five task heads:
# event detection, P/S phase picking, first-motion polarity, magnitude,
# back-azimuth, and epicentral distance.
#
# POSITIONING — this is NOT a second phase picker. The lab already runs
# `seisbench` (PhaseNet + EQTransformer) and that covers detection and
# P/S picking well. SeisT earns its own image for the four tasks SeisBench
# does not do: first-motion POLARITY, MAGNITUDE, BACK-AZIMUTH, and
# EPICENTRAL DISTANCE from a single three-component trace. Use seisbench
# for picks; use this for what happens after the pick. See README.md.
#
# Tiny by modern standards — the S/M/L backbones are 0.13 M / 0.31 M /
# 0.66 M parameters. Inference does not need a GPU.
#
# Upstream: https://github.com/senli1073/SeisT  (MIT)
# Paper:    https://doi.org/10.1109/TGRS.2024.3371503  (Li et al. 2024, IEEE TGRS)
# Stack:    pytorch/pytorch 2.5.1 + CUDA 12.1 (Python 3.11) + ObsPy + timm
#
# WEIGHTS ARE IN-REPO AND STAY THERE — a deliberate decision, see README
# ("Weights: baked, and why that is the right call here"). The 18
# checkpoints are committed under `pretrained/` in the upstream repo,
# total 29.3 MB, and there is no Hugging Face repo or other out-of-band
# source for them. They arrive with the SHA-pinned source checkout below.
# Nothing is downloaded at run time and the image needs no network.
#
# BASE-IMAGE CHOICE. Inference is CPU-fine, so a CUDA base is not needed
# for picking or magnitude estimation on an existing catalog, and the
# README says so. The image is CUDA-capable because the realistic lab
# path is FINE-TUNING: the shipped weights are trained on DiTing (China)
# and PNW, and magnitude / back-azimuth are the two tasks least likely to
# transfer to a new region unrecalibrated. Upstream ships a full DDP
# training engine (`main.py`, `start_distributed.sh`) for exactly that,
# and that is where an H100 pays. Base is plain `pytorch/pytorch:*-runtime`
# rather than an NGC PyTorch image: NGC would add ~7 GB and ships
# PIP_CONSTRAINT=/etc/pip/constraint.txt pinning numpy==1.26.4, and this
# stack gains nothing from NGC's fused kernels. Same base as `dinov3-sat`
# and `sam2`, so the torch ABI is coherent across the lab's GPU images.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    SEIST_HOME=/opt/seist \
    PYTHONPATH=/opt/seist

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency pins.
#
# Upstream's requirements.txt is a fully-pinned 2023 environment built
# around torch==2.0.1 / h5py==3.1.0 / numpy==1.24.2 / pandas==1.5.3, and
# it does not install on this base: h5py 3.1.0 (2020) publishes no
# cp311 wheel, so pip falls back to an sdist that wants a C toolchain and
# an HDF5 dev install. Checked against PyPI rather than assumed. Since
# that pin set has to move anyway, the whole block is date-matched to the
# pinned commit (2025-06) instead of to the paper's 2023 environment —
# per the transitive-pinning rule this repo learned the hard way.
#
# torch 2.5.1 comes from the base image and is deliberately BELOW 2.6.
# torch 2.6 flipped `torch.load` to weights_only=True by default, and
# upstream's `models/_factory.py:load_checkpoint` calls
# `torch.load(path, map_location=device)` with no weights_only argument.
# That trap was checked against a real checkpoint rather than assumed: on
# torch 2.10, `pretrained/seist_m_dpk_diting.pth` loads fine under the
# weights_only=True default — the released checkpoints are bare
# `OrderedDict` state dicts of tensors, which the weights_only unpickler
# accepts. So a future bump past 2.6 is safe on this axis; the pin is for
# ABI coherence with `dinov3-sat` and `sam2`, not to dodge the trap.
#
# timm and GPUtil keep upstream's exact pins. Only ONE symbol is imported
# from timm in the whole repo — `timm.models.layers.DropPath` — so 0.9.2
# is both what upstream tested and cheap to hold.
#
# obspy keeps upstream's exact 1.4.0 (it does publish a cp311 wheel).
# Only `obspy.signal.trigger.trigger_onset` is used, in the detection
# post-processing.
#
# pandas moves 1.5.3 -> 2.2.3. The repo's entire pandas surface is
# read_csv / concat / notnull / DataFrame / to_csv, all unchanged across
# that range; the dataset loaders use nothing that 2.x removed.
#
# Not installed, and deliberately: onnx, onnxruntime, openpyxl, xlrd,
# tensorboard-data-server, tensorboard-plugin-wit. No module in the repo
# imports onnx or onnxruntime (the ONNX pins are a leftover from the
# authors' export experiments), no loader reads Excel, and the last two
# are transitive deps of tensorboard.
RUN pip install --upgrade pip wheel \
    && pip install \
        "timm==0.9.2" \
        "obspy==1.4.0" \
        "GPUtil==1.4.0" \
        "h5py==3.12.1" \
        "numpy==1.26.4" \
        "pandas==2.2.3" \
        "matplotlib==3.9.2" \
        "tensorboard==2.18.0"

# Upstream at a pinned commit (2025-06-10, current repo head). There is no
# PyPI package and no setup.py, so the source is unpacked rather than
# pip-installed. The whole tree is taken — unlike most images here, the
# `pretrained/` checkpoints ARE part of it and must come along.
#
# Bumping this SHA requires bumping the image tag too.
ARG SEIST_GIT_SHA=3a1ebdd8f923682a9641d4f88817c3cc0659408f
RUN mkdir -p "${SEIST_HOME}" \
    && curl -fsSL "https://codeload.github.com/senli1073/SeisT/tar.gz/${SEIST_GIT_SHA}" -o /tmp/seist-src.tar.gz \
    && tar -xzf /tmp/seist-src.tar.gz -C "${SEIST_HOME}" --strip-components=1 \
    && rm /tmp/seist-src.tar.gz \
    && rm -rf "${SEIST_HOME}/images" \
    && mkdir -p "${SEIST_HOME}/logs" \
    && echo "${SEIST_GIT_SHA}" > /opt/seist/UPSTREAM_SHA
# `images/` is the 1.7 MB architecture figure from the README, dropped.
# `logs/` is created because upstream's `start.sh` redirects into it and
# fails on a fresh checkout otherwise.

COPY smoke_test.py /opt/seist/smoke_test.py

# Smoke test — no network, no downloads. Asserts the 18 checkpoints
# arrived with the checkout, that all 15 SeisT entrypoints registered, and
# that one checkpoint per task family loads strict=True and produces the
# documented output shape. The same script is the on-node smoke test (see
# SMOKE.md). Versions inside it are read through importlib.metadata, never
# pkg.__version__ (~/.claude/rules/ml-containers.md).
RUN python /opt/seist/smoke_test.py

LABEL org.opencontainers.image.title="seist" \
      org.opencontainers.image.description="SeisT multi-task seismogram transformer (Li et al. 2024, IEEE TGRS) — first-motion polarity, magnitude, back-azimuth and epicentral distance from three-component waveforms, plus detection and P/S picking. Complements seisbench rather than replacing it. 18 pretrained checkpoints (DiTing, PNW) ship in-image." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/seist/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/senli1073/SeisT" \
      bradleylab.model.paper="https://doi.org/10.1109/TGRS.2024.3371503" \
      bradleylab.model.task="multi-task seismology — first-motion polarity, magnitude, back-azimuth, epicentral distance, event detection, P/S phase picking" \
      bradleylab.model.sensor="waveform:seismic (3-component, 8192-sample window)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.domain="seismology" \
      bradleylab.model.weights.source="in-repo at the pinned upstream SHA — /opt/seist/pretrained/*.pth, 18 checkpoints, 29.3 MB; no runtime fetch, no network needed" \
      bradleylab.upstream.git_sha="${SEIST_GIT_SHA}"

WORKDIR /work
CMD ["bash"]
