# aifs — ECMWF AIFS Single 2.0 (data-driven global weather forecasting)
#
# Encoder-processor-decoder model (GNN encoder/decoder + sliding-window
# transformer processor) producing 6-hourly global forecasts on the N320
# grid (~31 km) out to 15 days. ECMWF runs it operationally 4x/day.
#
# Upstream:  https://github.com/ecmwf/anemoi-inference  (Apache-2.0)
# Weights:   https://huggingface.co/ecmwf/aifs-single-2.0
#            (CC-BY-4.0, ungated, 0.99 GB checkpoint / 1.00 GB repo)
# Stack:     CUDA 12.6 on Ubuntu 24.04 (Python 3.12) + PyPI torch 2.7.0
#            + the anemoi inference stack, pinned to the exact versions in
#            upstream's own pyproject.toml / uv.lock for this checkpoint.
#
# ---------------------------------------------------------------------------
# Why these pins, and why they are OLD
# ---------------------------------------------------------------------------
# Every anemoi version below is behind current PyPI — anemoi-inference is on
# 0.11.2 and anemoi-models on 0.18.0 as of 2026-08-18, while this image
# installs 0.8.3 and 0.9.3. That is deliberate, not staleness. An anemoi
# checkpoint is a pickled torch object whose classes live in `anemoi.models`;
# loading it runs `torch.load(..., weights_only=False)` and reconstructs those
# classes from the *installed* package. The versions that can reconstruct
# aifs-single-mse-2.0.ckpt are the ones ECMWF shipped alongside it, which are
# the ones enumerated in the checkpoint repo's pyproject.toml. Tracking latest
# here would not be "keeping current"; it would be loading a 2.0 checkpoint
# with a model library that has since changed its class layout.
#
# Move the pins only in step with a checkpoint change, and only after a real
# GPU load test — a mismatch surfaces as an unpickling error or, worse, as a
# silently different model.
#
# ---------------------------------------------------------------------------
# Base-image choice
# ---------------------------------------------------------------------------
# Three constraints decide this, in order:
#
#   1. Python 3.12. The only prebuilt flash-attn wheel that matches this torch
#      is cp312-only (see FLASH_ATTN_WHEEL below), and building flash-attn
#      from source is a multi-hour CI job. Ubuntu 24.04 ships 3.12 as its
#      system interpreter.
#   2. PyPI torch 2.7.0 exactly, not a vendor build. That flash-attn wheel is
#      tagged `cu12torch2.7cxx11abiFALSE` — a flash-attn binary is compiled
#      against one specific torch build's C++ ABI. The NGC PyTorch images
#      (`nvcr.io/nvidia/pytorch:25.04-py3`, used by evo2 and ntv3) ship
#      NVIDIA's own `torch 2.7.0a0`; dropping this wheel on top of that is the
#      ABI mismatch that shows up as an undefined-symbol ImportError the first
#      time an attention layer is constructed. torch 2.7.0's PyPI wheel bundles
#      the CUDA 12.6 runtime, hence the 12.6.3 base rather than 12.8.1.
#   3. numpy 1.x. anemoi-graphs 0.6.4 requires `numpy<2,>=1.26`, so upstream's
#      lock resolves numpy 1.26.4.
#
# Note that (3) means the NGC pip constraint file
# (`PIP_CONSTRAINT=/etc/pip/constraint.txt`, pinning numpy==1.26.4) would NOT
# have been a problem here — it is the *opposite* of the situation that forced
# `esm` off NGC. What rules NGC out for this image is (2), the torch build,
# not numpy. Recording that so a future reader does not "fix" the wrong thing.

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# libudunits2-0 is the UDUNITS-2 shared library and libudunits2-data is the
# unit-definition XML it reads at load time; they are separate Ubuntu packages
# and cfunits (a hard dependency of anemoi-transform, sdist-only, ctypes-based)
# fails at *import* without both. The build-time smoke test imports cfunits so
# a missing one fails here rather than mid-forecast.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential \
        libudunits2-0 libudunits2-data \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"

ENV UDUNITS2_XML_PATH=/usr/share/xml/udunits/udunits2.xml

# torch first and alone, so nothing later resolves a different build as a
# transitive dependency. No `--index-url` override: the default PyPI wheel is
# the cu126 build the flash-attn wheel below was compiled against.
RUN pip install --no-cache-dir torch==2.7.0 torch-geometric==2.6.1

# flash-attn as a prebuilt wheel. anemoi-models' FlashAttentionWrapper imports
# flash_attn lazily but unconditionally when the checkpoint asks for flash
# attention, which AIFS does; the SDPA fallback exists but uses far more memory
# (ecmwf/anemoi-inference#119), so it is a fallback, not an equivalent.
#
# This is the same wheel URL upstream's own uv.lock overrides its
# `flash-attn==2.7.4` requirement with — ECMWF publish it because compiling
# flash-attn takes hours. `--no-deps` keeps it from touching the torch pin;
# `--force-reinstall` matches upstream's install line. Override for a local
# experiment with `docker build --build-arg FLASH_ATTN_WHEEL=<url> aifs/`.
ARG FLASH_ATTN_WHEEL=https://github.com/cathalobrien/get-flash-attn/releases/download/v0.1-alpha/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
ENV FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL}
RUN pip install --no-cache-dir --force-reinstall --no-deps "flash-attn @ ${FLASH_ATTN_WHEEL}"

# The anemoi stack, in one resolver pass so a conflict fails loudly rather than
# being papered over by install order.
#
#   anemoi-inference[huggingface]  runner + CLI; the extra adds huggingface-hub
#                                  so `checkpoint: {huggingface: ...}` works.
#   anemoi-plugins-ecmwf-inference[opendata]
#                                  supplies `input: opendata` (the free ECMWF
#                                  Open Data feed) and pulls ecmwf-opendata,
#                                  earthkit-regrid and mir-python. mir-python
#                                  is also how the ecCodes *binary* library
#                                  arrives (mir-python -> mirlib -> eccodeslib),
#                                  so no apt libeccodes is needed.
#   anemoi-models                  the model classes the checkpoint unpickles into.
#   anemoi-graphs                  the encoder/decoder graph types; also the
#                                  package that forces numpy<2.
#   anemoi-transform               the pre/post-processor filters named in
#                                  inference.yaml (cos_sin_mean_wave_direction,
#                                  apply-mask).
#   anemoi-utils                   shared config/provenance layer.
#
# anemoi-datasets is deliberately absent: it is the zarr training-dataset
# reader, needed only when initial conditions come from an anemoi dataset. The
# Open Data and GRIB input paths do not touch it, and upstream's checkpoint
# pyproject lists it only under the `training` extra.
#
# torch, torch-geometric and numpy are restated here as resolver guards.
# Nothing in the list requires a different torch, but this image is one
# `pip install` away from the failure mode that has bitten the lab three times
# — a transitive dep quietly dragging in a new torch — and an explicit pin
# turns that into a resolver error at build time instead of a wrong answer at
# runtime. torch-geometric is restated because nothing in the list caps it: an
# unguarded resolve moves layer 1's 2.6.1 to 2.8.x, away from the version
# upstream's own pyproject pins.
#
# earthkit-data is the live instance of that failure mode, not a hypothetical
# one: anemoi-inference 0.8.3 asks only for `earthkit-data>=0.12.4`, with no
# ceiling. earthkit-data has since released 1.x, so an unpinned build today
# resolves to 1.1.1 — a major version the 0.8.3 runner was never tested
# against. Later anemoi-inference releases added the `<1` cap; 0.8.3 predates
# it, and upstream's own aifs pyproject supplies the cap by hand. So does this.
#
# The `-c` is what makes this layer terminate. Left to search on its own, pip
# walked mirlib, contourpy and cftime downwards release by release, reached the
# cfunits 3.3.2 sdist seventeen minutes in, and died there — that sdist's
# setup.py no longer runs under current setuptools. The pin set is not in
# conflict; uv resolves it in seconds and agrees with the checkpoint repo's own
# uv.lock throughout the anemoi / earthkit / ecCodes chain. constraints.txt is
# that resolution, so pip has one candidate per package and nothing to search.
# See the header of aifs/constraints.txt for how to regenerate it.
COPY constraints.txt /etc/pip/aifs-constraints.txt
RUN pip install --no-cache-dir -c /etc/pip/aifs-constraints.txt \
        "torch==2.7.0" \
        "torch-geometric==2.6.1" \
        "numpy==1.26.4" \
        "anemoi-inference[huggingface]==0.8.3" \
        "anemoi-plugins-ecmwf-inference[opendata]==0.2.1" \
        "anemoi-models==0.9.3" \
        "anemoi-graphs==0.6.4" \
        "anemoi-transform==0.1.16.post2" \
        "anemoi-utils==0.4.35.post3" \
        "earthkit-data==0.19.3" \
        "earthkit-regrid==0.5.1" \
        "ecmwf-opendata==0.3.29" \
        "eccodes==2.47.0" \
        "huggingface-hub==1.11.0"

# Weights are NOT baked. anemoi-inference calls huggingface_hub's
# snapshot_download for `checkpoint: {huggingface: "ecmwf/aifs-single-2.0"}`,
# which lands under HF_HOME — the checkpoint and the lsm.grib static field
# together, 1.00 GB. Point HF_HOME at a bind-mounted cache at runtime.
#
# HF is not the only cache that matters here: earthkit-regrid downloads its
# 0.25deg -> N320 interpolation matrices from sites.ecmwf.int on first use, and
# earthkit-data caches retrieved GRIB. Both live under XDG_CACHE_HOME, so the
# simplest runtime arrangement is to mount all of /root/.cache. See the README.
ENV HF_HOME=/root/.cache/huggingface \
    XDG_CACHE_HOME=/root/.cache

# Build-time smoke test, offline: no weight fetch, no network, no GPU.
# Versions come from package metadata per the ml-containers rule, never
# `pkg.__version__`.
#
# What this actually proves, beyond "pip exited 0":
#   - the model class the checkpoint unpickles into is importable;
#   - eccodes found its binary library (the eccodeslib chain above);
#   - cfunits found libudunits2 AND its XML (import-time failure otherwise),
#     and converts a real quantity through it — 3.6 km/h is 1 m/s — rather
#     than only constructing a Units object;
#   - torch is a CUDA build, not the CPU wheel;
#   - numpy stayed on 1.x, i.e. the anemoi-graphs ceiling still holds;
#   - torch-geometric is still layer 1's 2.6.1, i.e. the constraints file kept
#     the anemoi resolve from upgrading it;
#   - flash-attn is present and new enough for anemoi-models' own >=2.6.0 gate.
#
# flash_attn is checked by metadata rather than imported: its extension module
# dlopens the CUDA runtime, and the build runner has no GPU. A real load test
# has to run on an H100 — see SMOKE.md.
RUN python -c "import importlib.metadata as im, numpy, torch, cfunits, eccodes; \
import earthkit.data, earthkit.regrid; \
from packaging.version import Version; \
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec; \
from anemoi.inference.runners.simple import SimpleRunner; \
from ecmwf.opendata import Client; \
v = im.version; \
print('anemoi-inference', v('anemoi-inference'), '| anemoi-models', v('anemoi-models'), \
'| anemoi-graphs', v('anemoi-graphs'), '| anemoi-transform', v('anemoi-transform'), \
'| anemoi-utils', v('anemoi-utils'), '| anemoi-plugins-ecmwf-inference', v('anemoi-plugins-ecmwf-inference')); \
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| flash-attn', v('flash-attn'), \
'| numpy', v('numpy'), '| earthkit-data', v('earthkit-data'), '| earthkit-regrid', v('earthkit-regrid'), \
'| eccodes', v('eccodes'), '| ecmwf-opendata', v('ecmwf-opendata'), '| huggingface-hub', v('huggingface-hub')); \
ms = cfunits.Units('m s-1'); \
kmh = cfunits.Units('km h-1'); \
speed = float(cfunits.Units.conform(3.6, kmh, ms)); \
print('cfunits: 3.6 km h-1 ->', speed, 'm s-1'); \
assert ms.isvalid and kmh.isvalid, (ms, kmh); \
assert abs(speed - 1.0) < 1e-9, speed; \
assert torch.version.cuda is not None, 'CPU-only torch wheel installed'; \
assert Version(v('numpy')).major == 1, v('numpy'); \
assert v('torch-geometric') == '2.6.1', v('torch-geometric'); \
assert Version(v('flash-attn')) >= Version('2.6.0'), v('flash-attn')"

WORKDIR /workspace

LABEL org.opencontainers.image.title="aifs" \
      org.opencontainers.image.description="ECMWF AIFS Single 2.0 — data-driven global medium-range weather forecasting via anemoi-inference; CUDA 12.6 / Python 3.12 / torch 2.7.0; weights fetched at runtime; H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/aifs/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/ecmwf/anemoi-inference" \
      bradleylab.model.paper="https://arxiv.org/abs/2406.01465" \
      bradleylab.model.task="global medium-range weather forecasting, 6-hourly, N320 (~31 km)" \
      bradleylab.model.sensor="reanalysis/NWP-analysis gridded fields" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:ecmwf/aifs-single-2.0 (runtime fetch; CC-BY-4.0, ungated)" \
      bradleylab.build.flash_attn_wheel="${FLASH_ATTN_WHEEL}"
