# corrdiff — CorrDiff COSMO/ERA5 km-scale generative downscaling (NVIDIA Earth-2)
#
# Downscales 0.25 deg ERA5 onto the rotated-pole COSMO-REA grids over Europe:
# REA6 (6 km, 824 x 848, 45 output variables) and REA2 (2.2 km, 780 x 724,
# 22 output variables). Two inference modes from the same package — a
# deterministic regression DiT (98M params) and an EDM-preconditioned
# diffusion DiT (174M params) sampled with an 18-step Heun sampler.
#
# Upstream:  https://github.com/NVIDIA/earth2studio  (framework, Apache-2.0)
#            https://github.com/NVIDIA/physicsnemo   (DiT + diffusion, Apache-2.0)
# Weights:   https://huggingface.co/nvidia/corrdiff-cosmo-era5
#            OpenMDW-1.1, ungated, 2.33 GB across 19 files. The two
#            `invariants_*_ext.nc` files are separately CC BY 4.0 — see
#            README.md, this is the one licensing detail that is not
#            Apache-shaped.
# Papers:    https://arxiv.org/abs/2309.15214 (CorrDiff)
#            https://arxiv.org/abs/2206.00364 (EDM preconditioning)
#
# ---------------------------------------------------------------------------
# Base-image choice
# ---------------------------------------------------------------------------
# Same reasoning as the two sibling Earth-2 images: a plain CUDA base plus an
# explicit torch, not an NGC PyTorch image. The physicsnemo revision this model
# needs declares `torch>=2.10.0`, so on an NGC base (25.04 ships torch 2.7.0a0)
# pip would replace NVIDIA's tuned torch with a generic PyPI wheel — the
# cuequivariance 0.11.1 failure again — and NGC's global PIP_CONSTRAINT pinning
# numpy==1.26.4 has already cost this repo a CI failure (esm/README.md).
#
# `-devel` is mandatory here, not stylistic: NATTEN compiles its CUTLASS-based
# neighborhood-attention kernels from source and needs nvcc and cmake.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Ubuntu 24.04's system interpreter is Python 3.12. earth2studio requires
# >=3.11,<3.15, physicsnemo >=3.11,<=3.14, and earth2studio's own `cosmo`
# extra gates NATTEN on `python_version < '3.14'`. Asserted rather than
# trusted.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && python -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), sys.version"

ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# Our own pip constraint file, holding torch and torchvision only, so any
# later resolve that wants to move torch fails at build time instead of
# silently swapping the GPU stack.
RUN mkdir -p /etc/pip \
    && printf 'torch==%s\ntorchvision==%s\n' "${TORCH_VERSION}" "${TORCHVISION_VERSION}" \
        > /etc/pip/e2s-constraint.txt
ENV PIP_CONSTRAINT=/etc/pip/e2s-constraint.txt

ARG EARTH2STUDIO_VERSION=0.17.0
# 0.21.6 rather than the newest 0.21.7 because 0.21.6 is the newest release
# whose official wheel index carries a build for this image's exact
# torch 2.11.0 / cu128 / cp312 / x86_64 target (see NATTEN_WHEEL below).
# earth2studio's `cosmo` extra asks for `natten` with no floor, so the choice
# is ours, and a matching prebuilt wheel is worth more here than a patch
# release. Move it only in step with a wheel that matches the torch pin.
ARG NATTEN_VERSION=0.21.6
# The DiT in this model uses `attention_backend="natten2d_rope"`, which exists
# only on physicsnemo main — it is absent from the v2.1.1 tag, verified by
# diffing physicsnemo/models/dit/dit.py at both refs. earth2studio's `cosmo`
# extra therefore asks for `nvidia-physicsnemo>=2.2.0a0`, a version that has
# never been published to PyPI, and resolves it through `[tool.uv.sources]` to
# a git SHA. pip does not read that table, so `pip install
# earth2studio[cosmo]` fails outright and the SHA is restated here. It is
# upstream's own, copied from earth2studio 0.17.0's pyproject.toml.
ARG PHYSICSNEMO_REF=ced75d93d014f70bb691372788eee2d201171c12
ENV EARTH2STUDIO_VERSION=${EARTH2STUDIO_VERSION} \
    NATTEN_VERSION=${NATTEN_VERSION} \
    PHYSICSNEMO_REF=${PHYSICSNEMO_REF}

# NATTEN ships an sdist only on *PyPI*, but SHI-Labs publish prebuilt wheels
# of their own at https://whl.natten.org, one per (natten, torch, CUDA,
# CPython, machine) combination. This is one of theirs, matched to the torch
# pin above: torch 2.11.0, cu128, cp312, x86_64. Override for a local
# experiment with `docker build --build-arg NATTEN_WHEEL=<url> corrdiff/`.
#
# Compiling the sdist instead was tried and is the wrong trade. Two things go
# wrong with it:
#
#   * The arch variable does not take the value it looks like it takes.
#     natten's setup.py runs `float(arch) * 10` on it, so the documented
#     `NATTEN_CUDA_ARCH=90` becomes SM 900 and every kernel dies on
#     `nvcc fatal: Unsupported gpu architecture 'compute_900'`. The value that
#     means Hopper is the dotted `9.0`.
#   * Even spelled correctly, a Hopper build stamps out 94 CUTLASS translation
#     units under natten's default autogen policy and compiles them four at a
#     time on a 4-vCPU runner. That is the layer this workflow's 360-minute
#     timeout was written around, and it buys nothing the wheel does not
#     already contain.
#
# The wheel is not a downgrade in coverage. NATTEN's own release script
# (scripts/packaging/pkg_helpers.bash) builds every cu128 wheel with
# TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0", so
# 9.0 — the Hopper FNA path this model's DiT reaches through
# `attention_backend="natten2d_rope"` — is in it. The smoke test does not take
# that on trust: it runs cuobjdump over the installed libnatten and asserts an
# sm_90 cubin is present, which is also what catches the older failure mode
# where a kernel-less natten installs cleanly and exits 0.
ARG NATTEN_WHEEL=https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.6/natten-0.21.6%2Btorch2110cu128-cp312-cp312-linux_x86_64.whl
ENV NATTEN_WHEEL=${NATTEN_WHEEL}
RUN pip install --no-cache-dir "natten @ ${NATTEN_WHEEL}" \
    && python -c "import importlib.metadata as im; v = im.version('natten'); \
print('natten', v); \
assert v.split('+')[0] == '${NATTEN_VERSION}', v; \
assert v.split('+')[1] == 'torch2110cu128', v; \
so = [f for f in im.files('natten') if f.name.startswith('libnatten') and f.name.endswith('.so')]; \
assert len(so) == 1, so; \
print(so[0].locate())" | tee /tmp/libnatten.path \
    && cuobjdump --list-elf "$(tail -n1 /tmp/libnatten.path)" > /tmp/libnatten.elf \
    && grep -q 'sm_90' /tmp/libnatten.elf \
    && echo "libnatten carries sm_90 cubins: $(grep -c 'sm_90' /tmp/libnatten.elf)"

# One resolve for the framework, the git physicsnemo, and the rest of the
# `cosmo` extra.
RUN pip install --no-cache-dir \
        "earth2studio==${EARTH2STUDIO_VERSION}" \
        "nvidia-physicsnemo @ git+https://github.com/NVIDIA/physicsnemo.git@${PHYSICSNEMO_REF}" \
        "einops>=0.8.1" \
        "nvtx>=0.2.11"

# Weights are NOT baked — fetched on the first
# `CorrDiffCosmoEra5.load_model(...)` and cached here. Bind-mount a persistent
# dir over this path on Compute2; both resolutions live in one 2.33 GB HF repo
# and earth2studio pulls whichever files the requested mode/resolution needs.
ENV EARTH2STUDIO_CACHE=/root/.cache/earth2studio

# Build-time smoke test — offline, no weight fetch. Versions from package
# metadata per the ml-containers rule, never `pkg.__version__`.
#
# HAS_LIBNATTEN is the assertion that earns its keep: a natten carrying no
# CUDA kernels imports cleanly and only fails once a DiT block actually runs
# on an H100, hours into a job. The natten version is re-read here too, not
# just at install: the resolve above asks for `earth2studio`, whose `cosmo`
# extra requires `natten` with no floor, and an already-satisfied requirement
# is meant to be left alone — this asserts that outcome rather than assuming
# it, so a swap for the PyPI sdist fails the build. The physicsnemo SHA is
# read back from pip's own direct_url.json record so a silently drifted pin
# fails the build rather than shipping.
#
# load_default_package() is called, not merely looked up: it is the real entry
# point a job starts from, it resolves the Hugging Face package descriptor,
# and it fetches no weights.
RUN python -c "import importlib.metadata as im, json, os, torch; \
from natten import HAS_LIBNATTEN; \
from earth2studio.models.dx import CorrDiffCosmoEra5; \
from earth2studio.utils.imports import OptionalDependencyFailure as ODF; \
pnm_sha = json.loads(im.distribution('nvidia-physicsnemo').read_text('direct_url.json'))['vcs_info']['commit_id']; \
bad = [k for k in ODF.failures if k.endswith('corrdiff_cosmo_era5.py')]; \
print('earth2studio', im.version('earth2studio'), '| nvidia-physicsnemo', im.version('nvidia-physicsnemo'), pnm_sha, \
'| natten', im.version('natten'), 'libnatten', HAS_LIBNATTEN, '| torch', torch.__version__); \
assert torch.__version__.split('+')[0] == '${TORCH_VERSION}', torch.__version__; \
assert HAS_LIBNATTEN, 'natten carries no CUDA kernels — the prebuilt wheel was replaced'; \
assert im.version('natten') == '${NATTEN_VERSION}+torch2110cu128', im.version('natten'); \
assert pnm_sha == os.environ['PHYSICSNEMO_REF'], (pnm_sha, os.environ['PHYSICSNEMO_REF']); \
assert not bad, ('cosmo optional dependencies failed to import', bad); \
package = CorrDiffCosmoEra5.load_default_package(); \
print('default package', package); \
assert package is not None"

WORKDIR /workspace

LABEL org.opencontainers.image.title="corrdiff" \
      org.opencontainers.image.description="CorrDiff COSMO/ERA5 (NVIDIA Earth-2) — km-scale generative downscaling of ERA5 to COSMO-REA6/REA2 over Europe; earth2studio + physicsnemo DiT + NATTEN, weights fetched at runtime; H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/corrdiff/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/NVIDIA/earth2studio" \
      bradleylab.model.paper="https://arxiv.org/abs/2309.15214" \
      bradleylab.model.task="km-scale generative downscaling, ERA5 to COSMO-REA" \
      bradleylab.model.sensor="ERA5 reanalysis" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:nvidia/corrdiff-cosmo-era5 (runtime fetch; OpenMDW-1.1, ungated; invariants CC BY 4.0)" \
      bradleylab.model.weights.licenses="OpenMDW-1.1; CC-BY-4.0 (static invariants)" \
      bradleylab.build.earth2studio_version="${EARTH2STUDIO_VERSION}" \
      bradleylab.build.torch_version="${TORCH_VERSION}" \
      bradleylab.build.natten_version="${NATTEN_VERSION}" \
      bradleylab.build.natten_wheel="${NATTEN_WHEEL}" \
      bradleylab.build.physicsnemo_ref="${PHYSICSNEMO_REF}"
