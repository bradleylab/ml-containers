# n2n4m (rob-platt/N2N4M) — Noise2Noise denoising for CRISM Mars SWIR data.
#
# CRISM's L (SWIR) detector has degraded badly since 2006, and a large
# fraction of the later archive is written off as unusable spike/stripe
# noise. N2N4M is a 1D convolutional U-Net (1,092,945 parameters) trained
# Noise2Noise-style on 350 of CRISM's 438 L-sensor channels — the
# PLEBANI_WAVELENGTHS subset — that denoises spectra pixel-by-pixel and
# leaves the remaining channels untouched. The package also ships CoTCAT
# (Bultel et al. 2015) as a benchmark denoiser, Plebani-model image
# ratioing, and CRISM summary parameters.
#
# Complements `momo` in this repo: MOMO is a foundation model over Mars
# orbital *imagery* (HiRISE / CTX / THEMIS panchromatic-scale tiles);
# n2n4m operates on CRISM *hyperspectral* cubes and is the step that makes
# the degraded half of that archive analysable at all.
#
# Upstream code + weights: https://github.com/rob-platt/n2n4m  (MIT)
# Paper:                   Platt, Arcucci & John, arXiv:2403.17757 (2024)
# Ratioing dependency:     https://github.com/Banus/crism_ml  (Plebani
#                          et al. 2022, Icarus 376:114849)
# External data (NOT baked, see README): Plebani bland-pixel dataset,
#                          Zenodo 10.5281/zenodo.13338091, CC-BY-4.0.
#
# ---------------------------------------------------------------------
# Weights ARE baked, deliberately.
# ---------------------------------------------------------------------
# Upstream commits the trained weights and the fitted feature scaler into
# the source tree (`n2n4m/data/`) and declares them as package data, so
# `pip install` places them in site-packages. That is 4.4 MB + 13 KB, not a
# multi-GB download, and it means the image tag pins code and weights to
# one commit SHA — a stronger reproducibility guarantee than the
# fetch-at-runtime images in this repo, where `:v1` says nothing about
# which checkpoint a job actually loaded. Consequence: no network, no HF
# token, no cache mount at inference time. The smoke test asserts both
# files are present and loadable, so if upstream ever stops shipping them
# the build fails loudly instead of publishing a weightless image.
#
# ---------------------------------------------------------------------
# Base image: python:3.11-slim, not a CUDA base and not NGC.
# ---------------------------------------------------------------------
# The model is ~1.1 M parameters and runs fine on CPU; upstream picks the
# device opportunistically (`check_available_device()` returns cuda only if
# torch sees one). A CUDA *base image* would add ~2 GB of CUDA runtime
# libraries that the PyPI cu128 torch wheel already bundles as its own
# `nvidia-*` dependencies, and an NGC base would add its whole
# Megatron/apex/transformer-engine surface for a package that imports none
# of it. So: slim Debian base, CUDA capability carried by the torch wheel.
# One image therefore runs unchanged on Compute2's `general-cpu` and on
# `general-gpu` with an H100.
#
# Python 3.11 matches upstream's own environment.yml (3.11.4) and sits
# inside setup.py's `>=3.10,<3.14`.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# git: pip installs both n2n4m and crism_ml from git URLs.
# libgomp1: the slim base omits it. The heavyweight wheels here (torch,
# numpy, scipy, scikit-learn) each vendor their own copy, so this is
# defensive rather than known-required — ~150 KB to turn a class of
# import-time OSError on a slim base into a non-event.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# torch first so nothing later drags in a different build as a transitive
# dependency. cu128 for consistency with `esm` in this repo; H100 is sm_90,
# covered by cu126 and cu128 alike. The wheel runs on CPU-only hosts — it
# just reports torch.cuda.is_available() == False — which is what makes the
# single-image, both-partitions story work.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1

# Scientific stack, fully pinned. Every version below resolves to a cp311
# manylinux wheel, so no compiler is needed in this image.
#
# scikit-learn: the shipped scaler was pickled under 1.2.2, so any modern
# sklearn emits InconsistentVersionWarning on load. Verified that 1.7.2
# reads it correctly — RobustScaler's fitted state is center_/scale_ and
# transform is (X - center_)/scale_, which is arithmetically identical
# across these versions (checked against the unpickled arrays by hand).
# Pinning rather than floating keeps that verification meaningful.
#
# numpy 2.x: the scaler pickle references the numpy 1.x
# `numpy.core.multiarray._reconstruct` path; numpy 2's compatibility
# shim resolves it. Verified by loading the file under 2.4.6.
#
# pandas held at 2.x: upstream was developed against 1.5.3 and pandas 3.0
# changes string-dtype and copy-on-write defaults. Not a risk worth taking
# for a package whose pandas use is incidental.
#
# ipywidgets is not optional — n2n4m.plot imports it at module scope.
RUN pip install \
        "numpy==2.4.6" \
        "scipy==1.17.1" \
        "pandas==2.3.3" \
        "scikit-learn==1.7.2" \
        "joblib==1.5.3" \
        "matplotlib==3.10.7" \
        "pyarrow==22.0.0" \
        "spectral==0.25" \
        "mat73==0.65" \
        "Bottleneck==1.6.0" \
        "ipywidgets==8.1.9" \
        "ipykernel==7.3.0" \
        "pytest==8.4.2"

# Reproducibility pins. n2n4m's setup.py declares crism_ml as
# `git+...@master`, so an unpinned build silently tracks whatever Plebani's
# master branch is that day. Both refs are resolved to commit SHAs here,
# recorded in the labels, and asserted by the smoke test against pip's own
# direct_url.json records. Override locally with
# `docker build --build-arg N2N4M_REF=<sha> ...`; to change the pin in CI,
# edit these defaults and commit, so the pin lives in git history.
ARG N2N4M_REF=7e5c58a8f5b0ef032ca2f5afbb685e9a4e497a4a
ARG CRISM_ML_REF=028025e61cfd1d56874474a149ab3c0ee4764862
ENV N2N4M_REF=${N2N4M_REF} \
    CRISM_ML_REF=${CRISM_ML_REF}

# crism_ml first, at its pin. Its declared floors (joblib, matplotlib,
# numpy, scipy, spectral, Bottleneck, mat73) are all already satisfied
# above, so pip leaves the pinned versions alone.
RUN pip install "crism_ml @ git+https://github.com/Banus/crism_ml.git@${CRISM_ML_REF}"

# n2n4m with --no-deps. Its install_requires carries the unpinned
# `crism_ml @ git+...@master` direct URL, and pip treats direct-URL
# requirements as always-reinstall — installing with deps would overwrite
# the SHA pinned above. Its remaining requirements (torch, scikit-learn,
# pandas, pytest, ipykernel, pyarrow, ipywidgets) are installed above.
RUN pip install --no-deps "n2n4m @ git+https://github.com/rob-platt/n2n4m.git@${N2N4M_REF}"

# Build-time smoke test — offline, CPU-only, no network.
#
# Versions come from package metadata per the ml-containers rule, never
# pkg.__version__ (n2n4m exposes no __version__ at all).
#
# Beyond imports this asserts the things that actually rot:
#   1. Both git dependencies are installed at the SHAs this image claims.
#   2. The committed weights and fitted scaler are present in site-packages
#      at the paths importlib.resources resolves — i.e. the package_data
#      declaration in upstream's setup.py still holds.
#   3. The checkpoint LOADS under this torch. torch 2.6 flipped
#      torch.load's weights_only default to True and n2n4m calls
#      torch.load with no weights_only argument. Inspected the checkpoint:
#      it is a plain OrderedDict of tensors referencing only
#      torch._utils._rebuild_tensor_v2 / torch.{Float,Long}Storage, all on
#      the weights_only allowlist, so it loads clean — verified on torch
#      2.9.1. This assertion is what catches it if a future torch bump
#      changes that.
#   4. A real forward pass produces finite output of the right shape, and
#      the scaler transforms a 350-band batch. That reaches the actual
#      trained weights, not just the class definition.
RUN python -c "import importlib.metadata as im, json, os, warnings; \
warnings.simplefilter('ignore'); \
import numpy as np, torch, crism_ml; \
sha = lambda n: json.loads(im.distribution(n).read_text('direct_url.json'))['vcs_info']['commit_id']; \
n2n_sha, cml_sha = sha('n2n4m'), sha('crism_ml'); \
assert n2n_sha == os.environ['N2N4M_REF'], (n2n_sha, os.environ['N2N4M_REF']); \
assert cml_sha == os.environ['CRISM_ML_REF'], (cml_sha, os.environ['CRISM_ML_REF']); \
import n2n4m.plot, n2n4m.crism_image, n2n4m.cotcat_denoise, n2n4m.summary_parameters; \
from n2n4m.n2n4m_denoise import DEFAULT_MODEL_FILEPATH, DEFAULT_SCALER_FILEPATH, instantiate_default_model, load_scaler, create_dataloader; \
from n2n4m.model_functions import predict, check_available_device; \
assert DEFAULT_MODEL_FILEPATH.is_file(), DEFAULT_MODEL_FILEPATH; \
assert DEFAULT_SCALER_FILEPATH.is_file(), DEFAULT_SCALER_FILEPATH; \
w_mb = DEFAULT_MODEL_FILEPATH.stat().st_size / 1e6; \
assert w_mb > 4.0, w_mb; \
model = instantiate_default_model(); \
scaler = load_scaler(); \
assert scaler.n_features_in_ == 350, scaler.n_features_in_; \
spectra = scaler.transform(np.full((4, 350), 0.2)); \
out = predict(model, create_dataloader(spectra, batch_size=4), torch.device(check_available_device())); \
assert tuple(out.shape) == (4, 350), out.shape; \
assert bool(torch.isfinite(out).all()); \
print('n2n4m', im.version('n2n4m'), n2n_sha); \
print('crism_ml', im.version('crism_ml'), cml_sha); \
print('torch', torch.__version__, '| cuda-build', torch.version.cuda, '| numpy', im.version('numpy')); \
print('scikit-learn', im.version('scikit-learn'), '| scipy', im.version('scipy'), '| pandas', im.version('pandas')); \
print('spectral', im.version('spectral'), '| mat73', im.version('mat73')); \
print('weights', round(w_mb, 2), 'MB | params', sum(p.numel() for p in model.parameters()))"

LABEL org.opencontainers.image.title="n2n4m" \
      org.opencontainers.image.description="N2N4M (rob-platt) — Noise2Noise 1D U-Net denoiser for CRISM L-sensor SWIR hyperspectral data; trained weights and fitted scaler baked in from the pinned upstream commit. Also ships CoTCAT benchmark denoising, Plebani-model image ratioing, and CRISM summary parameters. Plebani bland-pixel dataset (Zenodo 13338091) is external and must be mounted." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/n2n4m/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/rob-platt/n2n4m" \
      bradleylab.model.paper="https://doi.org/10.48550/arXiv.2403.17757" \
      bradleylab.model.task="hyperspectral denoising — per-pixel spectral noise removal for degraded CRISM L-sensor SWIR data" \
      bradleylab.model.sensor="hyperspectral:CRISM L (SWIR, 438 bands, 350 denoised)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.domain="planetary-mars" \
      bradleylab.model.weights.source="baked — committed upstream at n2n4m/data/trained_model_weights.pt (4.4 MB) + n2n4m_feature_scaler.joblib; pinned by N2N4M_REF" \
      bradleylab.data.external="Plebani bland-pixel dataset, Zenodo 10.5281/zenodo.13338091 (CC-BY-4.0) — required for image ratioing, mount at runtime" \
      bradleylab.upstream.git_sha="${N2N4M_REF}" \
      bradleylab.build.crism_ml_ref="${CRISM_ML_REF}"

WORKDIR /work
CMD ["bash"]
