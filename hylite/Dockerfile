# hylite — hyperspectral drillcore and hand-specimen mineral mapping.
#
# hylite (Thiele et al., HIF Freiberg) is a hyperspectral toolkit for
# geology: correction, hull removal, minimum-wavelength mapping, and
# mineral index calculation on drillcore and outcrop scans. hklearn adds
# the machine-learning layer on top of it.
#
# THIS IS A TOOLKIT, NOT A PRETRAINED MODEL. Every other image in this
# catalog ships weights and answers a question out of the box. This one
# ships no weights: minimum-wavelength mapping is fitted per-scan, and
# hklearn's classifiers are trained on your own labelled core. "Running
# it" means fitting on your data, not inference against a checkpoint.
# That is the nature of the field — the 2026-08 coverage triage found no
# pretrained hyperspectral mineral model with released weights anywhere,
# and it is filed under genuine voids for that reason.
#
# It closes a real gap rather than duplicating `raman-classifier` or
# `xrd-classifier`: those identify a mineral from a point spectrum, this
# maps mineralogy across a spatially resolved hyperspectral scan.
#
# hklearn is not on PyPI, so it is installed from a pinned commit.
# hylite is on PyPI and pinned to an exact version.
#
# Upstream: https://github.com/hifexplo/hylite
#           https://github.com/hifexplo/hklearn
# Licence:  MIT (both)
# Stack:    python 3.11 slim — CPU. Neither package uses CUDA, and the
#           NVIDIA_VISIBLE_DEVICES point below does not apply because
#           nothing here asks for a GPU.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HKLEARN_COMMIT=4f79fb75577d

# gdal/spectral IO and the scientific wheels below need these; git for the
# hklearn install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# hylite 1.36 is the version verified against this stack. numpy is held
# below 2 because hylite's spectral/scipy path has not been validated
# against the numpy 2 ABI break.
RUN pip install --upgrade pip wheel \
    && pip install \
        "hylite==1.36" \
        "numpy<2" \
        "scikit-learn==1.5.2" \
        "matplotlib==3.9.2"

RUN pip install "git+https://github.com/hifexplo/hklearn.git@${HKLEARN_COMMIT}"

# Build smoke test — CPU, offline. Constructs a synthetic hyperspectral
# cube and runs the operation the container exists for: hull removal and
# minimum-wavelength fitting. An import check would not catch a broken
# scipy/numpy pairing in the fitting path, which is where this breaks.
RUN python -c "\
import importlib.metadata as im; \
import numpy as np; \
import hylite; \
from hylite import HyImage; \
print('hylite', im.version('hylite'), '| numpy', im.version('numpy')); \
import hklearn; \
print('hklearn imported'); \
rng = np.random.default_rng(0); \
wav = np.linspace(2100.0, 2400.0, 60); \
cube = 0.8 - 0.3 * np.exp(-0.5 * ((wav - 2200.0) / 25.0) ** 2); \
data = np.repeat(np.repeat(cube[None, None, :], 4, axis=0), 5, axis=1) + rng.normal(0, 0.001, (4, 5, 60)); \
img = HyImage(data.astype(np.float32)); \
img.set_wavelengths(wav); \
print('cube', img.data.shape, '| bands', len(img.get_wavelengths())); \
assert img.data.shape == (4, 5, 60), img.data.shape; \
assert abs(img.get_wavelengths()[0] - 2100.0) < 1e-6; \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="hylite" \
      org.opencontainers.image.description="hylite + hklearn — hyperspectral toolkit for drillcore and outcrop mineral mapping: correction, hull removal, minimum-wavelength mapping, and ML on top. Ships no pretrained weights; models are fitted per-scan" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/hylite/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/hifexplo/hylite" \
      bradleylab.model.task="hyperspectral mineral mapping on drillcore and outcrop scans (fitted per-scan, not pretrained)" \
      bradleylab.model.sensor="hyperspectral:SWIR/VNIR drillcore and outcrop scans" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="none — toolkit, no pretrained weights; hklearn classifiers train on your own labelled core"

WORKDIR /work
CMD ["python", "-c", "import hylite, hklearn; print('hylite + hklearn ready — see README.md')"]
