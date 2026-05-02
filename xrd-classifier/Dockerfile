# XRD-Classifier — automated phase identification from powder XRD patterns.
#
# autoXRD (Szymanski et al. 2021, Chem. Mater.) — probabilistic
# multi-phase identification using a 1D-CNN trained on simulated
# patterns from a user-supplied CIF library, with physics-informed
# augmentation (preferred orientation, crystallite-size broadening,
# strain). Multi-phase by design (defaults to up to 3 phases per
# pattern).
#
# This container ships the Li-Mn-Ti-O-F demo model bundled by
# upstream so a smoke run works on first pull. For arbitrary
# chemistries (mineralogy / sed-pet / new battery materials), users
# either drop CIFs into Novel-Space/All_CIFs and retrain via the
# bundled training pipeline, or build a downstream variant of this
# image with a chemistry-specific weights bake.
#
# Upstream: https://github.com/njszym/XRD-AutoAnalyzer
# PyPI:    https://pypi.org/project/autoXRD/
# Paper:   Szymanski et al. (2021), Chem. Mater. — doi:10.1021/acs.chemmater.1c01071
# Stack:   python:3.11-slim + TensorFlow >=2.16 + autoXRD + pymatgen + pyxtal
#
# CPU-only by design — autoXRD inference is ~10 s/pattern on CPU and
# the Compute2 use case (catalog-scale phase ID) parallelises across
# CPU job arrays better than it scales up a single GPU.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/opt/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# autoXRD pulls in TensorFlow + pymatgen + pyxtal; let pip resolve.
# TensorFlow CPU wheels are large (~600 MB) — accept the size, this
# is the canonical dependency for the upstream tool.
RUN pip install --upgrade pip wheel \
    && pip install \
        "autoXRD>=0.0.6" \
        "tensorflow>=2.16,<2.20" \
        "matplotlib>=3.8" \
        "huggingface_hub>=0.25.0"

# Clone the upstream repo at a pinned commit so the Example/ directory
# (Li-Mn-Ti-O-F demo Model.h5 + reference CIFs + run_CNN.py + Spectra/)
# is available inside the container. pip install doesn't ship Example/
# (it's not a Python package), so the clone is the canonical demo
# source. Bump the SHA when refreshing.
ARG AUTOXRD_GIT_SHA=bf32082521e45c0fcf5cf9ae9bd1321e76bf9012
RUN git clone https://github.com/njszym/XRD-AutoAnalyzer.git /opt/xrd-autoanalyzer \
    && cd /opt/xrd-autoanalyzer \
    && git checkout ${AUTOXRD_GIT_SHA} \
    && rm -rf .git \
    && rm -f Example/.DS_Store

# Smoke test: import + reach the public autoXRD API + verify the demo
# Model.h5 exists. No prediction at build time (that needs spectrum
# input). importlib.metadata for portable version
# (per ~/.claude/rules/ml-containers.md).
RUN python -c "import autoXRD, tensorflow as tf, pymatgen; \
import importlib.metadata as im; \
import os; \
print('autoXRD', im.version('autoXRD'), '| tensorflow', tf.__version__, '| pymatgen', pymatgen.__version__); \
assert os.path.exists('/opt/xrd-autoanalyzer/Example/Model.h5'), 'demo Model.h5 missing'; \
print('demo Model.h5 size:', os.path.getsize('/opt/xrd-autoanalyzer/Example/Model.h5'), 'bytes')"

LABEL org.opencontainers.image.title="xrd-classifier" \
      org.opencontainers.image.description="autoXRD (Szymanski et al. 2021) — probabilistic multi-phase identification from 1D powder XRD patterns; CPU runtime; ships Li-Mn-Ti-O-F demo model + retraining pipeline for arbitrary chemistries." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/xrd-classifier/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/njszym/XRD-AutoAnalyzer" \
      bradleylab.model.paper="https://doi.org/10.1021/acs.chemmater.1c01071" \
      bradleylab.model.task="powder X-ray diffraction phase identification (multi-phase, 1D CNN)" \
      bradleylab.model.sensor="spectrum:xrd-1d (intensity vs 2θ)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="upstream Example/Model.h5 (Li-Mn-Ti-O-F demo, 73 MB, baked in at /opt/xrd-autoanalyzer/Example/Model.h5); retrain on user CIFs via autoXRD pipeline for other chemistries" \
      bradleylab.upstream.git_sha="bf32082521e45c0fcf5cf9ae9bd1321e76bf9012"

WORKDIR /opt/xrd-autoanalyzer
CMD ["bash"]
