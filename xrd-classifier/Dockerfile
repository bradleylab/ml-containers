# XRD-Classifier — automated phase identification from powder XRD patterns.
#
# autoXRD (Szymanski et al. 2021, Chem. Mater.) — probabilistic
# multi-phase identification using a 1D-CNN trained on simulated
# patterns from a user-supplied CIF library, with physics-informed
# augmentation (preferred orientation, crystallite-size broadening,
# strain). Multi-phase by design (defaults to up to 3 phases per
# pattern).
#
# Ships the Li-Mn-Ti-O-F demo model bundled by upstream so a smoke
# run works on first pull. For arbitrary chemistries (mineralogy /
# sed-pet / new battery materials), users either drop CIFs into
# Novel-Space/All_CIFs and retrain via the bundled training pipeline,
# or build a downstream variant of this image with a chemistry-
# specific weights bake.
#
# Upstream: https://github.com/njszym/XRD-AutoAnalyzer
# PyPI:    https://pypi.org/project/autoXRD/
# Paper:   Szymanski et al. (2021), Chem. Mater. — doi:10.1021/acs.chemmater.1c01071
# Stack:   python:3.11-slim + TensorFlow >=2.16 + autoXRD + pymatgen + pyxtal
#
# AMD64-only: the upstream's prediction pipeline calls into BGMN
# (Rietveld refinement) at every spectrum, and BGMN ships only as a
# Linux x86_64 binary. There is no arm64 build of BGMN, so an arm64
# image would fail at first inference. Apple-Silicon Mac users must
# `docker pull --platform linux/amd64` and run via qemu emulation.

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

# Clone the upstream repo at a pinned commit FIRST. The clone is the
# single source of truth for both the package code (to lock the
# version against the Example/Model.h5 weights) and the demo assets
# (Example/ directory with run_CNN.py, References/, Spectra/).
# Bump the SHA when refreshing.
ARG AUTOXRD_GIT_SHA=bf32082521e45c0fcf5cf9ae9bd1321e76bf9012
RUN git clone https://github.com/njszym/XRD-AutoAnalyzer.git /opt/xrd-autoanalyzer \
    && cd /opt/xrd-autoanalyzer \
    && git checkout ${AUTOXRD_GIT_SHA} \
    && rm -rf .git \
    && rm -f Example/.DS_Store

# Install autoXRD from the cloned source — NOT from PyPI. This avoids
# the v1 layering bug where pip resolved 0.0.7 from PyPI while the
# cloned repo carried 0.0.6 source matching the bundled Model.h5
# weights, leading to nondeterministic version pickup based on cwd.
# Pin TensorFlow + the other heavy deps explicitly; let pip resolve
# the rest from autoXRD's pyproject.
RUN pip install --upgrade pip wheel \
    && pip install \
        "tensorflow>=2.16,<2.20" \
        "matplotlib>=3.8" \
        "huggingface_hub>=0.25.0" \
    && pip install /opt/xrd-autoanalyzer

# Pre-bake BGMN (Rietveld refinement engine — Linux x86_64 binary)
# at build time. Upstream's spectrum-analysis pipeline calls
# BGMNWorker() unconditionally during prediction, which lazily
# downloads BGMN from a personal Berkeley homepage and races under
# multiprocessing. Doing it once at build time avoids the runtime
# race and means first inference works without network access.
# Downloaded zip extracts to autoXRD/dara/data/BGMNwin/.
RUN python -c "from autoXRD.dara import download_bgmn; download_bgmn()" \
    && AUTOXRD_DATA=$(python -c "import autoXRD, os; print(os.path.join(os.path.dirname(autoXRD.__file__), 'dara/data'))") \
    && echo "AUTOXRD_DATA=${AUTOXRD_DATA}" \
    && ls -la "${AUTOXRD_DATA}/" \
    && test -d "${AUTOXRD_DATA}/BGMNwin" \
    && test -x "${AUTOXRD_DATA}/BGMNwin/bgmn"

# Smoke test: import + reach the public autoXRD API + verify the demo
# Model.h5 exists + verify BGMN baked successfully. importlib.metadata
# for portable version (per ~/.claude/rules/ml-containers.md, applies
# to every package referenced).
RUN python -c "import autoXRD, tensorflow as tf, pymatgen; \
import importlib.metadata as im; \
import os; \
print('autoXRD', im.version('autoXRD'), '| tensorflow', im.version('tensorflow'), '| pymatgen', im.version('pymatgen')); \
assert os.path.exists('/opt/xrd-autoanalyzer/Example/Model.h5'), 'demo Model.h5 missing'; \
print('demo Model.h5 size:', os.path.getsize('/opt/xrd-autoanalyzer/Example/Model.h5'), 'bytes'); \
bgmn_dir = os.path.join(os.path.dirname(autoXRD.__file__), 'dara/data/BGMNwin'); \
assert os.path.isdir(bgmn_dir), f'BGMN missing at {bgmn_dir}'; \
assert os.access(os.path.join(bgmn_dir, 'bgmn'), os.X_OK), 'bgmn not executable'; \
print('bgmn binary present and executable at', bgmn_dir)"

LABEL org.opencontainers.image.title="xrd-classifier" \
      org.opencontainers.image.description="autoXRD (Szymanski et al. 2021) — probabilistic multi-phase identification from 1D powder XRD patterns; CPU runtime; ships Li-Mn-Ti-O-F demo model + BGMN baked in for offline first inference. AMD64-only (BGMN is x86_64)." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/xrd-classifier/README.md" \
      org.opencontainers.image.version="v2" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/njszym/XRD-AutoAnalyzer" \
      bradleylab.model.paper="https://doi.org/10.1021/acs.chemmater.1c01071" \
      bradleylab.model.task="powder X-ray diffraction phase identification (multi-phase, 1D CNN)" \
      bradleylab.model.sensor="spectrum:xrd-1d (intensity vs 2θ)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="upstream Example/Model.h5 (Li-Mn-Ti-O-F demo, 73 MB, baked in at /opt/xrd-autoanalyzer/Example/Model.h5); retrain on user CIFs via autoXRD pipeline for other chemistries; BGMN refinement engine baked in at autoXRD/dara/data/BGMNwin/" \
      bradleylab.upstream.git_sha="bf32082521e45c0fcf5cf9ae9bd1321e76bf9012"

WORKDIR /opt/xrd-autoanalyzer
CMD ["bash"]
