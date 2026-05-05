# raman-classifier — Raman mineral identification via nearest-neighbour
# matching against the RRUFF reference library.
#
# Path A from the deferred raman-classifier slot in STATUS.md: classical
# RRUFF NN matcher via ramanspy. No learned weights, deterministic,
# defensible methodology. Path B (in-house Liu-2017-style 1D-CNN trained
# on RRUFF, Apache-2 weights deposited at Zenodo + HF) remains queued
# as a follow-up.
#
# The container bakes:
#   1. ramanspy (BSD-3) for spectrum I/O, preprocessing, and the RRUFF
#      dataset loader.
#   2. The RRUFF "excellent_unoriented" archive (~229 MB, the canonical
#      curated set) downloaded at build time so the runtime is offline.
#   3. A pre-computed numpy index of preprocessed + L2-normalised
#      reference spectra resampled onto a common 100-1500 cm^-1 grid,
#      so runtime matching is O(N) cosine over a small npz instead of
#      reloading + reprocessing the full archive on every call.
#
# Citation for the underlying data:
#   Lafuente B, Downs RT, Yang H, Stone N (2015). The power of databases:
#   the RRUFF project. In: Highlights in Mineralogical Crystallography,
#   T Armbruster & RM Danisi, eds., De Gruyter, Berlin, 1-30.
#
# Upstream ramanspy: https://github.com/barahona-research-group/RamanSPy
# RRUFF download:    https://www.rruff.net/zipped_data_files/raman/
# Stack:             python:3.11-slim + numpy + scipy + ramanspy

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel \
    && pip install \
        "numpy>=1.26,<2.3" \
        "scipy>=1.11" \
        "ramanspy>=0.2"

# Smoke test: ramanspy reachable and the RRUFF loader is exposed.
# Uses importlib.metadata.version per ml-containers rules — never
# rely on pkg.__version__ for transitive imports.
RUN python -c "import ramanspy as rp; \
import importlib.metadata as im; \
print('ramanspy', im.version('ramanspy'), '| numpy', im.version('numpy'), '| scipy', im.version('scipy')); \
assert callable(rp.datasets.rruff); \
assert callable(rp.preprocessing.Pipeline); \
assert callable(rp.preprocessing.baseline.ASLS); \
assert callable(rp.preprocessing.despike.WhitakerHayes); \
assert callable(rp.preprocessing.normalise.Vector); \
assert hasattr(rp, 'Spectrum')"

# Bake the index-build script so we can iterate on it without
# rebuilding from scratch (Docker's layer cache will reuse the pip
# install above).
RUN mkdir -p /opt/scripts /opt/rruff/raw
COPY scripts/build_rruff_index.py /opt/scripts/build_rruff_index.py
COPY scripts/raman_match.py       /opt/scripts/raman_match.py
COPY scripts/smoke_test.py        /opt/scripts/smoke_test.py

# Build the RRUFF index. This downloads ~229 MB of zipped spectra,
# preprocesses each, and writes a single ~50 MB npz. Build-time
# network is required; if the RRUFF servers are unreachable the build
# fails fast (intentional — a container with no reference library is
# useless).
RUN python /opt/scripts/build_rruff_index.py \
        --dataset excellent_unoriented \
        --out /opt/rruff_index.npz \
        --data-dir /opt/rruff/raw \
    && du -h /opt/rruff_index.npz \
    && rm -rf /opt/rruff   # raw zip archive no longer needed at runtime

# Round-trip smoke test: extract a random reference, perturb with mild
# noise, run the matcher, assert the truth mineral lands in top-3.
# Validates that build-time and runtime preprocessing pipelines are
# consistent and that the index isn't subtly corrupted.
RUN python /opt/scripts/smoke_test.py --seed 1 \
    && python /opt/scripts/smoke_test.py --seed 2

LABEL org.opencontainers.image.title="raman-classifier" \
      org.opencontainers.image.description="Raman mineral identification via nearest-neighbour matching against the RRUFF reference library; baked index, ramanspy preprocessing, cosine similarity. Path A (classical NN); Path B (in-house 1D-CNN) is a future follow-up." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/raman-classifier/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/barahona-research-group/RamanSPy" \
      bradleylab.model.paper="https://doi.org/10.1021/acs.analchem.4c00383" \
      bradleylab.model.task="Raman mineral identification (nearest-neighbour against RRUFF excellent_unoriented)" \
      bradleylab.model.sensor="raman-spectrum:1d" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="no learned weights; reference library is RRUFF excellent_unoriented baked at build time, citation Lafuente et al. 2015"

WORKDIR /work
CMD ["bash"]
