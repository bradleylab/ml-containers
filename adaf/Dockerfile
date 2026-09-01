# adaf — automatic detection of archaeological features in LiDAR terrain models.
#
# ZRC SAZU + Bias Variance Labs + The Discovery Programme. HRNet semantic
# segmentation and Faster R-CNN object detection over relief visualizations of
# a bare-earth DTM, for barrows, ringforts, enclosures, and "AO" (all
# archaeology). Field-tested on a 197 km2 Irish infrastructure corridor at 84%
# recall against known sites, where it also flagged 116 candidates that manual
# inspection had missed.
#
# ---------------------------------------------------------------------
# TWO LICENCES, AND THEY DIFFER. THIS MATTERS BEFORE YOU FINE-TUNE.
# ---------------------------------------------------------------------
# Code    : Apache-2.0 (ZRC SAZU)
# WEIGHTS : CC-BY-SA-4.0 -- Zenodo 10.5281/zenodo.15848663
#
# CC-BY-SA is a SHARE-ALIKE licence. Weights fine-tuned from these are an
# adapted work: running them internally and publishing detections is
# unencumbered, but RELEASING derived weights obliges you to release them under
# CC-BY-SA-4.0 too. If the lab ever needs permissively-licensed weights for a
# downstream, train from ImageNet initialisation inside this same Apache-2.0
# code instead -- that costs the archaeological pretraining and buys licence
# freedom. LICENSE.weights.md ships inside the image because CC-BY-SA requires
# the notice to travel with the work.
#
# ---------------------------------------------------------------------
# WHAT THE PRETRAINED WEIGHTS EXPECT. GET THIS WRONG AND SCORES DROP FOR
# REASONS THAT HAVE NOTHING TO DO WITH YOUR TERRAIN.
# ---------------------------------------------------------------------
# The filenames state the training regime: SLRM at 512 px tiles. SLRM is the
# Simple Local Relief Model, produced by rvt-py (which AiTLAS already depends
# on). Feed these models a hillshade, a raw DTM, or a different tile size and
# they will still return detections -- just worse ones. Match the training
# visualization.
#
# They were also trained on national ALS at roughly 0.5 m. A DTM at 0.25 m puts
# 4x more pixels on the same earthwork, so a barrow that filled a 512 px tile at
# 0.5 m fills a sixteenth of one at 0.25 m. Whether to resample down to match
# the receptive field or run native is an open question for any new survey, and
# is worth answering with a two-resolution run before trusting either.
#
# ---------------------------------------------------------------------
# PYTHON 3.9, DELIBERATELY, AND WHY IT CANNOT SIMPLY BE RAISED.
# ---------------------------------------------------------------------
# AiTLAS pins h5py<3.2.1, imagecodecs==2023.3.16 and ipykernel==6.15.0. h5py
# 3.2.0 predates Python 3.10 and publishes no cp310+ wheel, so a newer
# interpreter turns that pin into a source build that fails. Upstream's "newer
# Python should also work" is about GDAL, not about these pins. So this image
# is built on 3.9 from conda-forge, which is also where GDAL comes from --
# upstream installs GDAL/rasterio/fiona from cgohlke's win_amd64 wheels, which
# have no Linux equivalent and are the reason the published instructions are
# Windows-only.
#
# Upstream: https://github.com/EarthObservation/adaf
# Weights:  https://doi.org/10.5281/zenodo.15848663
# Paper:    Coz, Corns, Curran, Kocev & Kokalj (2026),
#           J. Archaeol. Sci: Reports 71:105733, doi:10.1016/j.jasrep.2026.105733
# Stack:    python 3.9 + torch 2.5.1 cu121, conda-forge GDAL

FROM condaforge/miniforge3:24.9.2-0

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ADAF_COMMIT=a44acdd678773666458ea50f544b5d2292bcef51

# This is NOT a CUDA base image, so it does not set these itself. Without them
# the enroot NVIDIA hook injects no driver and torch reports no GPU inside a
# job that was allocated one -- the silent failure this repo already shipped
# once, in n2n4m.
#
# THE COST OF THE TWO LINES BELOW: this image no longer starts on Compute2's
# general-cpu partition by default, because the hook looks for a GPU that is
# not allocated there. CPU jobs must override it:
#     srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# GDAL, rasterio and fiona from conda-forge: the Linux answer to upstream's
# cgohlke wheels.
#
# DO NOT PIN THESE TO EXACT VERSIONS. An earlier revision asked for
# gdal=3.9.2 / rasterio=1.3.11 / fiona=1.10.1 and conda dutifully found a
# combination satisfying all three -- against a libsqlite already in the base
# environment that was too old for the libgdal build it chose:
#
#   libgdal.so.35: undefined symbol: sqlite3_total_changes64
#   ModuleNotFoundError: No module named '_gdal'
#
# sqlite3_total_changes64 arrived in SQLite 3.37. What matters is that these
# resolve as a CONSISTENT SET, not that any one lands on a chosen version, so
# the solver is given the constraint that actually matters (libsqlite) and left
# free on the rest. --override-channels keeps defaults out of the solve, which
# is the other common source of this exact symbol error.
RUN conda install -y -n base -c conda-forge --override-channels \
        python=3.9 \
        gdal \
        rasterio \
        fiona \
        "libsqlite>=3.46" \
    && conda clean -afy

# Verify the geospatial stack HERE, immediately, rather than in the smoke test
# at the end.
#
# The build above takes about two minutes. The weight download further down
# takes most of an hour and moves 5.52 GB. When this import check lived only in
# the final smoke test, a broken GDAL cost a full 95-minute build to discover
# something provable in 120 seconds. Cheap checks go next to the step they
# check.
RUN python -c "\
from osgeo import gdal; \
import rasterio, fiona, sqlite3; \
print('gdal', gdal.__version__, '| rasterio', rasterio.__version__, '| fiona', fiona.__version__, '| sqlite', sqlite3.sqlite_version)"

# numpy is held at 1.26.4 by upstream: AiTLAS and the GDAL stack both predate
# the numpy 2 ABI break. opencv is held below 4.12 for the same reason.
RUN pip install --upgrade pip wheel \
    && pip install "numpy==1.26.4" "opencv-python-headless<4.12"

RUN pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

# ADAF pinned to a commit, not to main: the model-path handling below depends
# on how the source spells those paths today.
RUN git clone https://github.com/EarthObservation/adaf.git /opt/adaf \
    && git -C /opt/adaf checkout $ADAF_COMMIT \
    && git -C /opt/adaf rev-parse HEAD

# AiTLAS ships as a wheel vendored INSIDE the ADAF repo, pinned at 0.0.1, not
# from PyPI. Upstream of it is github.com/biasvariancelabs/aitlas (Apache-2.0).
# Installing the vendored copy is what upstream instructs; substituting the
# public package is a change of dependency, not a packaging detail.
RUN pip install /opt/adaf/installation/aitlas-0.0.1-py3-none-any.whl

ENV PYTHONPATH=/opt/adaf

# Second fail-fast gate, same reasoning as the GDAL check: AiTLAS carries tight
# pins (h5py<3.2.1, imagecodecs==2023.3.16) that can install cleanly and still
# fail to import. Prove the model classes resolve BEFORE the 5.52 GB download.
RUN python -c "\
import torch; \
from aitlas.models import FasterRCNN, HRNet; \
import rvt.vis; \
print('torch', torch.__version__, '| aitlas models:', FasterRCNN.__name__, HRNet.__name__)"

COPY fetch_weights.py /opt/fetch_weights.py
RUN python /opt/fetch_weights.py

COPY link_windows_paths.py /opt/link_windows_paths.py
RUN python /opt/link_windows_paths.py

COPY LICENSE.weights.md /opt/licenses/LICENSE.weights.md

# Build smoke test -- CPU, and honest about what it does not prove.
#
# It asserts the things a green build could otherwise hide: that the GDAL stack
# imports as a set, that AiTLAS's model classes resolve, that all eight weight
# files are present at BOTH path spellings, and that a checkpoint actually
# loads into a model rather than merely existing on disk. Loading is the part
# that matters -- eight intact TAR files prove nothing about whether AiTLAS can
# read them.
#
# It does NOT run a detection. That needs a real DTM and a GPU, and lives in
# SMOKE.md.
RUN python -c "\
import importlib.metadata as im, os, torch; \
from osgeo import gdal; \
import rasterio, fiona, rvt.vis, cv2, numpy; \
print('torch', im.version('torch'), '| numpy', numpy.__version__, '| gdal', gdal.__version__); \
print('rasterio', rasterio.__version__, '| fiona', fiona.__version__, '| cv2', cv2.__version__); \
from aitlas.models import FasterRCNN, HRNet; \
print('aitlas models resolved:', FasterRCNN.__name__, HRNet.__name__); \
w = '/opt/adaf-weights'; \
tars = sorted(f for f in os.listdir(w) if f.endswith('.tar')); \
print('weight files:', len(tars)); \
assert len(tars) == 8, tars; \
pkg = '/opt/adaf/adaf'; \
posix_ok = [t for t in tars if os.path.isfile(os.path.realpath(os.path.join(pkg, 'ml_models', t)))]; \
weird_ok = [t for t in tars if os.path.isfile(os.path.realpath(os.path.join(pkg, chr(46) + chr(92) + 'ml_models' + chr(92) + t)))]; \
print('resolving via posix names    :', len(posix_ok)); \
print('resolving via backslash names:', len(weird_ok)); \
assert len(posix_ok) == 8 and len(weird_ok) == 8, (len(posix_ok), len(weird_ok)); \
m = FasterRCNN({'num_classes': 2, 'learning_rate': 0.0001, 'pretrained': False, 'use_cuda': False, 'metrics': ['map']}); \
m.prepare(); \
m.load_model(os.path.join(w, 'OD_barrow.tar')); \
n = sum(p.numel() for p in m.model.parameters()); \
print('OD_barrow loaded; params', n); \
assert n > 1_000_000, n; \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="adaf" \
      org.opencontainers.image.description="ADAF — automatic detection of archaeological features (barrows, ringforts, enclosures) in LiDAR terrain models. HRNet segmentation + Faster R-CNN detection over SLRM relief visualizations" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/adaf/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0 AND CC-BY-SA-4.0" \
      bradleylab.model.upstream="https://github.com/EarthObservation/adaf" \
      bradleylab.model.paper="https://doi.org/10.1016/j.jasrep.2026.105733" \
      bradleylab.model.task="archaeological feature detection from LiDAR DTM relief visualizations" \
      bradleylab.model.sensor="lidar:airborne DTM (trained on ~0.5 m national ALS)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.classes="barrow, ringfort, enclosure, AO (all archaeology)" \
      bradleylab.model.input="SLRM relief visualization at 512 px tiles — matching the training regime; see README" \
      bradleylab.model.weights.source="zenodo:10.5281/zenodo.15848663 (CC-BY-SA-4.0; 8 TAR files, 5.52 GB, baked at /opt/adaf-weights)" \
      bradleylab.model.licence.note="code Apache-2.0; WEIGHTS CC-BY-SA-4.0 — share-alike attaches to fine-tuned derivatives, so releasing derived weights obliges CC-BY-SA-4.0. Notice shipped at /opt/licenses/LICENSE.weights.md"

WORKDIR /work
CMD ["python", "-c", "print('adaf ready — weights at /opt/adaf-weights, see README.md')"]
