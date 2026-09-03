# afwizard — spatially adaptive ground-point filtering for archaeological lidar.
#
# Heidelberg Scientific Software Center. The point of this tool is that ONE
# ground filter applied to a whole survey is wrong when the survey covers
# varied terrain: a setting that works on open slope oversmooths in dense
# vegetation, and vice versa. AFwizard lets you segment the area into polygons
# and apply a different filter and parameter set per segment, producing one
# coherent DTM plus the pipelines and metadata that made it.
#
# ---------------------------------------------------------------------
# WHY THIS IS IN THE CATALOG AND THE FOREST TOOLS ARE NOT A SUBSTITUTE.
# ---------------------------------------------------------------------
# fsct, forainet, forestformer3d and myria3d all classify ground, and all were
# designed around forest STRUCTURE — where the goal is a clean canopy/stem/
# ground split and the ground surface is a means to an end. Archaeological
# prospection wants the opposite emphasis: the anthropogenic micro-relief ON
# the ground surface, features of 20-50 cm relief that an aggressive filter
# smooths away as noise before anything downstream sees them.
#
# Doneus, Höfle, Kempf, Daskalakis & Shinoto (2022), Archaeological Prospection
# 29(4):503-524 (doi:10.1002/arp.1873) is the paper behind this tool and states
# the problem in those terms.
#
# NOT AN AUTOMATIC TOOL. It does not segment the landscape for you — you supply
# the polygon layer. That is the human-in-the-loop part, and it is by design.
#
# ---------------------------------------------------------------------
# PDAL ONLY. THE OTHER TWO BACKENDS ARE PROPRIETARY.
# ---------------------------------------------------------------------
# afwizard.pdal, afwizard.lastools and afwizard.opals are all importable, but
# only PDAL is open source and installed here. LAStools and OPALS are
# commercial and need licences this lab does not hold; upstream ships a
# separate docker/proprietary.dockerfile for sites that do. Calling
# set_lastools_directory() or set_opals_directory() in this image will not find
# anything, and that is expected rather than broken.
#
# ---------------------------------------------------------------------
# NOT BUILT FROM UPSTREAM'S DOCKERFILE, DELIBERATELY.
# ---------------------------------------------------------------------
# Upstream's Dockerfile builds on jupyter/base-notebook:2022-06-06 and installs
# from a git checkout, because the tool's primary interface is a JupyterLab
# widget. That base is pinned to a 2022 image and carries a full JupyterLab
# stack this image has no use for on a batch scheduler.
#
# afwizard 1.0.1 and afwizard-library are both published on conda-forge, which
# is upstream's own distribution channel, so this installs from there instead.
# Same software, current dependencies, no notebook server. The interactive
# widget apps (pipeline_tuning, select_best_pipeline) still import — they just
# need a Jupyter frontend to be useful, which is a workstation activity, not a
# SLURM one. The batch entry point, apply_adaptive_pipeline, is what runs here.
#
# Upstream: https://github.com/ssciwr/afwizard
# Docs:     https://afwizard.readthedocs.io
# Paper:    https://doi.org/10.1002/arp.1873
# Licence:  MIT. No model weights — this is software, not a trained model.

FROM condaforge/miniforge3:24.9.2-0

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# geojson is pinned below 3 because afwizard 1.0.1 was written against 2.x and
# geojson 3 changed FeatureCollection's constructor to take the features list
# rather than the whole collection. With 3.x installed, load_segmentation()
# fails with "string indices must be integers" -- the loader walking a dict's
# keys as if they were features. Upstream fixed this in commit 8fdbc5f on
# 2026-07-20 but has cut no release since v1.0.1 (2025-11-14), so conda-forge
# still ships the mismatch. Drop the pin when 1.0.2 lands.
#
# A dedicated environment, not `base`. Installing into miniforge's base means
# downgrading the Python it ships with, which leaves packages built against the
# newer stack behind — that is how the adaf image failed twice with an
# undefined libsqlite symbol before being rebuilt this way.
# hdf5 is listed explicitly even though nothing here reads HDF directly.
# Without it, conda resolves a pdal whose HDF and icebridge reader plugins
# link against a libhdf5_cpp that is not installed, and EVERY invocation --
# including `afwizard --help` -- prints several hundred characters of
#
#   (PDAL Error) Can't load library .../libpdal_plugin_reader_hdf.so:
#   libhdf5_cpp.so.320: cannot open shared object file
#
# to stderr. Nothing this lab does needs those readers, so the capability is
# not the point: the noise is. It fires before real output, interleaves with
# it, and would bury an actual error in a batch log.
RUN conda create -y -n afwizard -c conda-forge --override-channels \
        python=3.11 \
        afwizard \
        afwizard-library \
        pdal \
        python-pdal \
        gdal \
        hdf5 \
        "geojson<3" \
    && conda clean -afy

ENV PATH=/opt/conda/envs/afwizard/bin:$PATH \
    CONDA_DEFAULT_ENV=afwizard

# PROJ_LIB is set for the same reason upstream sets it: without it the conda
# PROJ database is not found and every reprojection fails at runtime rather
# than at import, which is the worst place to discover it. Upstream calls the
# underlying situation a bug; the workaround is theirs.
ENV PROJ_LIB=/opt/conda/envs/afwizard/share/proj

# The tuner: objective-driven parameter search, scored on the classified cloud.
# It drives the PDAL CLI, so it needs nothing beyond what is already here, and
# it is what an agent calls -- see README "Tuning for an objective".
COPY tuner /opt/afwizard-tuner
RUN pip install --no-deps /opt/afwizard-tuner \
    && echo "afwizard-tune: $(afwizard-tune list-criteria --json | python -c 'import json,sys; print(len(json.load(sys.stdin)))') criteria registered"

# The tuner's numeric stack, exercised without PDAL: a synthetic 60x60 DTM
# with one planted 10 m bump must show coherent relief at 2-10 m, and a flat
# one must not. This catches a scipy/rasterio import problem or a broken
# variogram at build time rather than on a compute node.
RUN python -c "\
import numpy as np; \
from rasterio.transform import from_origin; \
from afwizard_tuner import criteria; \
from afwizard_tuner.backend import Evaluation; \
yy, xx = np.mgrid[0:60, 0:60].astype(float); \
bump = 0.5 * np.exp(-(((yy - 30) ** 2 + (xx - 30) ** 2) / (2 * 4.0 ** 2))); \
mk = lambda z: Evaluation(vendor_class=np.empty(0, int), classification=np.empty(0, int), height_above_reference=np.empty(0), xyz=np.empty((0, 3)), dtm=z, mask=np.ones_like(z, bool), reference=np.zeros_like(z), dtm_transform=from_origin(0, 60, 1, 1), dtm_resolution_m=1.0, seconds=0.0); \
sv_bump = criteria.compute('structured_variance', mk(bump), scale_lo_m=2.0, scale_hi_m=10.0); \
sv_flat = criteria.compute('structured_variance', mk(np.zeros((60, 60))), scale_lo_m=2.0, scale_hi_m=10.0); \
rv_bump = criteria.compute('recovered_variance', mk(bump), scale_lo_m=2.0, scale_hi_m=10.0); \
print('structured_variance bump', round(sv_bump, 6), '| flat', round(sv_flat, 6), '| recovered', round(rv_bump, 6)); \
assert sv_bump > 1e-4 and sv_flat < 1e-9 and rv_bump > 1e-4, (sv_bump, sv_flat, rv_bump); \
print('tuner numeric stack OK')"

# Build smoke test — imports, versions, and the two things that would otherwise
# fail silently at runtime: that the community filter library actually resolves
# to pipelines (an empty library imports fine and filters nothing), and that
# PROJ has a real database behind it.
RUN python -c "\
import afwizard, pdal, json; \
from osgeo import gdal, osr; \
from afwizard.execute import apply_adaptive_pipeline; \
from afwizard.dataset import DataSet; \
from afwizard.filter import load_filter; \
print('afwizard', afwizard.__version__, '| pdal', pdal.__version__, '| gdal', gdal.__version__); \
print('batch entry point:', apply_adaptive_pipeline.__name__); \
srs = osr.SpatialReference(); \
assert srs.ImportFromEPSG(32615) == 0, 'PROJ database unusable — check PROJ_LIB'; \
print('PROJ ok: EPSG:32615 resolved'); \
import afwizard.library as lib; \
print('filter library module:', lib.__name__); \
import geojson, tempfile, os; \
assert int(geojson.__version__.split('.')[0]) < 3, geojson.__version__; \
fc = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {'pipeline': 'x'}, 'geometry': {'type': 'Polygon', 'coordinates': [[[0,0],[1,0],[1,1],[0,0]]]}}]}; \
d = tempfile.mkdtemp(); path = os.path.join(d, 's.geojson'); \
open(path, 'w').write(__import__('json').dumps(fc)); \
seg = afwizard.load_segmentation(path); \
assert len(seg['features']) == 1 and seg['features'][0]['properties']['pipeline'] == 'x', 'segmentation did not round-trip'; \
print('load_segmentation ok under geojson', geojson.__version__); \
import subprocess; \
res = subprocess.run(['pdal', '--version'], capture_output=True, text=True); \
print('pdal stderr on startup:', repr(res.stderr[:200])); \
assert 'load library' not in res.stderr, 'PDAL plugin failed to load - a shared library dependency is missing; see the hdf5 note above'; \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="afwizard" \
      org.opencontainers.image.description="AFwizard — spatially adaptive ground-point filtering, applying a different filter per terrain segment to preserve archaeological micro-relief a single global filter would smooth away" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/afwizard/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/ssciwr/afwizard" \
      bradleylab.model.paper="https://doi.org/10.1002/arp.1873" \
      bradleylab.model.task="spatially adaptive ground-point filtering for bare-earth DTM generation" \
      bradleylab.model.sensor="lidar:point cloud (LAS/LAZ)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="none — PDAL-based geometric filtering, no neural network and no CUDA path" \
      bradleylab.model.weights.source="none — this is software, not a trained model" \
      bradleylab.model.api="afwizard-tune list-criteria | tune --objective F --outdir D | wire; objective is YAML the caller composes (constraints, maximize, report); every evaluation is returned" \
      bradleylab.model.pins="geojson<3 until afwizard>1.0.1 ships the 8fdbc5f fix" \
      bradleylab.model.licence.note="MIT throughout. The LAStools and OPALS backends are proprietary and NOT installed; only the open-source PDAL backend is available in this image"

WORKDIR /work
CMD ["python", "-c", "import afwizard; print('afwizard', afwizard.__version__, '— batch entry point is afwizard.execute.apply_adaptive_pipeline; see README.md')"]
