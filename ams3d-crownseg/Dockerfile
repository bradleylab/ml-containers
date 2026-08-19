# AMS3D crown segmentation for airborne UAV lidar (v1).
#
# Stack: R 4.4 + lidR + crownsegmentr (Ferraz 2016 AMS3D) + Python post-processing.
# Target: WashU RIS Compute2 CPU nodes via Pyxis/enroot.
#
# Built from rocker/geospatial which ships GDAL, PROJ, sf, terra pre-installed
# with proper system libraries. Adds PDAL for COPC writing and a lean Python
# venv for per-tile attribute extraction, NN infill, and raster products.
#
# Pattern reference: ml-containers/segment-any-tree-h100/Dockerfile.

FROM rocker/geospatial:4.4.1

ENV DEBIAN_FRONTEND=noninteractive

# System deps. Jammy's apt pdal is 2.3.x which lacks writers.copc (the
# COPC writer plugin was added upstream in PDAL 2.4). We need COPC to
# build the merged landscape deliverable, so install PDAL from
# conda-forge via miniforge instead of apt.
# awscli installed inside the Python venv below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev \
    ca-certificates curl bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Miniforge → conda-forge PDAL (2.10+) with writers.copc plugin.
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
      -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda install -y -c conda-forge pdal \
    && /opt/conda/bin/conda clean -afy
ENV PATH="/opt/conda/bin:$PATH"

# R packages from CRAN.
# lidR: point cloud I/O, height normalization, LAS attribute manipulation.
# crownsegmentr: AMS3D (Ferraz 2016) CRAN implementation.
# RCSF: Cloth Simulation Filter, used by lidR::classify_ground(csf()) fallback
#       when a tile has no pre-classified ground points (class 2).
# arrow: parquet I/O for per-tile tree attributes.
# optparse: CLI arg parsing.
# CRAN packages that are still on CRAN. `crownsegmentr` is NOT among them any
# more — see below.
#
# install.packages() only WARNS when a package is unavailable, it does not
# error, so a missing package used to surface two layers later at library().
# The explicit check turns that into an immediate, named failure.
# crownsegmentr's own Imports/LinkingTo are listed explicitly because the
# tarball install below uses repos=NULL, which does NOT resolve dependencies —
# that is what made the first archive-install attempt fail with
# "dependency 'dbscan' is not available". From its DESCRIPTION:
#   Imports:   assertthat, data.table, dbscan, lidR, methods, Rcpp, sf, terra
#   LinkingTo: BH, progress, Rcpp
# lidR, sf, terra and Rcpp come preinstalled in rocker/geospatial.
RUN R -e "pkgs <- c('RCSF', 'data.table', 'optparse', 'arrow', 'jsonlite', \
                    'dbscan', 'assertthat', 'BH', 'progress'); \
          install.packages(pkgs, repos='https://cloud.r-project.org', Ncpus=4); \
          missing <- pkgs[!pkgs %in% rownames(installed.packages())]; \
          if (length(missing)) stop('CRAN install failed for: ', paste(missing, collapse=', '))"

# crownsegmentr from the CRAN *archive* tarball, checksum-pinned.
#
# CRAN archived it on 2026-06-09 as a knock-on of lidR being archived the same
# day, itself a knock-on of rlas. `install.packages('crownsegmentr')` then
# silently found nothing — it only warns on an unavailable package — and the
# build died two layers later at `library(crownsegmentr)`.
#
# Installing from the archive rather than from GitHub, for two reasons.
# `remotes::install_github` calls the GitHub API, which rate-limits anonymous
# requests and failed here asking for a token — a build should not need
# credentials to fetch public source. And 1.0.1 is the version this image was
# validated against; GitHub HEAD has moved on since and would be a different
# package, not a newer build of the same one.
#
# The sha256 is the pin. A changed tarball fails the build instead of silently
# installing something else.
ARG CROWNSEGMENTR_VERSION=1.0.1
ARG CROWNSEGMENTR_SHA256=1b64031a665ddf3c000d590bc6bd926b3113f19673a5d1b2647b38bd61f5e6c1
RUN curl -fsSL -o /tmp/crownsegmentr.tar.gz \
        "https://cran.r-project.org/src/contrib/Archive/crownsegmentr/crownsegmentr_${CROWNSEGMENTR_VERSION}.tar.gz" \
    && echo "${CROWNSEGMENTR_SHA256}  /tmp/crownsegmentr.tar.gz" | sha256sum -c - \
    && R -e "install.packages('/tmp/crownsegmentr.tar.gz', repos=NULL, type='source'); \
             if (!requireNamespace('crownsegmentr', quietly=TRUE)) \
                 stop('crownsegmentr failed to install from the CRAN archive')" \
    && rm -f /tmp/crownsegmentr.tar.gz

# Python venv for post-processing.
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"
RUN $VENV/bin/pip install --no-cache-dir \
    'laspy[lazrs]>=2.5' \
    numpy scipy pandas \
    geopandas shapely rasterio \
    pyarrow awscli

# Project scripts.
WORKDIR /opt/crownseg
COPY scripts/ ./
RUN chmod +x ./process_tile.sh ./stage_tiles.sh

# Validate the core stack loads.
RUN R -e "library(lidR); library(crownsegmentr); library(RCSF); library(data.table); library(optparse); library(arrow); cat('R stack OK\n')"
RUN $VENV/bin/python -c "import laspy, scipy.spatial, geopandas, rasterio, pyarrow; print('Python stack OK')"
RUN pdal --version && pdal --drivers 2>/dev/null | grep -q writers.copc && echo "PDAL + COPC writer OK"

LABEL org.opencontainers.image.title="ams3d-crownseg" \
      org.opencontainers.image.description="Adaptive mean-shift 3D crown segmentation (Ferraz et al. 2016) via R crownsegmentr; classical, no learned components" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/ams3d-crownseg/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      bradleylab.model.upstream="https://github.com/Lenostatos/crownsegmentr" \
      bradleylab.model.paper="https://doi.org/10.1016/j.rse.2016.05.028" \
      bradleylab.model.task="crown segmentation, point cloud (classical)" \
      bradleylab.model.sensor="lidar:airborne" \
      bradleylab.model.status="production" \
      bradleylab.model.weights.source="none (classical algorithm)"

ENTRYPOINT ["/opt/crownseg/process_tile.sh"]
