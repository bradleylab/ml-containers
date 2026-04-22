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
RUN R -e "install.packages(c('lidR', 'crownsegmentr', 'RCSF', 'data.table', 'optparse', 'arrow', 'jsonlite'), \
          repos='https://cloud.r-project.org', Ncpus=4)"

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

ENTRYPOINT ["/opt/crownseg/process_tile.sh"]
