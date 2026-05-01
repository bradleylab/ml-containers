# 3DFin — deterministic TLS stem detection + DBH estimation.
#
# Laino et al. (2024). 3DFin: a software for automated 3D forest
# inventories from terrestrial point clouds. Forestry 97:479-496.
#
# This container is the classical (non-DL) counterpart to TreeLearn /
# SAT on TLS data. CPU-only. Compute2 general-cpu.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3DFin's install-time deps want a C toolchain for dendromatics + lazrs.
# libgl1 is a transitive pyside6 requirement even when the GUI isn't used.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 libglib2.0-0 \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install 3DFin + common point-cloud tooling we'll need for pre- and
# post-processing.
RUN pip install --upgrade pip \
    && pip install \
        "3DFin" \
        "laspy[lazrs]>=2.5" \
        "numpy>=1.26,<3" \
        "pandas>=2.2" \
        "shapely>=2.0" \
        "pyproj>=3.6" \
        "geopandas>=1.0" \
        "pyarrow>=15" \
        xlsxwriter

# Sanity-check the headless CLI exists.
RUN 3DFin cli --help 2>&1 | head -5 || echo "3DFin CLI smoke-test (non-fatal at build time)"

WORKDIR /opt/tdfin
COPY scripts/ ./
RUN chmod +x *.sh *.py 2>/dev/null || true

LABEL org.opencontainers.image.title="3dfin" \
      org.opencontainers.image.description="3DFin (Laino et al. 2024) deterministic TLS stem detection + DBH; classical, CPU-only" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/3dfin/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="GPL-3.0-only" \
      bradleylab.model.upstream="https://github.com/3DFin/3DFin" \
      bradleylab.model.paper="https://doi.org/10.1093/forestry/cpae020" \
      bradleylab.model.task="stem detection + DBH (classical)" \
      bradleylab.model.sensor="lidar:tls" \
      bradleylab.model.status="production-capable" \
      bradleylab.model.weights.source="none (classical algorithm)"

ENTRYPOINT ["bash"]
