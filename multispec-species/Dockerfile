# Multispectral per-crown species classification (v1).
#
# Reads Altum-PT 5-band reflectance mosaics + lidar-derived crown
# polygons, extracts zonal spectral stats + derived vegetation indices,
# and trains an XGBoost multiclass species classifier using ForestGEO
# census labels.
#
# CPU-only by design — Approach 1 of the Tyson multispec plan. Future
# deep-learning approaches (CNN on crown chips, DeepTreeAttention,
# fusion with lidar) will get their own containers.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs for GDAL-backed rasterio + geopandas + shapely.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev \
    libgeos-dev libproj-dev \
    gcc g++ make \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# Scientific Python stack. Pin GDAL to the system version to avoid
# rasterio/GDAL ABI mismatch.
RUN GDAL_VERSION=$(gdal-config --version) \
    && pip install --upgrade pip \
    && pip install \
        "numpy>=1.26,<3" \
        "pandas>=2.2" \
        "pyarrow>=15" \
        "shapely>=2.0" \
        "pyproj>=3.6" \
        "GDAL==${GDAL_VERSION}.*" \
        "rasterio>=1.4" \
        "rasterstats>=0.20" \
        "geopandas>=1.0" \
        "scikit-learn>=1.5" \
        "xgboost>=2.1" \
        "joblib>=1.4" \
        awscli

WORKDIR /opt/multispec
COPY scripts/ ./
RUN chmod +x *.py 2>/dev/null || true

# Smoke-test imports at build time.
RUN python -c "import geopandas, rasterio, rasterstats, xgboost, sklearn, joblib, pyarrow; print('multispec stack OK')"

ENTRYPOINT ["python"]
