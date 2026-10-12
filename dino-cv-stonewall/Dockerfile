# syntax=docker/dockerfile:1
# dino-cv-stonewall — segmentation of historic dry-stone walls in lidar terrain
# derivatives (Huang, Islam, Armstrong, Bell, Khoshelham and Tomko, 2026,
# Science of Remote Sensing 13:100372).
#
# Two encoders, one reading the Multi-directional Hillshade (MHS) and one the
# Visualization for Archaeological Topography (VAT) of a 10 cm DEM, feed one
# decoder (a U-Net for the ResNets, a DPT head for the ViT) that outputs a wall
# probability per pixel. Each encoder was pretrained with DINO-CV, a
# self-distillation scheme across the two views, on 188,006 unlabeled tiles of
# the Budj Bim Cultural Landscape (Victoria, Australia), then fine-tuned on
# 7,102 tiles with hand-drawn masks of European historic dry-stone walls near
# Tae Rak (Lake Condah). Tiles are 400 × 400 px resized to 256 × 256.
# Evaluated by six-fold leave-one-area-out cross-validation, pixel metrics
# averaged over tiles: mIoU 68.6% and F1 81.4% for WideResNet-50-2, 68.3% and
# 81.2% for ResNet-50, 66.8% and 80.1% for ViT-S/16; WideResNet-50-2 keeps
# 63.8% mIoU fine-tuned on 10% of the labels.
#
# ---------------------------------------------------------------------
# RESTRICTED USE. WEIGHTS ARE NOT BAKED.
# ---------------------------------------------------------------------
# The repository's LICENSE is the Creative Commons Attribution-NonCommercial
# 4.0 International legal code. Its README adds that the repository holds
# culturally sensitive information, that data and models are for academic
# research and education only, and that commercial use is prohibited. The
# paper states that all data and derived results are owned by the Gunditj
# Mirring Traditional Owners Corporation (GMTOC), with the research done under
# a memorandum of understanding. The image carries the upstream code under
# CC BY-NC 4.0 with its license text. The fine-tuned checkpoints are not in the
# image and are never fetched into it: the lab holds a private copy and mounts
# it at /weights at run time.
#
# Input contract: MHS and VAT tiles must be computed beforehand. The paper
# names the Relief Visualization Toolbox but gives no parameters for either
# view, and the repository has no code that makes them, so this image does not
# compute them (see README.md).
#
# Upstream: https://github.com/MLinArcheaomatics/BudjBimStoneWall-DINO-CV @ fa4e65a (CC BY-NC 4.0)
# Weights:  University of Melbourne Mediaflux share links in upstream's README
# Paper:    https://doi.org/10.1016/j.srs.2026.100372 (CC BY 4.0)

FROM condaforge/miniforge3:24.9.2-0

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# GDAL (for osgeo_utils.gdal_sieve, which upstream's prediction uses), rasterio,
# geopandas and rio-tiler from conda-forge in their own environment, as adaf
# does. rio-tiler is imported at module level by upstream's dataset code.
RUN conda create -y -n stonewall -c conda-forge --override-channels \
        python=3.11 "numpy<2" gdal rasterio geopandas shapely rio-tiler tifffile \
        scikit-learn pyyaml tqdm matplotlib \
    && conda clean -afy
ENV PATH=/opt/conda/envs/stonewall/bin:$PATH \
    CONDA_DEFAULT_ENV=stonewall \
    PYTHONNOUSERSITE=1 \
    PROJ_LIB=/opt/conda/envs/stonewall/share/proj

# Upstream pins no versions. torch 2.5.1 cu121 is the build adaf runs on the
# H100s. opencv, torchinfo, torchmetrics and tensorboard are module-level
# imports of upstream's utilities and trainer; timm only for its supervised
# ViT baseline, which inference does not use.
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install "opencv-python-headless<4.12" torchinfo torchmetrics tensorboard timm

ARG STONEWALL_GIT_SHA=fa4e65a9eca87f253abb40ba09d2905eae247a4c
RUN mkdir -p /opt/stonewall/src /opt/licenses \
    && curl -fsSL "https://github.com/MLinArcheaomatics/BudjBimStoneWall-DINO-CV/archive/${STONEWALL_GIT_SHA}.tar.gz" \
        | tar -xz --strip-components=1 -C /opt/stonewall/src \
    && rm -rf /opt/stonewall/src/exp /opt/stonewall/src/figs \
    && cp /opt/stonewall/src/LICENSE /opt/licenses/LICENSE.dino-cv-stonewall.CC-BY-NC-4.0 \
    && echo "${STONEWALL_GIT_SHA}" > /opt/stonewall/UPSTREAM_SHA
ENV STONEWALL_HOME=/opt/stonewall/src \
    PYTHONPATH=/opt/stonewall/src

COPY scripts/stonewall-predict scripts/stonewall-selftest /usr/local/bin/
RUN mkdir -p /weights /work

# Build smoke test, with no network so nothing can be fetched.
RUN --network=none python -c "\
import importlib.metadata as im, torch; \
from osgeo import gdal; from osgeo_utils.gdal_sieve import gdal_sieve; \
import seg_utils, data.dataset, model.segmentation, train.trainer; \
print('torch', torch.__version__, '| torchvision', im.version('torchvision'), '| rasterio', im.version('rasterio'), \
      '| gdal', gdal.__version__, '| rio-tiler', im.version('rio-tiler'))"
# All three architectures built as upstream builds them, a checkpoint written
# in upstream's format from random weights, loaded back strictly, and the
# prediction script run end to end on two synthetic 400 × 400 tile pairs.
RUN --network=none NVIDIA_VISIBLE_DEVICES=void stonewall-selftest

LABEL org.opencontainers.image.title="dino-cv-stonewall" \
      org.opencontainers.image.description="DINO-CV dry-stone wall segmentation from lidar DEM derivatives (multi-directional hillshade + VAT, 10 cm) — two-view Siamese ResNet-50 / WideResNet-50-2 / ViT-S/16; restricted use: weights NOT baked, mounted at /weights" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/dino-cv-stonewall/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="CC-BY-NC-4.0" \
      bradleylab.model.upstream="https://github.com/MLinArcheaomatics/BudjBimStoneWall-DINO-CV" \
      bradleylab.model.paper="https://doi.org/10.1016/j.srs.2026.100372" \
      bradleylab.model.task="binary semantic segmentation of historic dry-stone walls in lidar terrain visualizations" \
      bradleylab.model.sensor="lidar:airborne DEM at 10 cm, as MHS and VAT relief visualizations" \
      bradleylab.model.input="paired MHS and VAT GeoTIFF tiles, same file name in mhs/ and vat/; 400 x 400 px at 10 cm in training; computed outside the image" \
      bradleylab.model.status="experimental" \
      bradleylab.model.domain="archaeology" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.use_restriction="Research and education only, no commercial use (CC BY-NC 4.0); culturally sensitive — data and derived results owned by the Gunditj Mirring Traditional Owners Corporation; weights held privately by the lab and mounted at run time, never redistributed" \
      bradleylab.model.weights.source="NOT baked — fine-tuned checkpoints (one per held-out area) from upstream's University of Melbourne Mediaflux links, held privately by the lab, mounted at /weights and SHA-256-checked before loading" \
      bradleylab.model.licence.note="Upstream LICENSE is the CC BY-NC 4.0 legal code (GitHub reports NOASSERTION); README: culturally sensitive, research/education only, commercial use prohibited; paper: data and derived results owned by GMTOC. Code included with its license text; weights not redistributed" \
      bradleylab.upstream.git_sha="${STONEWALL_GIT_SHA}"

WORKDIR /work
CMD ["bash", "-c", "echo 'dino-cv-stonewall ready: mount the lab checkpoints at /weights, then stonewall-predict --help (see README.md)'"]
