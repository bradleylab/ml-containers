# syntax=docker/dockerfile:1
# hunting-pits — detection of hunting pits in airborne-lidar DEMs
# (Lidberg, Westphal, Brax, Sandström and Östlund, 2024, J. Field Archaeol. 49:395–405).
#
# A TensorFlow U-Net reads the profile curvature of a 0.5 m DEM, computed with
# WhiteboxTools and clipped to [-2, 1] before scaling to [0, 1], and labels each
# pixel pit or background. Predicted pixels are merged into polygons, and
# polygons of 30 square map units or less (m² in a metric CRS) are dropped.
# The model was pretrained on lunar impact craters and trained on 2,519
# hand-mapped hunting pits in northern Sweden, with 20% of the 250 × 250 px
# chips held out at random for testing.
# On that test set it is the paper's best of ten terrain indices, two DEM
# resolutions and three architectures: object-based F1 0.76, recall 70%, and
# 15% of its detections false (precision 85%).
#
# Code and weights: MIT. The weights are the repository's own trained.h5 for
# this model, baked at /opt/hunting-pits/weights and SHA-256-checked at build.
# Upstream's ready-made image (williamlidberg/hunting_pits:v1, 13.5 GB) is not
# re-hosted: this recipe rebuilds from the repository at a pinned commit.
#
# Upstream: https://github.com/williamlidberg/Detection-of-hunting-pits-using-airborne-laser-scanning-and-deep-learning @ e45f3ca
# Paper:    https://doi.org/10.1080/00934690.2024.2364428 (CC BY 4.0)

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
        ca-certificates git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# GDAL, geopandas and the TIFF readers from conda-forge in their own
# environment, as adaf does. numpy stays below 2 because TensorFlow 2.15 is
# built against the numpy 1 ABI.
#
# WhiteboxTools comes from the same solve, pinned to one conda-forge build.
# Upstream's inference script calls whitebox.download_wbt() on import, which
# fetches whatever binary whiteboxgeo.com serves that day, so the version that
# computes the profile curvature would drift. v2.3.0 was the current release
# when upstream built its image (2023-08-31). It can no longer be compiled from
# its tag: its lock file pins the crate tsp-rs 0.1.0, which was removed from
# crates.io, and the name was registered again in January 2026 by another
# author for an unrelated crate, so cargo --locked refuses the checksum. The
# conda-forge build below was compiled on 2023-03-29, before that.
RUN conda create -y -n hp -c conda-forge --override-channels \
        python=3.11 "numpy=1.26" gdal geopandas tifffile imagecodecs \
        "whitebox_tools=2.3.0=py311hfe55011_0" \
    && conda clean -afy \
    && /opt/conda/envs/hp/bin/whitebox_tools --version
ENV PATH=/opt/conda/envs/hp/bin:$PATH \
    CONDA_DEFAULT_ENV=hp \
    PYTHONNOUSERSITE=1 \
    PROJ_LIB=/opt/conda/envs/hp/share/proj

# The paper used TensorFlow 2.6 and upstream's image TensorFlow 2.8 (NGC
# 22.04). 2.15 is the last release whose Keras is Keras 2, which reads the
# legacy HDF5 weights through the same load_weights call. torch is imported by
# upstream's utils/unet.py for an unused PyTorch model class (CA-Net), so the
# CPU wheel is enough.
RUN pip install "tensorflow[and-cuda]==2.15.1" \
    && pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install whitebox==2.3.6

# The whitebox package looks for the binary beside its own module and, with
# WBT_PATH set, skips its download step, so the import-time download in
# upstream's script becomes a no-op without editing the script.
RUN WB=$(python -c "import os, whitebox; print(os.path.dirname(whitebox.__file__))") \
    && ln -s /opt/conda/envs/hp/bin/whitebox_tools "$WB/whitebox_tools"
ENV WBT_PATH=/opt/conda/envs/hp/bin

# Only the files inference needs, at the pinned commit: the repository is
# 1.2 GB with its notebooks, figures and training data. The test chip and
# upstream's own prediction for it serve as the build smoke test.
ARG HP_GIT_SHA=e45f3cae615d86f16bafb962d2208b32fa600f80
ARG HP_REPO=https://github.com/williamlidberg/Detection-of-hunting-pits-using-airborne-laser-scanning-and-deep-learning.git
RUN git clone --filter=blob:none --no-checkout "${HP_REPO}" /opt/hunting-pits/src \
    && cd /opt/hunting-pits/src \
    && git sparse-checkout set --no-cone /LICENSE /semantic_segmentation/inference_unet_from_dem.py \
        /semantic_segmentation/utils/ /semantic_segmentation/trained_models/UNet/05m/ /data/test_chip/ \
    && git checkout "${HP_GIT_SHA}" \
    && rm -rf .git semantic_segmentation/utils/ca_net/saved_models semantic_segmentation/utils/ca_net/pictures \
        semantic_segmentation/utils/ca_net/result semantic_segmentation/utils/__pycache__ \
    && echo "${HP_GIT_SHA}" > /opt/hunting-pits/UPSTREAM_SHA

# The best model's weights, where upstream's README points its inference
# command. The other two 0.5 m U-Nets in that folder stay beside it.
ARG HP_WEIGHTS_SHA256=b058c491ced798865b25b6eac1b3627c06a9fe5e21e4eb995f1c1cb8d9a8614c
RUN mkdir -p /opt/hunting-pits/weights \
    && cp /opt/hunting-pits/src/semantic_segmentation/trained_models/UNet/05m/profile_curvature1/trained.h5 \
        /opt/hunting-pits/weights/unet_05m_profile_curvature.h5 \
    && echo "${HP_WEIGHTS_SHA256}  /opt/hunting-pits/weights/unet_05m_profile_curvature.h5" | sha256sum -c -

# Upstream writes its intermediate rasters to a hard-coded
# /workspace/temp_inference/. On Compute2 the image filesystem is read-only, so
# that path is a link into /tmp, which the wrapper creates and locks per run.
RUN mkdir -p /workspace && ln -s /tmp/hunting-pits-inference /workspace/temp_inference

COPY scripts/hunting-pits-infer scripts/hunting-pits-selftest /usr/local/bin/
RUN mkdir -p /work

# Build smoke test, with no network so nothing can be fetched.
RUN --network=none python -c "\
import importlib.metadata as im; \
import tensorflow as tf, torch, whitebox; \
from osgeo import gdal; import geopandas, tifffile; \
print('tensorflow', tf.__version__, '| keras', im.version('keras'), '| torch', torch.__version__, \
      '| whitebox', im.version('whitebox'), '| gdal', gdal.__version__, '| geopandas', im.version('geopandas'))"
# Upstream's script, unmodified, on upstream's test chip with the baked
# weights; prints how its polygons compare with the prediction upstream
# committed for the same chip. CPU only here: the runner has no GPU.
RUN --network=none NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES="" hunting-pits-selftest

LABEL org.opencontainers.image.title="hunting-pits" \
      org.opencontainers.image.description="Hunting-pit detection in 0.5 m airborne-lidar DEMs — TensorFlow U-Net on WhiteboxTools profile curvature, polygon output; lunar-crater pretraining, trained on 2,519 pits in northern Sweden; weights baked" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/hunting-pits/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/williamlidberg/Detection-of-hunting-pits-using-airborne-laser-scanning-and-deep-learning" \
      bradleylab.model.paper="https://doi.org/10.1080/00934690.2024.2364428" \
      bradleylab.model.task="semantic segmentation of hunting pits (dug hollows) in lidar DEMs, vectorized to polygons" \
      bradleylab.model.sensor="lidar:airborne DEM at 0.5 m (trained on Swedish national ALS, 1–2 points/m²)" \
      bradleylab.model.input="0.5 m DEM GeoTIFF tiles; profile curvature computed in the image with WhiteboxTools 2.3.0 (conda-forge)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.domain="archaeology" \
      bradleylab.model.weights.source="upstream repository, semantic_segmentation/trained_models/UNet/05m/profile_curvature1/trained.h5 @ e45f3ca (MIT), baked at /opt/hunting-pits/weights" \
      bradleylab.model.licence.note="MIT code and weights (upstream LICENSE, Copyright (c) 2021 williamlidberg); paper CC BY 4.0" \
      bradleylab.upstream.git_sha="${HP_GIT_SHA}" \
      bradleylab.wbt.version="2.3.0 (conda-forge py311hfe55011_0)"

WORKDIR /work
CMD ["bash", "-c", "echo 'hunting-pits ready: hunting-pits-infer <dem_dir> <out_dir> (see README.md)'"]
