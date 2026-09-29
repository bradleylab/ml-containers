# fraxtex — pixel-wise geological fracture segmentation of outcrop imagery
# (fraXteX; Fatihi, Caldeira, Beucler, Thiele and Samsu).
#
# Four ONNX models trained on FraXet, 8,953 co-registered 256x256 RGB+DEM
# patches of UAV orthomosaics from three published outcrop datasets. Two
# modalities, which the authors keep as separate arms of the benchmark:
#   rgb/     U-Net (from scratch) and SAM 2 (frozen hiera-tiny encoder,
#            fine-tuned decoder, prompt-free); raw 0-255 RGB in, normalization
#            inside the graph.
#   rgbdem/  U-Net and SegFormer (ResNet-34 encoder) on RGB plus a DEM band;
#            0-1 input, no normalization in the graph, fixed 256x256 tiles.
# Each file carries its input contract in metadata_props, and the lab's
# scripts read it from there rather than hard-coding it (scripts/fraxtex_onnx.py).
#
# The upstream training code (github.com/ayoubft/fractex2D.pt) has no license,
# so none of it is in this image. It was read to learn the preprocessing and
# evaluation protocol; fraxtex-predict, fraxtex-eval-fraxet, fraxet-stage and
# fraxtex-selftest are the lab's own. Inference needs only ONNX Runtime.
#
# Weights:  https://huggingface.co/ayoubft/fraXteX @ 890ccb4 (MIT)
#           RGB+DEM pair also on Zenodo, doi:10.5281/zenodo.17866853 (CC-BY-4.0)
# Data:     FraXet, doi:10.5281/zenodo.17069947 (CC-BY-4.0), fetched at run time
# Paper:    https://doi.org/10.5194/egusphere-2026-1097 (EGUsphere preprint)

FROM python:3.12-slim-bookworm

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the user's ~/.local off sys.path: enroot bind-mounts
# $HOME, and a host package shadowing an image one has broken images in this
# catalog before.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1

# libexpat1: the rasterio manylinux wheel that fraxtex-eval-fraxet imports links
# against the system expat, which python:slim does not ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime-gpu 1.26.0 is the last PyPI release whose default wheel targets
# CUDA 12 (built with 12.8, cuDNN 9, per the ONNX Runtime CUDA provider table;
# 1.27 moved to CUDA 13). The [cuda,cudnn] extras bring the CUDA runtime,
# cuBLAS, cuFFT, cuRAND, NVRTC and cuDNN as pip wheels, so no CUDA base image
# is needed. Compute2's driver, 580.105.08, supports CUDA up to 13.0.
# constraints.txt is the full resolved set; floating ranges are what broke
# other images here months after they last built.
COPY constraints.txt /opt/constraints.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -c /opt/constraints.txt \
        "onnxruntime-gpu[cuda,cudnn]==1.26.0" numpy rasterio scikit-image scipy

# The weights are small (470 MB for all four) and MIT-licensed, so they are
# baked. Each file is fetched at a pinned Hugging Face revision and checked
# against the sha256 the Hub reports for it, so a moved or re-uploaded file
# fails the build. The model card comes along because it is where the license,
# citation and contract tables are stated.
ARG FRAXTEX_REV=890ccb47e6a8e015126bab5dab13cbeca39066de
ARG HF=https://huggingface.co/ayoubft/fraXteX/resolve/${FRAXTEX_REV}
RUN mkdir -p /opt/fraxtex/models/rgb /opt/fraxtex/models/rgbdem \
    && cd /opt/fraxtex/models \
    && curl -fsSL -o rgb/unet-rgb.onnx           "$HF/rgb/unet-rgb.onnx" \
    && curl -fsSL -o rgb/sam2-rgb.onnx           "$HF/rgb/sam2-rgb.onnx" \
    && curl -fsSL -o rgbdem/unet-rgbdem.onnx     "$HF/rgbdem/unet-rgbdem.onnx" \
    && curl -fsSL -o rgbdem/segformer-rgbdem.onnx "$HF/rgbdem/segformer-rgbdem.onnx" \
    && curl -fsSL -o MODEL_CARD.md               "$HF/README.md" \
    && printf '%s  %s\n' \
        b50e3c714a0447fae3da175c999c239b503cb98c0c3363e8d790f7c0fd7948fe rgb/unet-rgb.onnx \
        a2eb4608b9db305aa3ccf714a521a0915155cc789f87c92ac5582a47fe610267 rgb/sam2-rgb.onnx \
        126c71f7f4bc3fa0ef173a82dee84d19520260cc43e2c5fd79acff8fb3801ec8 rgbdem/unet-rgbdem.onnx \
        07f53bce423590b03391417037576151f33c9d8b2a58b033df57d80c998a4403 rgbdem/segformer-rgbdem.onnx \
        e5c94ee87b2815e98d941a9a30335985b6137d42781e0ba9ffcfc04820302de9 MODEL_CARD.md \
       | sha256sum -c -

# sam2-rgb.onnx contains Meta's SAM 2.0 hiera-tiny image encoder unchanged
# (frozen during fine-tuning). Meta releases those checkpoints under
# Apache-2.0, which asks that the license travel with redistributed copies.
ARG SAM2_REF=2b90b9f5ceec907a1c18123530e92e794ad901a4
RUN mkdir -p /opt/licenses \
    && curl -fsSL -o /opt/licenses/SAM2-LICENSE \
        "https://raw.githubusercontent.com/facebookresearch/sam2/${SAM2_REF}/LICENSE" \
    && echo "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  /opt/licenses/SAM2-LICENSE" \
       | sha256sum -c -

COPY scripts/fraxtex_onnx.py /opt/fraxtex/lib/
COPY scripts/fraxtex-predict scripts/fraxtex-eval-fraxet scripts/fraxet-stage scripts/fraxtex-selftest \
     /usr/local/bin/
ENV PYTHONPATH=/opt/fraxtex/lib \
    FRAXTEX_MODEL_DIR=/opt/fraxtex/models

RUN mkdir -p /work

# Build smoke test, offline and on CPU: prints versions, asserts the CUDA
# provider is compiled in (no GPU at build time, so it cannot run here), then
# opens all four models, prints each embedded contract and runs noise that
# matches it through each, checking output shape and range.
RUN python -X faulthandler /usr/local/bin/fraxtex-selftest --device cpu

LABEL org.opencontainers.image.title="fraxtex" \
      org.opencontainers.image.description="fraXteX pixel-wise fracture segmentation of outcrop orthomosaics: U-Net and SAM 2 on RGB, U-Net and SegFormer on RGB+DEM, trained on FraXet; ONNX Runtime on CUDA 12, weights baked, tiled GeoTIFF inference" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/fraxtex/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      bradleylab.model.upstream="https://huggingface.co/ayoubft/fraXteX" \
      bradleylab.model.paper="https://doi.org/10.5194/egusphere-2026-1097" \
      bradleylab.model.task="pixel-wise fracture-trace probability segmentation of outcrop imagery (binary; per-pixel sigmoid)" \
      bradleylab.model.sensor="image:rgb (UAV orthomosaics of outcrops, mm- to cm-scale GSD), optional co-registered DEM" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.weights.source="baked — huggingface.co/ayoubft/fraXteX @ 890ccb47e6a8e015126bab5dab13cbeca39066de, four ONNX files (470 MB) at /opt/fraxtex/models, sha256-checked" \
      bradleylab.model.licence.note="Weights MIT (Hugging Face ayoubft/fraXteX). The RGB+DEM U-Net and SegFormer are also released on Zenodo (doi:10.5281/zenodo.17866853) under CC-BY-4.0. sam2-rgb embeds Meta's frozen SAM 2 hiera-tiny encoder, Apache-2.0; license at /opt/licenses/SAM2-LICENSE. FraXet data CC-BY-4.0, not baked. Upstream training code (ayoubft/fractex2D.pt) carries no license and is not included; the inference and evaluation scripts are the lab's own"

WORKDIR /work
CMD ["bash", "-c", "echo 'fraxtex ready: fraxtex-predict --model {unet-rgb,sam2-rgb,unet-rgbdem,segformer-rgbdem} IN.tif OUT.tif (see README.md)'"]
