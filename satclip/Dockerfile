# satclip — SatCLIP location encoders (Klemmer et al., AAAI 2025).
#
# SatCLIP trains a location encoder and an image encoder contrastively on
# S2-100K: 100,000 Sentinel-2 L2A patches (256 x 256 px, 12 bands at 10 m)
# sampled roughly uniformly over land. The location encoder is the product: it
# maps (lon, lat) through a spherical-harmonics basis of L Legendre polynomials
# and a SIREN network to a 256-dimensional embedding, used as a covariate for
# downstream geospatial regression and classification. L = 10 gives a smoother,
# coarser encoder; L = 40 a finer one.
#
# ---------------------------------------------------------------------
# LOCATION ENCODER ONLY.
# ---------------------------------------------------------------------
# Upstream's documented loader, load.get_satclip, rebuilds the full Lightning
# module. Every released checkpoint has vision_layers = moco_resnet18,
# moco_resnet50 or moco_vit16, and for those the model constructor fetches
# torchgeo's SENTINEL2_ALL_MOCO backbone weights from the network, only to build
# an image encoder that embedding locations never uses. The image uses upstream's
# own load_lightweight.get_satclip_loc_encoder instead, which rebuilds only the
# location encoder from the checkpoint's hyperparameters and runs offline with
# torch, numpy and einops. The image encoder and training stack (Lightning,
# torchgeo, timm, albumentations, rasterio) are not installed.
#
# All six released checkpoints are baked (598 MB together, MIT, ungated). The
# paper names ViT16 as its default backbone, Table 2's caption names ResNet50,
# and Table 6 shows the six differ by task, so none is redundant.
#
# Upstream: https://github.com/microsoft/satclip @ 1ead106   MIT
# Weights:  https://huggingface.co/microsoft (SatCLIP-{ResNet18,ResNet50,ViT16}-{L10,L40})   MIT
# Paper:    https://arxiv.org/abs/2311.17179

FROM python:3.12-slim-bookworm

# enroot exposes the GPU only when the image asks for it. The location encoder
# is small enough to run on CPU; on a CPU partition override with
#   srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
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

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Upstream pins no torch. torch 2.14.0 is on the PyTorch index for CUDA 12.6
# and 13.0; 13.0 is the newest CUDA that Compute2's driver 580.105.08 supports,
# and PyTorch's v2.14.0 release build table compiles its 13.0 x86_64 wheels for
# sm_90 among others. constraints.txt is the full resolved set, so a rebuild
# installs the same versions and the resolver cannot swap in another torch build.
COPY constraints.txt /opt/constraints.txt
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install -c /opt/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        torch einops numpy pandas pyarrow

# Upstream is not a pip package: there is no setup.py or pyproject.toml, and
# its modules import each other by bare name (`from location_encoder import`,
# `import positional_encoding as PE`). So the source tree goes to /opt/satclip
# and its inner satclip/ directory onto PYTHONPATH, as upstream's notebooks do
# with sys.path.append('./satclip'). Importing it as a package instead would
# run satclip/__init__.py, which pulls in the uninstalled training stack.
# The tarball, not a clone, keeps git out of the image.
ARG SATCLIP_REF=1ead1067e9a00d44b132403b6bb2ce061e48d48e
RUN mkdir -p /opt/satclip \
    && curl -fsSL "https://github.com/microsoft/satclip/archive/${SATCLIP_REF}.tar.gz" \
        | tar -xz --strip-components=1 -C /opt/satclip \
    && python -m compileall -q /opt/satclip/satclip
# compileall: spherical_harmonics_ylm.py is 7.7 MB of generated code, and with
# PYTHONDONTWRITEBYTECODE every process would otherwise recompile it on import.
ENV PYTHONPATH=/opt/satclip/satclip \
    SATCLIP_WEIGHTS=/opt/satclip-weights

# Each checkpoint is fetched at a pinned Hugging Face revision and checked
# against the SHA-256 Hugging Face lists for that file.
RUN mkdir -p "$SATCLIP_WEIGHTS" && cd "$SATCLIP_WEIGHTS" \
    && fetch() { \
        curl -fsSL -o "satclip-$1.ckpt" \
          "https://huggingface.co/microsoft/SatCLIP-$2/resolve/$3/satclip-$1.ckpt" \
        && echo "$4  satclip-$1.ckpt" | sha256sum -c - ; } \
    && fetch vit16-l40    ViT16-L40    0ef2acc0b91d3c8cfdf4a0cc03207095989287ab c54c794cb8f95d4a141506656581240008570194b309c91217d86a6d4da5399f \
    && fetch vit16-l10    ViT16-L10    829a79064fade0233c68bf01b7a3155be1b8778b c7596a5f7aacf72a7c49587a2a5a80f23cf61000a36b46e471c1422b273ca9c0 \
    && fetch resnet50-l40 ResNet50-L40 b66714c7308a55d12849d1ea43f892440d78e41a 50df6453c20ce82b51ceba89811dcf1a21e7fa9d625cbc630f4cdcc4fc0de53a \
    && fetch resnet50-l10 ResNet50-L10 f41ed774863f2b0297e6f096eb3e3490888c03c2 87ad5e8ca152e9564896f48de033e8d4278665bce049d36b3794847a430e69b1 \
    && fetch resnet18-l40 ResNet18-L40 983a9d9d1f2f933f9a6a6225a214917fb58659d8 6e2ed15146566170829bae8dd506d8274cd6fe4534c0ef407668dc3ccb04d2f1 \
    && fetch resnet18-l10 ResNet18-L10 5263ff2455427581783c399808bdaa6c83857578 c4af9559afb43d9c8be2781d4f70ce3f8a1087ea5a3773906c7c50ed05479ebd \
    && du -sh "$SATCLIP_WEIGHTS"

COPY scripts/satclip-embed scripts/satclip-selftest scripts/satclip-verify-elevation /usr/local/bin/
RUN mkdir -p /work

# Build smoke test, offline: everything it reads is already in the image.
# torch imports on its own under faulthandler, so a compiled-extension crash
# names itself instead of failing silently inside a larger step.
RUN python -X faulthandler -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda); assert torch.version.cuda and torch.version.cuda.startswith('13.0'), 'CUDA 13.0 torch was replaced'"
# Every baked checkpoint loads and gives finite 256-d embeddings in which
# St. Louis is nearer to Kirkwood, MO than to the Sahara or the Amazon basin;
# package versions are printed from their installed metadata.
RUN python -X faulthandler /usr/local/bin/satclip-selftest
# The CLI end to end, including its refusal of an out-of-range row.
RUN cd /tmp \
    && printf 'name,lon,lat\nst_louis,-90.19777778,38.62722222\nsahara,13,23\n' > pts.csv \
    && satclip-embed pts.csv emb.parquet \
    && satclip-embed pts.csv emb.npy --checkpoint resnet50-l10 \
    && python -c "import numpy as np, pandas as pd; assert pd.read_parquet('emb.parquet').shape == (2, 259); assert np.load('emb.npy').shape == (2, 256)" \
    && printf 'lon,lat\n200,10\n' > bad.csv \
    && ! satclip-embed bad.csv bad.npy 2>/dev/null \
    && test ! -e bad.npy \
    && rm -f pts.csv emb.parquet emb.npy bad.csv \
    && echo 'SMOKE OK'

LABEL org.opencontainers.image.title="satclip" \
      org.opencontainers.image.description="SatCLIP — global location encoders pretrained contrastively on Sentinel-2 (S2-100K): (lon, lat) to a 256-d embedding for downstream geospatial regression and classification. Location encoder only; all six released checkpoints baked" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/satclip/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/microsoft/satclip" \
      bradleylab.model.paper="https://arxiv.org/abs/2311.17179" \
      bradleylab.model.task="location embedding: (lon, lat) to a 256-d SatCLIP embedding, used as a covariate for geospatial regression and classification" \
      bradleylab.model.sensor="location:lonlat (pretrained against Sentinel-2 L2A, 12 bands at 10 m)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.weights.source="baked at /opt/satclip-weights: huggingface.co/microsoft/SatCLIP-{ViT16,ResNet50,ResNet18}-{L40,L10}, each at a pinned revision and SHA-256-checked, 598 MB together" \
      bradleylab.model.licence.note="MIT for the code (microsoft/satclip) and for all six checkpoints (Hugging Face model cards). The model card scopes use to research"

WORKDIR /work
CMD ["bash", "-c", "echo 'satclip ready: satclip-embed <points.csv|.parquet> <out.parquet|.npy> [--checkpoint vit16-l40] (see README.md)'"]
