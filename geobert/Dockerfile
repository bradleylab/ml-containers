# syntax=docker/dockerfile:1
# geobert — named-entity recognition for geoscience text (botryan96/GeoBERT).
#
# A BERT token classifier, fine-tuned from SciBERT according to its model card,
# that tags four entity types: GeoPetro (geoscience terms), GeoMeth (tools and
# methods), GeoLoc (geological locations) and GeoTime (geological time).
#
# The Hugging Face repository states no license, and there is no paper or code
# repository behind it, so nothing from it is in this image. geobert-ner
# downloads the weights on first use from a pinned revision into a cache the
# caller names, and checks the TensorFlow checkpoint against its published
# SHA-256 before loading it. Not to be confused with the GeoBERT of Liu et al.
# (2022, Earth Sci. Inform.) or of Denli et al. (2021, ADIPEC; trained on the
# authors' internal records).
#
# Weights: https://huggingface.co/botryan96/GeoBERT @ 308126d (tf_model.h5 only)

FROM python:3.11-slim

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TF_CPP_MIN_LOG_LEVEL=1

# The checkpoint is TensorFlow-only. transformers 5 removed TensorFlow support,
# and 4.44 is a 4.x release that accepts TensorFlow 2.15, the last release on
# Keras 2. The other pins are the versions this stack resolved to when it was
# tested.
RUN pip install \
        "tensorflow[and-cuda]==2.15.1" \
        "transformers==4.44.2" \
        "tokenizers==0.19.1" \
        "huggingface_hub==0.25.2" \
        "numpy==1.26.4"

COPY scripts/geobert-ner scripts/geobert-selftest /usr/local/bin/

RUN --network=none python -c "import importlib.metadata as im; import tensorflow as tf, transformers; \
from transformers import TFAutoModelForTokenClassification, pipeline; \
print('tensorflow', tf.__version__, '| keras', im.version('keras'), '| transformers', im.version('transformers'), \
      '| tokenizers', im.version('tokenizers'), '| huggingface_hub', im.version('huggingface_hub'), '| numpy', im.version('numpy'))"

# End to end on a tiny random model with GeoBERT's label set; the real weights
# are fetched and hash-checked only at run time.
RUN --network=none NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES="" geobert-selftest

WORKDIR /work

LABEL org.opencontainers.image.title="geobert" \
      org.opencontainers.image.description="GeoBERT (botryan96) named-entity recognition for geoscience text: GeoPetro, GeoMeth, GeoLoc and GeoTime spans. Weights not baked: downloaded at a pinned revision and SHA-256-checked at run time" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/geobert/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="NOASSERTION" \
      bradleylab.model.upstream="https://huggingface.co/botryan96/GeoBERT" \
      bradleylab.model.task="named-entity recognition in geoscience text (GeoPetro, GeoMeth, GeoLoc, GeoTime)" \
      bradleylab.model.sensor="text:english" \
      bradleylab.model.status="experimental" \
      bradleylab.model.domain="geoscience text" \
      bradleylab.model.use_restriction="WashU-internal research use only. The model repository states no license; the weights are not redistributed and are downloaded by the user at run time" \
      bradleylab.model.weights.source="NOT baked: huggingface:botryan96/GeoBERT @ 308126d86d5219921c6956bc63f819f6d405fedf, tf_model.h5 SHA-256 67c1b5687d3b6a39a82de7287349dac1ac36849a75d7921581f1935c7cf227ce, fetched by geobert-ner into --cache-dir" \
      bradleylab.model.weights.license="none stated on the Hugging Face repository" \
      bradleylab.model.licence.note="No license on the model repository or on any of the author's Hugging Face repositories; no paper or code repository. The image contains only the lab's wrapper and public libraries"
