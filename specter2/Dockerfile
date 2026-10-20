# syntax=docker/dockerfile:1
# specter2 — scientific-document embeddings (AllenAI SPECTER2).
#
# A SciBERT-initialized encoder trained on over 6 M citation triplets, plus one
# adapter per task format trained on the SciRepEval tasks: proximity (paper to
# paper retrieval and nearest neighbors), adhoc_query (short search queries),
# classification and regression (features for linear models). A paper is
# embedded from its title and abstract as the [CLS] vector, 768 wide.
#
# Base model and all four adapters are Apache-2.0 and are baked at pinned
# Hugging Face revisions, SHA-256-checked at build; the image runs offline.
#
# Upstream: https://github.com/allenai/SPECTER2 (Apache-2.0)
# Paper:    Singh, D'Arcy, Cohan, Downey and Feldman, SciRepEval, EMNLP 2023

# transformers 4.50 and later refuse to torch.load a .bin checkpoint under
# torch < 2.6, and SPECTER2 ships only .bin files. chai-1 already uses this base.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/opt/specter2/lib

# The model card loads the adapters with the adapters library; 1.2.0 requires
# transformers ~=4.51.3. The rest are the versions this stack resolved to when
# it was tested, with numpy kept on 1.x.
RUN pip install \
        "adapters==1.2.0" \
        "transformers==4.51.3" \
        "tokenizers==0.21.4" \
        "huggingface_hub==0.36.2" \
        "numpy==1.26.4"

COPY scripts/specter2_weights.py /opt/specter2/lib/
COPY scripts/specter2-embed scripts/specter2-selftest scripts/specter2-fetch-weights /usr/local/bin/

RUN specter2-fetch-weights && rm -rf /root/.cache/huggingface
ENV HF_HUB_OFFLINE=1

RUN --network=none python -c "import importlib.metadata as im, torch; import adapters, transformers; \
from adapters import AutoAdapterModel; \
print('torch', torch.__version__, '| cuda-build', torch.version.cuda, '| adapters', im.version('adapters'), \
      '| transformers', im.version('transformers'), '| tokenizers', im.version('tokenizers'), \
      '| huggingface_hub', im.version('huggingface_hub'), '| numpy', im.version('numpy'))"

# Loads the baked base model with every adapter and embeds the model card's two
# example papers and one query on the CPU, network disabled.
RUN --network=none NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES="" specter2-selftest

WORKDIR /work

LABEL org.opencontainers.image.title="specter2" \
      org.opencontainers.image.description="SPECTER2 (AllenAI) scientific-document embeddings from title and abstract, 768-d, with the proximity, adhoc_query, classification and regression adapters baked" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/specter2/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/allenai/SPECTER2" \
      bradleylab.model.paper="https://api.semanticscholar.org/CorpusID:254018137" \
      bradleylab.model.task="scientific-document and query embedding (768-d) for retrieval, nearest-neighbor search, classification and regression features" \
      bradleylab.model.sensor="text:english scientific titles and abstracts" \
      bradleylab.model.status="experimental" \
      bradleylab.model.domain="scientific literature" \
      bradleylab.model.weights.source="baked: huggingface:allenai/specter2_base @ 3447645 and adapters allenai/specter2 @ 2081559, specter2_adhoc_query @ 3f44488, specter2_classification @ d843816, specter2_regression @ a9d105d; SHA-256-checked at build" \
      bradleylab.model.weights.license="Apache-2.0 (model cards)"
