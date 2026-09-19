# tabfm — zero-shot classification and regression on tabular data.
#
# Google Research (2026). TabFM reads your training rows as *context* and
# predicts on new rows in a single forward pass: no per-dataset training,
# no hyperparameter search. It was pretrained on synthetic datasets
# generated from structural causal models, and it takes mixed numeric and
# categorical columns as they come.
#
# The API is scikit-learn shaped — TabFMClassifier / TabFMRegressor with
# fit and predict — so it drops into an existing tabular workflow as an
# alternative to a tuned gradient-boosted tree.
#
# ---------------------------------------------------------------------
# WEIGHTS ARE NOT BAKED, AND HERE THE LICENSE LEAVES NO CHOICE.
# ---------------------------------------------------------------------
# The TabFM Non-Commercial License v1.0 restriction 3(b) is flat: you
# will not "Distribute the TabFM Model or a Derivative." Publishing the
# checkpoints inside an image on a public registry is distribution, so
# baking them is not an option the way it is for an Apache-2.0 model.
# (They are also 13.1 GB across the two checkpoints, which would be
# reason enough on its own.)
#
# So this image carries the environment and no TabFM Model. The weights
# are fetched at run time from the ungated Hugging Face repository into a
# mounted cache — no token needed, unlike sam3.
#
# Consequences to know about:
#   - First use needs outbound network. On Compute2 that means staging the
#     cache from a login node, then running offline with HF_HUB_OFFLINE=1.
#     SMOKE.md shows both steps.
#   - The build smoke test cannot load weights, so it verifies the package
#     imports and both estimator classes resolve. Weight loading is proven
#     in SMOKE.md instead.
#
# ---------------------------------------------------------------------
# WHAT THE LICENSE RESTRICTS, BEYOND REDISTRIBUTION.
# ---------------------------------------------------------------------
# Non-commercial AND non-production. Academic research, internal
# benchmarking and experimentation are expressly permitted. Restriction
# 3(a) reaches further than most non-commercial terms: it covers the
# Outputs too, so predictions may not be used "in commercial decision-
# making, client deliverables, or paid products/services." Contract or
# consulting work is therefore out, not just selling the model.
#
# The full text travels at /opt/licenses/LICENSE.tabfm.md — anyone who
# fetches weights into this container is bound by it and should be able
# to read it without leaving the node.
#
# Upstream: https://github.com/google-research/tabfm          (Apache-2.0)
# Weights:  https://huggingface.co/google/tabfm-1.0.0-pytorch (ungated,
#           tabfm-non-commercial-v1.0, 13.1 GB: classification 6.56 GB +
#           regression 6.59 GB)
# Stack:    pytorch 2.5.1 + CUDA 12.1, matching the other H100 images.
#           tabfm needs Python >= 3.11; this base ships 3.11.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# enroot exposes the GPU only when the image asks for it; without this a
# CUDA torch reports cuda_available False on an H100 node. On a CPU
# partition pass  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the operator's ~/.local off sys.path: enroot
# bind-mounts $HOME, and a host package shadowing an image one is a class
# of bug this catalog has already paid for once.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TABFM_MODEL_ID=google/tabfm-1.0.0-pytorch

# The constraint pins torch to the base image's CUDA build. tabfm's
# [pytorch] extra requires a bare `torch`, and an unconstrained resolve is
# how a CPU wheel silently replaces a CUDA one.
#
# safetensors is explicit because nothing else here pulls it in. The
# checkpoints are .safetensors and huggingface_hub imports the reader
# lazily, inside the weight-loading call, so its absence is invisible
# until the first real load — the build smoke test cannot reach it, and
# did not. Found on the H100 (job 3109333): ModuleNotFoundError at
# hub_mixin._load_as_safetensor. transformers would have supplied it, and
# this image has no transformers.
RUN pip install --upgrade pip wheel \
    && printf 'torch==2.5.1\n' > /tmp/constraints.txt \
    && pip install -c /tmp/constraints.txt \
        "tabfm[pytorch]==1.0.1" \
        "safetensors>=0.4" \
    && rm /tmp/constraints.txt

COPY LICENSE.tabfm.md /opt/licenses/LICENSE.tabfm.md

# Build smoke test — offline, no weights. HF_HUB_OFFLINE makes an
# accidental fetch fail loudly rather than pull 13 GB into a layer and
# distribute checkpoints the license forbids distributing.
RUN HF_HUB_OFFLINE=1 python -c "\
import importlib.metadata as im; \
import torch; \
from tabfm import TabFMClassifier, TabFMRegressor; \
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0; \
import safetensors; \
print('tabfm', im.version('tabfm'), '| torch', im.version('torch'), '| scikit-learn', im.version('scikit-learn'), '| safetensors', im.version('safetensors')); \
print('cuda build:', torch.version.cuda); \
assert torch.version.cuda is not None, 'CUDA torch was replaced by a CPU wheel'; \
assert tuple(int(p) for p in torch.version.cuda.split('.')[:2]) >= (11, 8), f'sm_90 needs CUDA >= 11.8, got cu{torch.version.cuda}'; \
assert callable(TabFMClassifier) and callable(TabFMRegressor); \
assert hasattr(tabfm_v1_0_0, 'load'), dir(tabfm_v1_0_0); \
print('estimators resolved:', TabFMClassifier.__name__, TabFMRegressor.__name__); \
print('weights are NOT baked — fetched at runtime, see README.md'); \
print('SMOKE OK')"

# Guard: nothing large may have landed in the image cache. A build that
# quietly downloaded a checkpoint would both bloat the image and publish
# weights the license forbids publishing.
RUN test ! -d /opt/hf-cache || [ "$(du -sm /opt/hf-cache | cut -f1)" -lt 100 ] \
    || (echo 'FAIL: weights present in the image; they must not be baked' && exit 1)

LABEL org.opencontainers.image.title="tabfm" \
      org.opencontainers.image.description="TabFM — Google Research zero-shot tabular foundation model; in-context classification and regression on mixed-type tables through a scikit-learn API. Weights are NOT baked; fetched at runtime because the license forbids redistribution" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/tabfm/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="LicenseRef-TabFM-NonCommercial" \
      bradleylab.model.upstream="https://github.com/google-research/tabfm" \
      bradleylab.model.task="zero-shot tabular classification and regression by in-context learning" \
      bradleylab.model.sensor="tabular — mixed numeric and categorical columns" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="optional" \
      bradleylab.model.weights.source="huggingface-hub:google/tabfm-1.0.0-pytorch — ungated but NOT baked (license forbids redistribution); runtime fetch, 13.1 GB across classification/ and regression/" \
      bradleylab.model.licence.note="TabFM Non-Commercial License v1.0 at /opt/licenses/LICENSE.tabfm.md — non-commercial AND non-production; restriction 3(a) reaches the Outputs, so predictions may not feed client deliverables or paid services; 3(b) forbids redistributing weights or derivatives"

WORKDIR /work
CMD ["python", "-c", "print('tabfm ready — weights fetch at runtime, see README.md')"]
