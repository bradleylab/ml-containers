# clean — CLEAN enzyme EC-number prediction (Yu et al., Science 2023)
#
# Contrastive learning over ESM-1b embeddings that assigns EC numbers to
# amino-acid sequences. Two-stage inference: ESM-1b produces a mean-pooled
# per-sequence embedding, then CLEAN's LayerNormNet projects it and assigns
# EC numbers by max-separation against pre-computed EC cluster centres.
#
# Upstream:  https://github.com/tttianhao/CLEAN
# Paper:     Yu et al. (2023), Science 379:1358 — doi:10.1126/science.adf2465
# Stack:     python:3.10-slim + CPU-only torch + fair-esm + CLEAN
#
# LICENSE — NOT MIT, research use only. GitHub's repo metadata advertises
# MIT, but the tree contains no LICENSE file; the only licence artifact is
# "NON-EXCLUSIVE RESEARCH USE LICENSE FOR CLEAN SOFTWARE.pdf". Terms for the
# pretrained weights are not stated anywhere. This image is built and used
# under the research-use reading (university research). Any use beyond that
# needs the authors' written agreement. See README.md.
#
# CPU-ONLY BY DESIGN. Upstream's own Dockerfile installs CPU torch, the
# authors document CPU inference as supported, and the documented floor is
# >12 GB *system RAM*, not VRAM. Do not build a CUDA variant of this image
# and do not request a GPU for it — CLEAN's code will happily use CUDA if it
# finds it, which is how an H100 ends up idling on a job that does not need
# one.

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Reproducibility pins. Upstream ships no PyPI release and its Dockerfile
# floats both torch and the ESM checkout, so an unpinned rebuild is not
# reproducible — and, today, not even functional (see TORCH_VERSION below).
# Change a pin by editing these defaults and committing, so the pin lives in
# git history; the CI workflow deliberately passes no build-args.
ARG CLEAN_GIT_SHA=f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15
ARG ESM_GIT_SHA=839c5b82c6cd9e18baa7a88dcbed3bd4b6d48e47
# torch 2.6.0 flipped the `torch.load` default to weights_only=True, which
# refuses the ESM-1b checkpoint (it pickles an argparse.Namespace). 2.5.1 is
# the last release below that flip. The manuscript used 1.11.0; pass
# --build-arg TORCH_VERSION=1.11.0 for a bit-faithful rebuild. The `+cpu`
# local version is appended at install time — it exists only on PyTorch's CPU
# index, so pip cannot silently fall back to PyPI's CUDA wheel.
ARG TORCH_VERSION=2.5.1
# app/requirements.txt pins fair-esm==2.0.0, but the README states the
# manuscript results came from 1.0.2 and warns that predictions depend on the
# ESM-1b version. 1.0.2 is chosen here; ESM_GIT_SHA is the matching v1.0.2
# tag so the cloned extract.py and the installed package are the same
# release. See README.md for the tradeoff.
ARG FAIR_ESM_VERSION=1.0.2
ENV CLEAN_GIT_SHA=${CLEAN_GIT_SHA} \
    ESM_GIT_SHA=${ESM_GIT_SHA}

# Clone at the pinned commit and record the SHA before dropping .git, so the
# smoke test can prove the image contains the code it claims to.
RUN git clone https://github.com/tttianhao/CLEAN.git /opt/CLEAN \
    && cd /opt/CLEAN \
    && git checkout ${CLEAN_GIT_SHA} \
    && git rev-parse HEAD > /opt/CLEAN/GIT_SHA \
    && rm -rf .git

WORKDIR /opt/CLEAN/app

# Upstream's requirements first (old, exact pins on numpy/scipy/pandas that
# the rest of the stack is built against), then CPU torch from PyTorch's own
# index, then the fair-esm override. --no-deps on the override keeps the
# resolved tree intact; fair-esm declares no runtime requirements of its own.
RUN pip install --upgrade pip "setuptools<85" wheel \
    && pip install -r requirements.txt \
    && pip install "torch==${TORCH_VERSION}+cpu" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --force-reinstall --no-deps "fair-esm==${FAIR_ESM_VERSION}"

# CLEAN shells out to ./esm/scripts/extract.py by a path relative to the
# working directory (CLEAN/utils.py: retrive_esm1b_embedding), so this clone
# is load-bearing, not a duplicate of the pip package. Pinned to the tag
# matching FAIR_ESM_VERSION so script and package agree.
RUN git clone https://github.com/facebookresearch/esm.git esm \
    && cd esm \
    && git checkout ${ESM_GIT_SHA} \
    && git rev-parse HEAD > /opt/CLEAN/ESM_GIT_SHA \
    && rm -rf .git

# Weights are NOT baked. Two separate downloads are needed at runtime and
# both are mount points:
#   data/pretrained         CLEAN's own weights (~141 MB, Google Drive only)
#   /root/.cache/torch      ESM-1b via torch.hub (~7.3 GB, dl.fbaipublicfiles.com)
# data/esm_data holds the per-sequence embeddings written during inference;
# mount it too if you want that cache to survive the job.
RUN mkdir -p data/pretrained data/esm_data results/inputs /root/.cache/torch/hub/checkpoints
ENV TORCH_HOME=/root/.cache/torch

# Upstream installs its package with the legacy `python build.py install`
# (build.py is a plain setuptools script, not a setup.py). That path still
# works — verified on setuptools 84 — but it is deprecated, hence the
# setuptools ceiling above: an unpinned setuptools that finally removes the
# `install` command would break this layer silently on the next rebuild.
RUN python build.py install

# Build-time smoke test. Offline by design — no weight fetch, so it proves
# the install rather than the model. Versions come from package metadata per
# the ml-containers rule, never `pkg.__version__`.
#
# Note it does NOT `import esm`: the cloned esm/ directory shadows the
# fair-esm package as a namespace package whenever the working directory is
# on sys.path, which is exactly the situation here. Reading the version from
# metadata sidesteps the shadowing entirely.
RUN python -c "import importlib.metadata as im, os, torch; \
from CLEAN.model import LayerNormNet; \
print('CLEAN', im.version('CLEAN'), '| fair-esm', im.version('fair-esm'), \
      '| torch', torch.__version__, '| numpy', im.version('numpy'), \
      '| scikit-learn', im.version('scikit-learn')); \
assert torch.version.cuda is None, ('expected a CPU-only torch build', torch.version.cuda); \
assert tuple(int(p) for p in torch.__version__.split('+')[0].split('.')[:2]) < (2, 6), \
    ('torch >= 2.6 defaults torch.load to weights_only=True and cannot load ESM-1b', torch.__version__); \
assert os.path.isfile('esm/scripts/extract.py'), 'esm/scripts/extract.py missing — CLEAN calls it by relative path'; \
assert open('/opt/CLEAN/GIT_SHA').read().strip() == os.environ['CLEAN_GIT_SHA']; \
assert open('/opt/CLEAN/ESM_GIT_SHA').read().strip() == os.environ['ESM_GIT_SHA']; \
print('LayerNormNet', LayerNormNet)"

CMD ["bash"]

LABEL org.opencontainers.image.title="clean" \
      org.opencontainers.image.description="CLEAN (Yu et al., Science 2023) — enzyme EC-number prediction from amino-acid sequence via contrastive learning over ESM-1b embeddings; CPU-only, >12 GB RAM; weights mounted at runtime, not baked. Research use only." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/clean/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="LicenseRef-CLEAN-Non-Exclusive-Research-Use" \
      bradleylab.model.upstream="https://github.com/tttianhao/CLEAN" \
      bradleylab.model.paper="https://doi.org/10.1126/science.adf2465" \
      bradleylab.model.task="enzyme commission (EC) number prediction from protein sequence" \
      bradleylab.model.sensor="protein-sequence" \
      bradleylab.model.status="experimental" \
      bradleylab.model.use_restriction="research-only" \
      bradleylab.model.weights.source="NOT baked — CLEAN weights (~141 MB) from Google Drive mounted at /opt/CLEAN/app/data/pretrained; ESM-1b (~7.3 GB) from dl.fbaipublicfiles.com cached under TORCH_HOME" \
      bradleylab.upstream.git_sha="${CLEAN_GIT_SHA}" \
      bradleylab.build.esm_git_sha="${ESM_GIT_SHA}" \
      bradleylab.build.fair_esm_version="${FAIR_ESM_VERSION}" \
      bradleylab.build.torch_version="${TORCH_VERSION}"
