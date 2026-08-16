# chai-1 — Chai-1 biomolecular structure prediction (Apache-2.0)
#
# Co-folds protein / RNA / DNA / small-molecule complexes from sequence, with
# no MSA required by default. Chai Discovery's open model.
#
# Upstream:  https://github.com/chaidiscovery/chai-lab  (release v0.6.1,
#            2025-03-18)
# Weights:   https://chaiassets.com  (partial mirror at
#            https://huggingface.co/chaidiscovery/chai-1 — see README)
# Licence:   Apache-2.0 for code AND weights. Chai Discovery relicensed in
#            Nov 2024 from the original Sept-2024 research-only terms, so
#            academic and commercial use are both permitted.
# Stack:     pytorch/pytorch 2.6.0 (torch 2.6.0, CUDA 12.6, Python 3.11)
#            + chai_lab==0.6.1.
#
# Base-image choice — this image cannot share evo2/esm's NGC base. chai_lab
# v0.6.1 pins `torch>=2.3.1,<2.7`, and NGC PyTorch 25.04 ships torch
# 2.7.0a0+79aa174. Whether that pre-release satisfies `<2.7` depends on pip's
# pre-release handling rather than on anything stable, which is not a thing to
# rest a build on. chai_lab also pins `numpy~=1.21` (i.e. <2.0), which an NGC
# base resists for the same reason boltz does — NVIDIA's stack sits behind a
# global pip constraint file. pytorch/pytorch 2.6.0 is the highest release
# under chai_lab's torch ceiling, ships Python 3.11 (>=3.10 required), and its
# CUDA 12.6 build covers Hopper (sm_90) and the bfloat16 support Chai-1 needs.

FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Pinned to the tag, not `main`. Upstream itself recommends pinning, and
# `main` has drifted off the v0.6.1 dependency pins (torch<2.7, rdkit
# ~=2024.9.5) that this base image was chosen to satisfy.
ARG CHAI_VERSION=0.6.1
ENV CHAI_VERSION=${CHAI_VERSION}

RUN pip install --no-cache-dir "chai_lab==${CHAI_VERSION}"

# Weights are NOT baked — ~7.0 GB across 8 assets is fetched on first run.
# Without CHAI_DOWNLOADS_DIR the destination is `<site-packages>/downloads`,
# which on HPC is read-only or purge-prone; set it so it can be bind-mounted.
# It is read at import time, so it must be in the environment before the
# process starts, not set from inside Python.
ENV CHAI_DOWNLOADS_DIR=/opt/chai-downloads
RUN mkdir -p /opt/chai-downloads

# Build-time smoke test. Offline: no chai_lab module touches CUDA or the
# network at import scope, so the inference entrypoint imports cleanly on a
# CPU runner. Versions come from package metadata per the ml-containers rule —
# note that chai_lab *does* expose `__version__`, and we deliberately do not
# use it. The torch bound is asserted numerically because a dependency that
# drags in a torch outside `>=2.3.1,<2.7` is the failure this image's base
# choice exists to prevent.
RUN python -c "import importlib.metadata as im, os, torch; \
from chai_lab.chai1 import run_inference; \
scripts = {e.name for e in im.entry_points(group='console_scripts')}; \
print('chai_lab', im.version('chai_lab'), '| torch', torch.__version__, \
'| numpy', im.version('numpy'), '| rdkit', im.version('rdkit'), \
'| entrypoint', run_inference.__name__); \
assert im.version('chai_lab') == os.environ['CHAI_VERSION'], im.version('chai_lab'); \
assert im.version('numpy').startswith('1.'), im.version('numpy'); \
mm = tuple(int(p) for p in torch.__version__.split('+')[0].split('.')[:2]); \
assert (2, 3) <= mm < (2, 7), torch.__version__; \
assert 'chai-lab' in scripts, sorted(scripts)"

WORKDIR /workspace

LABEL org.opencontainers.image.title="chai-1" \
      org.opencontainers.image.description="Chai-1 biomolecular structure prediction — protein/RNA/DNA/ligand co-folding, MSA-free by default; pytorch 2.6.0 + chai_lab 0.6.1; A100/H100 80GB" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/chai-1/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/chaidiscovery/chai-lab" \
      bradleylab.model.paper="https://doi.org/10.1101/2024.10.10.615955" \
      bradleylab.model.task="biomolecular structure prediction, MSA-free co-folding" \
      bradleylab.model.sensor="biomolecular-sequence" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="chaiassets.com (runtime fetch; Apache-2.0, ungated; HF mirror is partial)" \
      bradleylab.build.chai_version="${CHAI_VERSION}"
