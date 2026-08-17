# boltz — Boltz-2 biomolecular structure + binding-affinity prediction (MIT)
#
# Co-folds protein / RNA / DNA / small-molecule complexes and, from the same
# input file, predicts binding affinity for a nominated ligand chain.
# Successor to Boltz-1.
#
# Upstream:  https://github.com/jwohlwend/boltz  (tag v2.2.1, 2025-09-08)
# Weights:   https://huggingface.co/boltz-community  (MIT, ungated)
# Licence:   MIT for code AND weights — academic and commercial use permitted.
# Stack:     pytorch/pytorch 2.8.0 (torch 2.8.0, CUDA 12.9, Python 3.11)
#            + boltz[cuda]==2.2.1 + a version-pinned cuEquivariance stack.
#
# Base-image choice. Two of boltz's own pins decide it. `numpy>=1.26,<2.0`
# argues against an NGC PyTorch base: NGC images ship a large NVIDIA-built
# stack behind a global pip constraint file (`PIP_CONSTRAINT=/etc/pip/
# constraint.txt`, visible in the 25.04 image config), and forcing a numpy
# downgrade underneath that stack is how those images break. `requires-python
# >=3.10,<3.13` rules out anything on 3.13. A plain pytorch/pytorch tag has
# no such stack to disturb and lets pip resolve the tree in one pass; 2.8.0
# ships Python 3.11, satisfies boltz's `torch>=2.2`, is contemporaneous with
# both boltz v2.2.1 and the cuEquivariance wheels pinned below, and its CUDA
# 12.9 build covers Hopper (sm_90).

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ARG BOLTZ_VERSION=2.2.1
# cuEquivariance is pinned, not merely bounded, for a specific reason. boltz's
# `[cuda]` extra asks only for `>=0.5.0`, and cuequivariance-ops-torch-cu12
# 0.11.1 (2026-08-07) promoted `torch>=2.11` from a test extra to a hard
# runtime dependency — so an unpinned build today silently replaces the base
# image's torch and its entire CUDA stack with PyPI wheels. 0.6.1 (2025-09-04)
# is the release that was current four days before boltz v2.2.1 shipped, i.e.
# what `pip install boltz[cuda]==2.2.1` resolved to on release day.
ARG CUEQ_VERSION=0.6.1
# The torch the base image ships. Asserted by the smoke test so that a
# dependency which quietly pulls a different torch fails the build.
ARG BASE_TORCH_VERSION=2.8.0
ENV BOLTZ_VERSION=${BOLTZ_VERSION} \
    CUEQ_VERSION=${CUEQ_VERSION} \
    BASE_TORCH_VERSION=${BASE_TORCH_VERSION}

# One resolve, so the cuEquivariance pins are visible to pip while it is
# solving boltz's `[cuda]` extra rather than being applied afterwards.
RUN pip install --no-cache-dir \
        "boltz[cuda]==${BOLTZ_VERSION}" \
        "cuequivariance-torch==${CUEQ_VERSION}" \
        "cuequivariance-ops-torch-cu12==${CUEQ_VERSION}" \
        "cuequivariance-ops-cu12==${CUEQ_VERSION}"

# Weights are NOT baked — ~6.2 GB is fetched on first run (boltz2_conf.ckpt,
# boltz2_aff.ckpt, mols.tar). BOLTZ_CACHE must be an ABSOLUTE path; boltz
# raises rather than falling back if it is relative. The default here is a
# real directory inside the image so an unmounted run still works, but on
# Compute2 bind-mount scratch over it so the fetch happens once.
ENV BOLTZ_CACHE=/opt/boltz-cache
RUN mkdir -p /opt/boltz-cache

# Build-time smoke test. Offline: boltz reaches cuEquivariance only from
# inside `kernel_triangular_mult()`, not at module scope, so the Boltz2 model
# class imports cleanly on a CPU runner with no GPU and no network. Versions
# come from package metadata per the ml-containers rule, never
# `pkg.__version__`. The asserts pin down the three things that rot silently:
# the cuEquivariance version, the numpy major, and the torch build.
RUN python -c "import importlib.metadata as im, os, torch; \
from boltz.model.models.boltz2 import Boltz2; \
scripts = {e.name for e in im.entry_points(group='console_scripts')}; \
print('boltz', im.version('boltz'), '| cuequivariance-torch', im.version('cuequivariance-torch'), \
'| cuequivariance-ops-torch-cu12', im.version('cuequivariance-ops-torch-cu12'), \
'| numpy', im.version('numpy'), '| rdkit', im.version('rdkit'), \
'| torch', torch.__version__, '| model class', Boltz2.__name__); \
assert im.version('boltz') == os.environ['BOLTZ_VERSION'], im.version('boltz'); \
assert im.version('cuequivariance-ops-torch-cu12') == os.environ['CUEQ_VERSION'], im.version('cuequivariance-ops-torch-cu12'); \
assert im.version('numpy').startswith('1.'), im.version('numpy'); \
assert torch.__version__.startswith(os.environ['BASE_TORCH_VERSION']), torch.__version__; \
assert 'boltz' in scripts, sorted(scripts)"

WORKDIR /workspace

LABEL org.opencontainers.image.title="boltz" \
      org.opencontainers.image.description="Boltz-2 biomolecular structure and binding-affinity prediction — protein/RNA/DNA/ligand co-folding; pytorch 2.8.0 + boltz[cuda] 2.2.1 with pinned cuEquivariance kernels; H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/boltz/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/jwohlwend/boltz" \
      bradleylab.model.task="biomolecular structure prediction + binding affinity" \
      bradleylab.model.sensor="biomolecular-sequence" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:boltz-community + model-gateway.boltz.bio (runtime fetch; MIT, ungated)" \
      bradleylab.build.boltz_version="${BOLTZ_VERSION}" \
      bradleylab.build.cuequivariance_version="${CUEQ_VERSION}"
