# galaxi — multiphase identification from powder X-ray diffraction.
#
# Tong, Jin, Xu, Rao, Jiang and Szymanski (2026), arXiv:2609.06908. GALAXI
# trains one independent binary classifier per phase instead of a single
# multi-class model, uses them to narrow a pattern down to a few plausible
# phases, then settles the combination by Rietveld refinement. Reported
# micro-F1 0.935 on experimental patterns, holding up under low impurity
# fractions, small crystallite size, peak shifts, sample displacement and
# texture.
#
# Because the per-phase models are independent, a new phase is added by
# training one more classifier -- the existing ones are untouched. That is
# the property this image exists for: building a catalog for the minerals
# this lab actually encounters, rather than relying on someone else's.
#
# Related image: `xrd-classifier` (autoXRD) does the same job by the older
# single-model route, and GALAXI's own paper benchmarks against that
# lineage. Both stay; this is not a drop-in replacement, and any comparison
# should be run on the same patterns before retiring anything.
#
# ---------------------------------------------------------------------
# WHAT THIS IMAGE HOLDS, AND WHAT HAS TO BE STAGED BESIDE IT.
# ---------------------------------------------------------------------
# The image is code only. Three separate assets live outside it, because
# each is large, each is fetched from a host that throttles, and none
# belongs in a layer:
#
#   COD reference structures   ~5.5 GB   galaxi-setup-cod
#   background/negative profiles ~670 MB galaxi-setup-bg-profiles
#   365-phase pretrained catalog ~618 MB figshare 10.6084/m9.figshare.33360183
#
# The first two are needed to GENERATE TRAINING DATA; neither is needed to
# evaluate a pattern against models you already have. The third is
# upstream's worked example (battery and oxide chemistry), useful for
# checking the install end to end and not much else for geology.
#
# GALAXI resolves both data paths from the environment before falling back
# to ~/.local/share/galaxi -- and that fallback is a trap here, because
# enroot bind-mounts $HOME, so a host directory would silently shadow
# anything staged. GALAXI_COD_DIR and GALAXI_BG_PROFILES are therefore set
# to mount points below, and the README says what to mount onto them.
#
# The COD download runs through gdown against Google Drive, which throttles
# non-interactive clients -- the same failure `kpconv` documents for its
# weights. Stage it once from a login node, not from a job.
#
# Upstream: https://github.com/Szymanski-Group/galaxi @ eb88287  (MIT)
# Weights:  https://doi.org/10.6084/m9.figshare.33360183  (MIT, 618 MB)
# Paper:    https://arxiv.org/abs/2609.06908
# Web:      https://galaxi-xrd.com serves a 64,594-structure library that is
#           NOT downloadable; only the 365-phase example ships as files.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the operator's ~/.local off sys.path; enroot
# bind-mounts $HOME and a host package shadowing an image one has broken
# images in this catalog before. MPLBACKEND because matplotlib is a hard
# dependency and there is no display on a compute node.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    GALAXI_COD_DIR=/data/cod \
    GALAXI_BG_PROFILES=/data/bg_profiles

# git for the pinned DARA fork below; build-essential because that fork is
# a source install rather than a wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG GALAXI_REF=eb88287901e555aef20be7977d7ddcfbd77de5d2
RUN git clone https://github.com/Szymanski-Group/galaxi.git /opt/galaxi \
    && cd /opt/galaxi \
    && git checkout "$GALAXI_REF" \
    && python -m pip install --upgrade pip wheel \
    && python -m pip install .

# The mount points the two staged datasets land on. Created empty so a run
# without them fails by finding nothing rather than by failing to resolve a
# path, which is the clearer error.
RUN mkdir -p /data/cod /data/bg_profiles

# Build smoke test — offline, no staged data, no weights. Exercises the
# crystallography stack on a CIF the repository already carries, which is
# the part most likely to break on a dependency bump, and confirms the two
# data paths resolve to the mount points rather than to $HOME.
RUN python -c "\
import importlib.metadata as im; \
import os, glob; \
import galaxi; \
from galaxi.paths import get_default_cod_dir, get_default_bg_profiles_path; \
from galaxi.workflows.streamlined_workflow import StreamlinedWorkflow; \
import torch, pymatgen.core as pmg; \
from pymatgen.analysis.diffraction.xrd import XRDCalculator; \
print('galaxi', im.version('galaxi'), '| torch', im.version('torch'), '| pymatgen', im.version('pymatgen'), '| pyxtal', im.version('pyxtal')); \
print('cuda build:', torch.version.cuda); \
assert torch.version.cuda is not None, 'CUDA torch was replaced by a CPU wheel'; \
cod = get_default_cod_dir(); bg = get_default_bg_profiles_path(); \
print('COD dir:', cod, '| bg profiles:', bg); \
assert cod.startswith('/data'), cod; \
assert bg.startswith('/data'), bg; \
cif = sorted(glob.glob('/opt/galaxi/tutorials/data/cif_files/*.cif'))[0]; \
s = pmg.Structure.from_file(cif); \
p = XRDCalculator().get_pattern(s); \
print('simulated', len(p.x), 'reflections from', os.path.basename(cif), '-> 2theta', round(float(p.x[0]), 2), 'to', round(float(p.x[-1]), 2)); \
assert len(p.x) > 5; \
assert callable(StreamlinedWorkflow); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="galaxi" \
      org.opencontainers.image.description="GALAXI — multiphase identification from powder XRD by one independent binary classifier per phase, plus Rietveld refinement. Trains new phase classifiers without retraining existing ones. Code only; COD structures, background profiles and the pretrained example catalog are staged separately" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/galaxi/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/Szymanski-Group/galaxi" \
      bradleylab.model.paper="https://arxiv.org/abs/2609.06908" \
      bradleylab.model.task="powder XRD multiphase identification; trains one binary detection model per phase" \
      bradleylab.model.sensor="powder X-ray diffraction patterns (2-theta, intensity)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="optional" \
      bradleylab.model.weights.source="NOT baked — 365-phase pretrained catalog at figshare 10.6084/m9.figshare.33360183 (MIT, 618 MB); the 64,594-structure library is web-only and not downloadable. Training needs COD structures (~5.5 GB) and background profiles (~670 MB) staged to GALAXI_COD_DIR and GALAXI_BG_PROFILES" \
      bradleylab.model.licence.note="MIT for both the code and the released pretrained weights. The large reference library behind galaxi-xrd.com is not distributed and using it means sending patterns to a third-party service"

WORKDIR /work
CMD ["python", "-c", "print('galaxi ready — stage COD and background profiles, then see README.md')"]
