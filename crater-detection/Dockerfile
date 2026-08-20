# crater-detection — lunar crater detection and crater-based navigation.
#
# Doppenberg (2021), MSc work at TU Delft: a Mask R-CNN variant that
# detects crater rims in lunar imagery, plus a matching stage that
# identifies detected craters against a catalog for absolute position
# fixing.
#
# WHY THIS AND NOT LunarFM. LunarFM is the stronger lunar model, and we
# have access to it — but it is PolyForm Strict 1.0.0, which forbids
# redistribution and derivative works, so it cannot be baked into a GHCR
# image at all. This is the only lunar model in the catalog that can be
# containerized normally. The two do different jobs regardless: LunarFM
# gives surface embeddings, this gives crater instances.
#
# ---------------------------------------------------------------------
# Two upstream traps, both worked around here.
# ---------------------------------------------------------------------
#
# 1. THE CHECKPOINTS ARE GIT LFS POINTERS. blobs/CraterRCNN.pth is 134
#    bytes in a plain clone — a text pointer, not weights. The real
#    object is 221 MB. A `git clone` without LFS bakes the pointer and
#    the image fails at load time with an unpickling error that names
#    nothing useful. This build fetches the LFS object over HTTPS
#    explicitly and asserts the size, so a pointer can never ship.
#
# 2. requirements.txt CANNOT INSTALL AS WRITTEN. It contains
#      git+git://github.com/SurRenderSoftware/surrender_client_API@master
#    and GitHub disabled the unauthenticated git:// protocol in 2022, so
#    that line fails regardless of whether the repo exists. `surrender`
#    is a rendering client used to synthesise training data, not to run
#    inference. Rather than guess, the dependencies below are installed
#    explicitly and the smoke test imports the detection path — if
#    surrender were genuinely required for inference, this build fails
#    rather than shipping a broken image.
#
# Upstream is unmaintained since 2025-01, so the commit is pinned.
#
# Upstream: https://github.com/wdoppenberg/crater-detection
# Licence:  MIT
# Stack:    pytorch 2.5.1 + CUDA 12.1, matching the other vision images

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CRATER_COMMIT=1365817fb62247378bb04085d41a2216a666ac25 \
    PYTHONPATH=/opt/crater-detection

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git curl \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Explicit, pinned, and deliberately not requirements.txt — see trap 2.
# surrender is omitted; the smoke test is what proves that is safe.
RUN pip install --upgrade pip wheel \
    && pip install \
        "numpy<2" \
        "scipy==1.14.1" \
        "pandas==2.2.3" \
        "h5py==3.12.1" \
        "scikit-learn==1.5.2" \
        "networkx==3.4.2" \
        "astropy==6.1.4" \
        "numba==0.60.0" \
        "spiceypy==6.0.0" \
        "tqdm>=4.66" \
        "matplotlib==3.9.2"

RUN git clone --depth 1 https://github.com/wdoppenberg/crater-detection.git /opt/crater-detection \
    && cd /opt/crater-detection \
    && git fetch --depth 1 origin "${CRATER_COMMIT}" \
    && git checkout "${CRATER_COMMIT}"

# Fetch the real LFS objects over HTTPS and refuse anything pointer-sized.
# 221297063 bytes is the size recorded in the LFS pointer for CraterRCNN.
RUN cd /opt/crater-detection/blobs \
    && for f in CraterRCNN.pth DeepMoon.pth; do \
         curl -fsSL -o "$f" "https://github.com/wdoppenberg/crater-detection/raw/${CRATER_COMMIT}/blobs/$f"; \
         sz=$(stat -c%s "$f"); \
         echo "$f: $sz bytes"; \
         if [ "$sz" -lt 1000000 ]; then echo "ERROR: $f is $sz bytes — an LFS pointer, not weights"; exit 1; fi; \
       done

# Build smoke test — CPU, offline. Loads the real checkpoint and asserts it
# is a populated state dict. This is also what proves the surrender
# omission is safe: if the detection path imported it, this would fail.
RUN python -c "\
import importlib.metadata as im; \
import torch; \
print('torch', im.version('torch'), '| numpy', im.version('numpy')); \
sd = torch.load('/opt/crater-detection/blobs/CraterRCNN.pth', map_location='cpu', weights_only=False); \
state = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd; \
keys = list(state.keys()) if hasattr(state, 'keys') else []; \
print('checkpoint entries', len(keys), '| first', keys[:2]); \
assert len(keys) > 50, len(keys); \
import src.detection as d; \
print('detection module imported:', d.__name__); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="crater-detection" \
      org.opencontainers.image.description="Lunar crater detection (Mask R-CNN) and crater-catalog matching for absolute position fixing; the only containerizable lunar model in the catalog" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/crater-detection/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/wdoppenberg/crater-detection" \
      bradleylab.model.task="lunar crater instance detection and catalog matching" \
      bradleylab.model.sensor="image:lunar orbital greyscale" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="git-lfs:wdoppenberg/crater-detection blobs/CraterRCNN.pth (221 MB, fetched over HTTPS at build; pointer-size guard in the build)" \
      bradleylab.model.licence.note="upstream unmaintained since 2025-01; commit pinned"

WORKDIR /work
CMD ["python", "-c", "print('crater-detection ready — see README.md')"]
