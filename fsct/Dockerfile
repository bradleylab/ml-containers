# FSCT — Forest Structural Complexity Tool (Krisanski et al. 2021)
# https://github.com/SKrisanski/FSCT
#
# CPU-only build. FSCT hard-pins torch==1.9.0+cu111 + matching PyG
# extension wheels. These do not run on H100 (SM 9.0); porting to torch
# 2.x + CUDA 12 is a multi-day exercise we already did for SegmentAnyTree
# and do not need to repeat — FSCT inference is CPU-friendly, and a
# ~60 M-point TLS file finishes in tens of minutes on a modern CPU.
#
# Run on Compute2 general-cpu partition, no GPU allocation.
# (The FSCT source supports GPU; use_CPU_only=True is hardcoded here.)

# bookworm, not bullseye: Debian 11 LTS ended 2026-08-31 and its security
# archive is being thinned -- three builds in a row died on 404s for bullseye
# packages the index still listed. FSCT's own pins are Python and torch
# 1.9.0+cu111 (manylinux2014 wheels), neither of which cares which Debian
# release supplies libc.
FROM python:3.9-slim-bookworm

# PYTHONNOUSERSITE=1: enroot bind-mounts $HOME on Compute2, the login node's
# Python is 3.9, and so is this image's -- so ~/.local/lib/python3.9/site-packages
# would otherwise shadow the container's packages (found via a foreign pyproj in
# the adaf image, 2026-09-04).
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONNOUSERSITE=1

# System deps:
#   - build-essential: hdbscan / python-louvain compile C extensions
#   - libgl1 / libglib2.0-0: matplotlib / scikit-image runtime
#   - python3-tk: FSCT's run_tools.py imports tkinter at module top; we
#     don't use the GUI but the import must succeed
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 python3-tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/SKrisanski/FSCT.git

WORKDIR /opt/FSCT

# ---- Pinned deps from FSCT's requirements.txt, CPU-only torch ----
#
# `--extra-index-url`, NOT `--index-url`. This is load-bearing and was the
# cause of a 2026-08 build failure that appeared with no diff to this recipe.
#
# `--index-url` REPLACES PyPI rather than adding to it, so the PyTorch index
# becomes the only source for everything — including transitive deps and,
# critically, the *build* dependencies pip needs if it ever falls back to an
# sdist. In August 2026 pip resolved `typing-extensions` (a torch dependency,
# unpinned upstream) to a source distribution, then tried to fetch `flit_core`
# to build it, found no PyPI to fetch it from, and failed with
# "pip subprocess to install build dependencies did not run successfully".
# The torch wheels themselves were fine and still are — the rot was in an
# unpinned transitive dep with no fallback index.
#
# With `--extra-index-url` the `+cpu` local version still forces torch and
# torchvision to come from the PyTorch index (those builds exist nowhere else),
# while everything else resolves from PyPI as normal.
#
# typing-extensions is additionally pinned so this cannot recur: it is a typing
# shim with no effect on what the model computes.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==1.9.0+cpu torchvision==0.10.0+cpu \
        typing-extensions==4.12.2

RUN pip install \
        -f https://data.pyg.org/whl/torch-1.9.0+cpu.html \
        torch-scatter==2.0.8 \
        torch-sparse==0.6.11 \
        torch-cluster==1.5.9 \
        torch-spline-conv==1.2.1 \
        torch-geometric==1.7.2

# numpy first so hdbscan's wheel selection has a compatible ABI.
RUN pip install numpy==1.22.3

# Rest of FSCT's deps. hdbscan is left unpinned so pip picks the newest
# cp39-compatible binary wheel (FSCT's hdbscan usage is stable across
# versions). sklearn==0.0 meta-package from upstream requirements is
# dropped (it errors on modern pip).
RUN pip install \
        "laspy[lazrs]==2.1.2" \
        pandas==1.4.2 \
        scikit-learn==1.0.2 \
        scipy==1.8.0 \
        hdbscan \
        scikit-image==0.19.2 \
        scikit-spatial \
        networkx==2.8 \
        matplotlib==3.5.2 \
        mdutils \
        markdown

# Force CPU mode as the default in other_parameters.py — the CLI flag
# --gpu can override, but the build is CPU-only so leave this on.
RUN sed -i "s/'use_CPU_only': False/'use_CPU_only': True/" \
    scripts/other_parameters.py || true

# Install our headless CLI entrypoint alongside upstream scripts
COPY run_cli.py /opt/FSCT/scripts/run_cli.py
RUN chmod +x /opt/FSCT/scripts/run_cli.py

# Verify the install
RUN python -c "import torch; print('torch:', torch.__version__); \
    import torch_scatter, torch_cluster, torch_geometric, laspy; \
    print('pyg-scatter:', torch_scatter.__version__); \
    print('laspy:', laspy.__version__); \
    import sys; sys.path.insert(0, '/opt/FSCT/scripts'); \
    from run_tools import FSCT; print('FSCT importable: OK')"

ENV PYTHONPATH=/opt/FSCT/scripts:/opt/FSCT

LABEL org.opencontainers.image.title="fsct" \
      org.opencontainers.image.description="Forest Structural Complexity Tool (Krisanski et al. 2021) — TLS DL stem-point semantic seg + classical cylinder fit for DBH; CPU-only" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/fsct/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="GPL-3.0" \
      bradleylab.model.upstream="https://github.com/SKrisanski/FSCT" \
      bradleylab.model.paper="https://doi.org/10.3390/rs13081413" \
      bradleylab.model.task="stem-point semantic segmentation + DBH" \
      bradleylab.model.sensor="lidar:tls" \
      bradleylab.model.status="production" \
      bradleylab.model.gpu="none — CPU-only — FSCT hard-pins torch==1.9.0+cu111 with matching PyG extension wheels, which cannot target sm_90; an H100 allocation would be wasted" \
      bradleylab.model.weights.source="upstream-repo:in-tree" \
      bradleylab.model.licence.note="GPL-3.0 (upstream LICENSE file, verified 2026-09-04) — copyleft: anything linked into a distributed derivative must be GPL-compatible"

WORKDIR /opt/FSCT/scripts
ENTRYPOINT ["python", "/opt/FSCT/scripts/run_cli.py"]
CMD ["--help"]
