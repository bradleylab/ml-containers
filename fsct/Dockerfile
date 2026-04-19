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

FROM python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

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
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==1.9.0+cpu torchvision==0.10.0+cpu

RUN pip install \
        -f https://data.pyg.org/whl/torch-1.9.0+cpu.html \
        torch-scatter==2.0.8 \
        torch-sparse==0.6.11 \
        torch-cluster==1.5.9 \
        torch-spline-conv==1.2.1 \
        torch-geometric==1.7.2

# numpy + Cython must exist before hdbscan's source build runs (hdbscan
# 0.8.28 doesn't ship Python 3.9 manylinux wheels for recent glibc).
# Install them first, then use --no-build-isolation so the downstream
# source build picks up the installed numpy headers.
RUN pip install \
        numpy==1.22.3 \
        Cython==0.29.28

# Rest of FSCT's deps (upstream requirements minus torch-family + minus
# the broken sklearn==0.0 meta-package that errors on modern pip).
# hdbscan bumped to 0.8.38 which ships binary wheels and is API-compatible
# with FSCT's usage.
RUN pip install --no-build-isolation \
        laspy==2.1.2 lazrs \
        pandas==1.4.2 \
        scikit-learn==1.0.2 \
        scipy==1.8.0 \
        hdbscan==0.8.38 \
        scikit-image==0.19.2 \
        networkx==2.8 \
        python-louvain==0.16 \
        matplotlib==3.5.2 \
        tqdm

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

WORKDIR /opt/FSCT/scripts
ENTRYPOINT ["python", "/opt/FSCT/scripts/run_cli.py"]
CMD ["--help"]
