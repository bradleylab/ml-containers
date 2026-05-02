# NeuralHydrology — LSTM rainfall-runoff and streamflow prediction.
#
# Library from the NeuralHydrology team (Kratzert, Klotz, Gauch et al.)
# for training and running deep-learning hydrology models. Classical
# CAMELS-pretrained LSTM checkpoints are tiny and run trivially on
# CPU; users supply their own checkpoint directory and forcing data.
# A CUDA variant for continental-scale training is intentionally not
# baked here — training is the GPU-bound case, inference isn't, and
# this image targets inference.
#
# Upstream: https://github.com/neuralhydrology/neuralhydrology
# PyPI:    https://pypi.org/project/neuralhydrology/
# Stack:    Python 3.11-slim + PyTorch 2.5 CPU + neuralhydrology>=1.13
#
# No weights baked. Users bind-mount a directory containing the
# checkpoint (`run_dir/`, with `model_epochXXX.pt` and `config.yml`)
# and the forcing data (CAMELS layout or compatible).

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# netcdf4 needs libhdf5 + libnetcdf at runtime (binary wheels for these
# package the C libs, but a minimal toolchain is wise for source builds
# of numba/scipy on rare arch combos).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        libhdf5-dev libnetcdf-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU wheels — matches seisbench for ABI consistency across
# our CPU-slim images.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

# NeuralHydrology + transitive scientific stack. Loose lower bound on
# the package itself; pip resolves the rest.
RUN pip install \
        "neuralhydrology>=1.13" \
        "xarray>=2024.1" \
        "netcdf4>=1.6" \
        "h5py>=3.10"

# Smoke test: import the package, confirm console scripts resolve, and
# that LSTM head class is importable. No checkpoint or data fetch.
# (neuralhydrology stores its version at neuralhydrology.__about__.__version__,
# not on the top-level module — importlib.metadata is portable.)
RUN python -c "import torch, xarray as xr, numba; \
import importlib.metadata as im; \
from neuralhydrology.modelzoo.cudalstm import CudaLSTM; \
print('neuralhydrology', im.version('neuralhydrology'), '| torch', torch.__version__, '| xarray', xr.__version__, '| numba', numba.__version__); \
assert CudaLSTM is not None" \
    && nh-run --help > /dev/null

LABEL org.opencontainers.image.title="neuralhydrology" \
      org.opencontainers.image.description="NeuralHydrology — Python library for training and running deep-learning hydrology models (LSTM rainfall-runoff, streamflow prediction); CPU runtime; CAMELS-style inputs supplied at runtime." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/neuralhydrology/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="BSD-3-Clause" \
      bradleylab.model.upstream="https://github.com/neuralhydrology/neuralhydrology" \
      bradleylab.model.paper="https://doi.org/10.21105/joss.04050" \
      bradleylab.model.task="rainfall-runoff / streamflow prediction (LSTM)" \
      bradleylab.model.sensor="time-series:meteorological-forcing+streamflow" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="user-supplied (checkpoint directory bind-mounted at runtime; pretrained CAMELS checkpoints from the NeuralHydrology research blog)"

WORKDIR /work
CMD ["bash"]
