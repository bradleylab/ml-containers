# SeisBench — seismic phase picking for waveform data.
#
# Woollam et al. (SeisBench, 2022) — toolbox for ML-based seismology
# bundling maintained reimplementations of PhaseNet (Zhu & Beroza
# 2019) and EQTransformer (Mousavi et al. 2020) plus a public model
# zoo. Zero-shot on new stations and networks.
#
# CPU inference handles a station-day in minutes; H100 is unnecessary
# for typical use. On continental/decadal catalogs the bottleneck is
# I/O parallelism, not model compute, so this image stays CPU-only by
# design. A CUDA variant can be added later if needed.
#
# Upstream: https://github.com/seisbench/seisbench
# Paper:    https://doi.org/10.1785/0220210324  (Woollam et al. 2022, SRL)
# Stack:    Python 3.11-slim + PyTorch 2.5 CPU + ObsPy
#
# Weights are fetched on demand from the SeisBench model zoo (S3) the
# first time `Model.from_pretrained(...)` runs. Cache lives at
# /opt/seisbench-cache (env: SEISBENCH_CACHE_ROOT). Bind-mount a
# persistent host dir there to avoid re-fetching per job.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SEISBENCH_CACHE_ROOT=/opt/seisbench-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU wheels — laptop-runnable; matches the model's primary
# tier. Pin matches the version exercised in `deepforest/` so a
# coherent torch ABI lives across our CPU + GPU images.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

# SeisBench + ObsPy. Loose lower bounds; pinned at build time by the
# resolver. Verify on first build that the resolved versions are
# compatible with the smoke test below.
RUN pip install \
        "seisbench>=0.7" \
        "obspy>=1.4" \
        "h5py>=3.10" \
        "pandas>=2.0"

RUN mkdir -p /opt/seisbench-cache

# Smoke test: import + reach the model classes. No weight fetch at
# build time (those are runtime; build-time network is flaky).
RUN python -c "import seisbench, obspy, torch; \
import seisbench.models as sbm; \
print('seisbench', seisbench.__version__, '| obspy', obspy.__version__, '| torch', torch.__version__); \
assert sbm.PhaseNet is not None and sbm.EQTransformer is not None"

LABEL org.opencontainers.image.title="seisbench" \
      org.opencontainers.image.description="SeisBench seismic phase picking — PhaseNet (Zhu & Beroza 2019) + EQTransformer (Mousavi et al. 2020) via the SeisBench model zoo; CPU runtime." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/seisbench/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/seisbench/seisbench" \
      bradleylab.model.paper="https://doi.org/10.1785/0220210324" \
      bradleylab.model.task="seismic phase picking (P/S onset detection)" \
      bradleylab.model.sensor="waveform:seismic" \
      bradleylab.model.status="production" \
      bradleylab.model.weights.source="seisbench-model-zoo (runtime fetch into SEISBENCH_CACHE_ROOT=/opt/seisbench-cache)"

WORKDIR /work
CMD ["bash"]
