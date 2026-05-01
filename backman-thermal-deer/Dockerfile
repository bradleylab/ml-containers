# Backman 2025 thermal animal detector — runtime container.
#
# Frame-level animal detection in thermal video. ONNX recurrent (LSTM)
# model with built-in frameSkip; takes 640x512 thermal frames and emits
# per-frame detections with class labels (arboreal vs ground).
#
# The model itself (model.onnx + generateVideoPredictions.py) is
# distributed by the paper authors as an inferenceExample/ directory
# and is NOT baked into this image. Mount it at runtime via
# --container-mounts and pass the path to the wrapper script as
# --model-dir.
#
# Used end-to-end on Tyson 2026-03-23 thermal flights (DJI XT2): 21
# deer detected after size + persistence filtering across 6 flights /
# 30 segments / 128,638 frames. Wrapper at:
#   tyson-deer-survey/tyson-thermal-deer-survey/scripts/run_backman_inference.py
#
# CPU-only by default. The ONNX model is small (LSTM head on a thin
# CNN backbone) and runs comfortably without a GPU; if a GPU is
# available, swap `onnxruntime` for `onnxruntime-gpu` and rebuild.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OpenCV needs libgl1 + libglib2.0 even when reading video headlessly.
# ffmpeg supplies the codecs OpenCV uses for MP4 demuxing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Runtime stack matching what the upstream generateVideoPredictions.py
# expects: ONNX inference, OpenCV for video I/O + drawing, numpy/pandas
# for the prediction CSV, tqdm for progress.
RUN pip install --upgrade pip \
    && pip install \
        "onnxruntime>=1.17,<2" \
        "opencv-python-headless>=4.9" \
        "numpy>=1.26,<3" \
        "pandas>=2.2" \
        "tqdm>=4.66"

# Smoke-test imports at build time so the GHA build fails fast on
# dependency drift rather than at first run on Compute2.
RUN python -c "import onnxruntime, cv2, numpy, pandas, tqdm; print('backman runtime OK', onnxruntime.__version__, cv2.__version__)"

LABEL org.opencontainers.image.title="backman-thermal-deer" \
      org.opencontainers.image.description="Backman et al. 2025 thermal animal detector — runtime container (model.onnx + generateVideoPredictions.py bind-mounted at runtime)" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/backman-thermal-deer/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="NOASSERTION" \
      bradleylab.model.upstream="https://doi.org/10.5281/zenodo.14799290" \
      bradleylab.model.paper="https://doi.org/10.1111/2041-210X.70006" \
      bradleylab.model.task="frame-level animal detection in thermal video" \
      bradleylab.model.sensor="thermal:dji-xt2-640x512" \
      bradleylab.model.status="production" \
      bradleylab.model.weights.source="bind-mount at runtime (Zenodo deposit; license unverified)"

WORKDIR /work
ENTRYPOINT ["python"]
