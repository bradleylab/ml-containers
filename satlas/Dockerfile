# Satlas — pre-trained foundation models for satellite + aerial imagery.
#
# Allen AI's SatlasPretrain (Bastani et al., ICCV 2023) — backbones
# pre-trained on a large remote-sensing corpus, distributed via the
# `satlaspretrain_models` loader package. Available variants cover
# Sentinel-2 (RGB and 9-band MS), Sentinel-1 (VH+VV), Landsat 8/9
# (all bands), and 0.5–2 m/px aerial imagery, with Swin-v2-{Base,Tiny}
# and ResNet{50,152} backbones.
#
# This image is GPU-primary (Compute2 H100 sm_90) since the course
# tier for Satlas is "Compute2 primary; runs on laptop but slow."
# CPU inference still works — pass `device='cpu'` to
# `Weights().get_pretrained_model(...)`.
#
# Upstream: https://github.com/allenai/satlaspretrain_models
# Paper:    https://openaccess.thecvf.com/content/ICCV2023/html/Bastani_SatlasPretrain_A_Large-Scale_Dataset_for_Remote_Sensing_Image_Understanding_ICCV_2023_paper.html
# Stack:    nvidia/cuda 12.1 + Python 3.11 + PyTorch 2.5 cu121 + satlaspretrain_models
#
# Weights NOT baked. The `Weights().get_pretrained_model(model_id)`
# call fetches the .pth from `allenai/satlas-pretrain` on HF Hub each
# time it runs (the upstream loader does NOT cache on disk by design;
# weights are streamed into RAM via `requests.get` + `BytesIO`). For
# repeated jobs, wrap the loader with a manual disk cache or
# pre-download with `wget` and pass a local path.

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
        wget ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
    && wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.11 /tmp/get-pip.py && rm /tmp/get-pip.py \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch cu121 wheels — H100 sm_90 supported. Matches deepforest /
# pointstowood for ABI consistency across our GPU images.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1

# Satlas loader + light deps (the package itself only needs torch /
# torchvision / requests / matplotlib; we add Pillow for image I/O).
RUN pip install \
        "satlaspretrain-models>=0.3.1" \
        "Pillow>=10.0" \
        "requests>=2.31"

# Smoke test: import + reach the public Weights API. No network,
# no model load. importlib.metadata for portable version
# (per ~/.claude/rules/ml-containers.md).
RUN python -c "import torch, satlaspretrain_models; \
import importlib.metadata as im; \
from satlaspretrain_models.utils import Backbone, Head, SatlasPretrain_weights; \
w = satlaspretrain_models.Weights(); \
print('satlaspretrain-models', im.version('satlaspretrain-models'), '| torch', torch.__version__, '| cuda-build', torch.version.cuda); \
print('available checkpoints:', len(SatlasPretrain_weights)); \
assert callable(w.get_pretrained_model)"

LABEL org.opencontainers.image.title="satlas" \
      org.opencontainers.image.description="Satlas (Allen AI) — SatlasPretrain foundation models for Sentinel-2/-1, Landsat 8/9, and aerial imagery; H100 sm_90; weights via satlaspretrain_models loader from HF Hub at runtime." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/satlas/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0" \
      bradleylab.model.upstream="https://github.com/allenai/satlaspretrain_models" \
      bradleylab.model.paper="https://openaccess.thecvf.com/content/ICCV2023/html/Bastani_SatlasPretrain_A_Large-Scale_Dataset_for_Remote_Sensing_Image_Understanding_ICCV_2023_paper.html" \
      bradleylab.model.task="remote-sensing pretrained backbones (Swin-v2 / ResNet) for downstream classification, detection, segmentation, regression" \
      bradleylab.model.sensor="image:multi (Sentinel-2 RGB+MS, Sentinel-1 SAR, Landsat 8/9 all-bands, aerial RGB)" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="huggingface.co/allenai/satlas-pretrain (runtime fetch via satlaspretrain_models.Weights().get_pretrained_model — no on-disk cache by upstream design)"

WORKDIR /work
CMD ["bash"]
