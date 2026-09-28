# copernicus-fm — Copernicus-FM, the unified Copernicus foundation model.
#
# Wang et al. (ICCV 2025), "Towards a Unified Copernicus Foundation Model for
# Earth Vision". A ViT-B/16 that extends DOFA's wavelength-conditioned
# hypernetwork in two directions: spectral sensors are described by central
# wavelength AND bandwidth, and non-spectral variables (Sentinel-5P trace gases,
# the Copernicus DEM) by a language embedding of the variable's name. Location,
# acquisition time and patch area enter as optional Fourier-encoded metadata
# tokens, and the patch-embedding kernel is resized per modality, so one set of
# weights takes Sentinel-1 GRD, Sentinel-2 TOA, Sentinel-3 OLCI, Sentinel-5P and
# DEM input. Pretrained on Copernicus-Pretrain (18.7M images) with masked image
# modeling and continual distillation.
#
# The model code is TorchGeo's implementation (torchgeo.models.copernicusfm_base),
# not the upstream research code: TorchGeo is MIT, carries no CC-BY-NC MAE code,
# and is the same library the lab's `dofa` image uses.
#
# ---------------------------------------------------------------------
# WEIGHTS ARE NOT BAKED.
# ---------------------------------------------------------------------
# The ViT-B checkpoint (557,931,327 bytes, CC-BY-4.0, ungated) is the only file
# the model needs for spectral input. It was pretrained on Llama 3.2 encodings
# of variable names, and Meta's Llama 3.2 Community License places conditions on
# distributing a model trained with Llama outputs that upstream's CC-BY-4.0
# release does not address. So the lab keeps the file on its own storage rather
# than redistributing it in a public image: scripts/stage_weights.py fetches it
# through TorchGeo's pinned weights enum to Storage3 Active/copernicus-fm/weights,
# and a run mounts that directory at /weights, where TORCH_HOME points.
#
# The Llama-3.2-1B encodings of the variable names (116,962 bytes) are NOT baked:
# they are distributed under the Llama 3.2 Community License, not CC-BY-4.0, and
# variable-mode users fetch them themselves (README.md).
#
# Upstream: https://github.com/zhu-xlab/Copernicus-FM @ da95635
#           Apache-2.0 with MIT and CC-BY-NC-4.0 third-party portions (none shipped here)
# Model:    TorchGeo 0.10.0, torchgeo/models/copernicusfm.py (MIT)
# Weights:  https://huggingface.co/torchgeo/copernicus-fm @ f395812 (CC-BY-4.0),
#           a renamed, unmodified copy of wangyi111/Copernicus-FM
# Paper:    https://arxiv.org/abs/2503.11849

FROM python:3.12-slim-bookworm

# enroot exposes the GPU only when the image asks for it. On a CPU
# partition override with  srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void ...
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTHONNOUSERSITE keeps the user's ~/.local off sys.path: enroot bind-mounts
# $HOME, and a host package shadowing an image one has broken images in this
# catalog before. TORCH_HOME points into the staged-weights mount; torch.hub
# resolves it at every load, so it lives in the image environment.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TORCH_HOME=/weights/torch-cache

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# torch 2.13.0 / torchvision 0.28.0 are the versions TorchGeo 0.10.0's own
# uv.lock was released against. PyTorch publishes 2.13.0 for CUDA 12.6, 12.9,
# 13.0 and 13.2; 13.2 is above what Compute2's driver (580.105.08) supports, and
# 12.9 matches `nasa-ibm-lunar-fm`, which also runs TorchGeo on those nodes.
# PyTorch builds its 12.9 x86-64 wheels with sm_90 kernels for the H100.
# constraints.txt is the full resolved set, held to TorchGeo's lockfile versions
# wherever the lockfile has the package, and it pins torch to the +cu129 build so
# the resolver cannot swap in the PyPI one.
COPY constraints.txt /opt/constraints.txt
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install -c /opt/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        torch torchvision torchgeo==0.10.0

COPY scripts/cfm-verify-eurosat-s1 /usr/local/bin/
COPY scripts/stage_weights.py /opt/copernicus-fm/stage_weights.py

RUN mkdir -p /work /data /weights

# Build smoke test. torch and torchgeo import in separate steps under
# faulthandler, because a compiled-extension segfault in this stack has been
# silent in `terramind` and `prithvi-eo` until the imports were split.
RUN python -X faulthandler -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda); assert torch.version.cuda and torch.version.cuda.startswith('12.9'), 'CUDA 12.9 torch was replaced'"
RUN python -X faulthandler -c "import importlib.metadata as im, torchgeo.models; print('torchgeo', im.version('torchgeo'))"

# Offline forward passes through a randomly initialized encoder, since the
# weights are staged, not baked. The proxy variables point at a closed port, so
# any attempt to download during the test fails the build. Wavelengths and bandwidths
# (nm) are the paper's Table 9 values and upstream's Copernicus-Bench configs;
# kernel sizes are the paper's (16 for S1/S2, 4 for S5P). The variable-mode pass
# uses a random 2048-vector in place of a Llama name encoding: it checks that
# the path runs, not what it computes. The S1 metadata values are arbitrary but
# inside the encoder's ranges, to exercise the metadata path the S2 pass skips.
RUN https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 \
    python -X faulthandler -c "\
import importlib.metadata as im, torch; \
torch.manual_seed(0); torch.set_grad_enabled(False); \
assert torch.hub.get_dir() == '/weights/torch-cache/hub', torch.hub.get_dir(); \
from torchgeo.models import copernicusfm_base; \
m = copernicusfm_base(weights=None).eval(); \
unknown = torch.full((2, 4), float('nan')); \
s2 = m(torch.randn(2, 13, 224, 224), unknown, \
       wavelengths=[440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1370, 1610, 2190], \
       bandwidths=[20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180], \
       input_mode='spectral', kernel_size=16); \
known = torch.tensor([[-90.3, 38.6, 19000.0, 0.0256]] * 2); \
s1 = m(torch.randn(2, 2, 224, 224), known, wavelengths=[5e7, 5e7], bandwidths=[1e9, 1e9], \
       input_mode='spectral', kernel_size=16); \
var = m(torch.randn(2, 1, 56, 56), unknown, language_embed=torch.randn(2048), \
        input_mode='variable', kernel_size=4); \
out = {'S2 13-band 224px': s2, 'S1 VV+VH 224px': s1, 'variable 56px': var}; \
bad = {k: tuple(v.shape) for k, v in out.items() if tuple(v.shape) != (2, 768) or not torch.isfinite(v).all()}; \
assert not bad, f'wrong shape or non-finite output: {bad}'; \
assert not torch.allclose(s2[0], s2[1]), 'embedding ignores its input'; \
print({k: tuple(v.shape) for k, v in out.items()}); \
print('torchgeo', im.version('torchgeo'), '| torchvision', im.version('torchvision'), '| timm', im.version('timm'), '| kornia', im.version('kornia'), '| einops', im.version('einops'), '| lightning', im.version('lightning')); \
print('SMOKE OK')"

LABEL org.opencontainers.image.title="copernicus-fm" \
      org.opencontainers.image.description="Copernicus-FM — ViT-B/16 Earth-observation foundation model with spectral (wavelength + bandwidth) and variable-name hypernetworks and metadata encoding, for Sentinel-1/2/3/5P and Copernicus DEM input, run through TorchGeo. Weights mounted at runtime" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/copernicus-fm/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT AND CC-BY-4.0" \
      bradleylab.model.upstream="https://github.com/zhu-xlab/Copernicus-FM" \
      bradleylab.model.paper="https://arxiv.org/abs/2503.11849" \
      bradleylab.model.task="Earth-observation embedding extraction (foundation model) for frozen-encoder or fine-tuned classification, segmentation, change detection and regression" \
      bradleylab.model.sensor="image:multi (Sentinel-1 GRD, Sentinel-2 TOA, Sentinel-3 OLCI, any spectral sensor by wavelength and bandwidth; Sentinel-5P CO/NO2/SO2/O3 and Copernicus DEM by variable-name embedding)" \
      bradleylab.model.status="experimental" \
      bradleylab.model.gpu="sm_90" \
      bradleylab.model.weights.source="NOT baked — huggingface.co/torchgeo/copernicus-fm @ f395812, CopernicusFM_ViT_base_varlang-085350e4.pth (558 MB), staged lab-side on Storage3 at Active/copernicus-fm/weights by /opt/copernicus-fm/stage_weights.py and mounted at /weights; Llama-3.2 variable-name embeddings not staged" \
      bradleylab.model.licence.note="Model code is TorchGeo (MIT). Weights CC-BY-4.0, attribution to Wang et al. 2025 (arXiv:2503.11849). Upstream research code is Apache-2.0 with MIT and CC-BY-NC-4.0 portions and is not in the image. The variable-name embeddings are under the Llama 3.2 Community License and are not baked"

WORKDIR /work
CMD ["bash", "-c", "echo 'copernicus-fm ready: from torchgeo.models import copernicusfm_base, CopernicusFM_Base_Weights (see README.md)'"]
