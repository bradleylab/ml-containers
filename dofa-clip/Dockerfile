# dofa-clip — multispectral CLIP via DOFA's wavelength-conditioned
# image encoder + SigLIP-style text alignment (Xiong et al. 2025).
#
# Path B build: uses the upstream `xiong-zhitong/DOFA-CLIP` repository
# (Apache-2.0 code) which vendors a fork of open_clip with the
# wavelength-conditioning trunk wired in. The standard HF transformers
# Path A (`BiliSakura/DOFA-CLIP-{ViT-B-16,VIT-L-14}` mirrors) was
# evaluated and found broken: the text encoder self-attention is
# stored as combined `in_proj.{weight,bias}` (open_clip / SigLIP
# convention) but HF CLIPModel expects separate q/k/v_proj — weights
# silently dropped, text embeddings collapse to ~identical vectors
# across unrelated prompts. See the deferral note in
# `geospatial-containers/STATUS.md` for the diagnosis.
#
# This container ships the so400m-384-EO checkpoint
# (`earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO`, ~1.7 GB safetensors)
# baked at build time. It is the only DOFA-CLIP variant exposed by the
# upstream open_clip fork.
#
# # IMPORTANT — LICENSE
#
# Code: Apache-2.0 (this repo + xiong-zhitong/DOFA-CLIP).
# Weights: **CC-BY-NC-4.0** per the HF model card metadata for
# `earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO`. Use of this image is
# restricted to non-commercial purposes; commercial users must obtain
# explicit upstream permission (xiongzhitong@gmail.com). This is the
# only NC-licensed image in the bradleylab catalog as of v1.
#
# Upstream: https://github.com/xiong-zhitong/DOFA-CLIP
# Paper:    https://arxiv.org/abs/2503.06312 (Xiong et al. 2025)
# Weights:  https://huggingface.co/earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO
# Stack:    python:3.11-slim + PyTorch 2.5 CPU + vendored open_clip + timm

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

# git is required to clone the upstream xiong-zhitong/DOFA-CLIP repo
# at build time (it ships the vendored open_clip we need to install).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU wheels — matches seisbench / remoteclip / geoclip / dofa
# for ABI consistency across the catalog.
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1

# Clone the upstream repo and install its vendored open_clip fork as
# editable. einops is missing from the upstream pyproject; add it
# explicitly. transformers is needed for the HF tokenizer used at
# inference.
RUN git clone --depth 1 https://github.com/xiong-zhitong/DOFA-CLIP.git /opt/DOFA-CLIP \
    && cd /opt/DOFA-CLIP/open_clip \
    && pip install -e . \
        einops \
        "transformers>=4.40,<5" \
        "huggingface_hub>=0.25,<1.0" \
        "Pillow>=10.0" \
        safetensors

RUN mkdir -p /opt/hf-cache /opt/scripts
COPY scripts/dofa_clip_score.py /opt/scripts/dofa_clip_score.py

# Build-time smoke test #1: imports + vendored open_clip reachable.
# importlib.metadata.version per ml-containers rules.
RUN python -c "import torch, open_clip, transformers, einops; \
import importlib.metadata as im; \
print('open_clip', im.version('open-clip-torch'), '| transformers', im.version('transformers'), '| timm', im.version('timm'), '| torch', torch.__version__); \
from open_clip import create_model_from_pretrained, get_tokenizer; \
assert callable(create_model_from_pretrained); \
assert callable(get_tokenizer)"

# Build-time smoke test #2 + weight bake: load so400m-384-EO, run
# forward pass on the bundled airplane.png with text panel, assert
# (a) text embeddings differentiate (pairwise cos < 0.95) and
# (b) the airport prompt scores higher than unrelated prompts.
# This catches the Path A failure mode (text-attention weight drop)
# and validates the vendored open_clip is actually loading the
# pretrained text encoder.
RUN python -c "import torch; torch.set_grad_enabled(False); \
import torch.nn.functional as F; \
from open_clip import create_model_from_pretrained, get_tokenizer; \
from PIL import Image; \
hf_repo = 'hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO'; \
model, preprocess = create_model_from_pretrained(hf_repo); \
tokenizer = get_tokenizer(hf_repo); model.eval(); \
prompts = ['a busy airport', 'a forest', 'a stadium']; \
text = tokenizer(prompts, context_length=model.context_length); \
txt = F.normalize(model.encode_text(text), dim=-1); \
pair = float((txt[0] @ txt[1]).item()); \
assert pair < 0.95, f'text embeddings collapsed: cos[0,1]={pair:.4f}'; \
img = preprocess(Image.open('/opt/DOFA-CLIP/airplane.png').convert('RGB')).unsqueeze(0); \
wvs = torch.tensor([0.665, 0.560, 0.490]); \
feat = model.visual.trunk(img, wvs); \
img_n = F.normalize(feat[0] if isinstance(feat, tuple) else feat, dim=-1); \
cos = (img_n @ txt.T).squeeze(0).tolist(); \
print('image-text cos:', dict(zip(prompts, [round(c, 4) for c in cos]))); \
assert cos[0] > cos[1] and cos[0] > cos[2], f'airport not top-1: {cos}'; \
print('SMOKE PASS')" \
    && du -sh /opt/hf-cache \
    && echo 'baked DOFA-CLIP so400m-384-EO weights at /opt/hf-cache'

LABEL org.opencontainers.image.title="dofa-clip" \
      org.opencontainers.image.description="DOFA-CLIP — multispectral CLIP via wavelength-conditioned image encoder + SigLIP text alignment (Xiong et al. 2025). CPU runtime. Vendored open_clip fork from xiong-zhitong/DOFA-CLIP. WEIGHTS LICENSE CC-BY-NC-4.0 (non-commercial only)." \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/dofa-clip/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="CC-BY-NC-4.0" \
      bradleylab.model.upstream="https://github.com/xiong-zhitong/DOFA-CLIP" \
      bradleylab.model.paper="https://arxiv.org/abs/2503.06312" \
      bradleylab.model.task="multispectral / RGB image-text retrieval and zero-shot scene scoring (CLIP architecture, SigLIP text encoder, DOFA wavelength-conditioned image trunk)" \
      bradleylab.model.sensor="image:multispectral,rgb-satellite" \
      bradleylab.model.status="utility" \
      bradleylab.model.weights.source="HuggingFace Hub earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO via vendored open_clip fork; baked at build (~1.7 GB safetensors)" \
      bradleylab.model.weights.license="CC-BY-NC-4.0 (per HF model card; commercial use prohibited without explicit upstream permission)"

WORKDIR /work
CMD ["bash"]
