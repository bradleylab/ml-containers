# dofa-clip

[DOFA-CLIP](https://arxiv.org/abs/2503.06312) (Xiong et al. 2025) — multispectral CLIP via DOFA's wavelength-conditioned image encoder + SigLIP-style text alignment. Pretrained on **GeoLangBind-2M** (~2M EO image-caption pairs). Image trunk is ViT-L/14 with the dynamic hypernetwork from base DOFA; text encoder is the so400m SigLIP variant.

This is **Path B** of the DOFA-CLIP container — built against the upstream `xiong-zhitong/DOFA-CLIP` repo's vendored open_clip fork. Path A (the `BiliSakura/DOFA-CLIP-{ViT-B-16,VIT-L-14}` HF transformers mirrors) was evaluated and does not work as published; see "What happened to Path A?" below.

## ⚠ License

**Weights are CC-BY-NC-4.0** per the [`earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO`](https://huggingface.co/earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO) HF model card. The only NC-licensed image in the bradleylab catalog as of v1.

- Use for academic / non-commercial research is **OK**.
- Container redistribution is OK as long as the upstream license is preserved (it is — see the GHCR package metadata and the `Weights license` row in MODEL_CARDS.md).
- **Commercial use requires explicit permission from the upstream authors** (Xiong et al., contact `xiongzhitong@gmail.com`).

The Python code (this repo + `xiong-zhitong/DOFA-CLIP`) is Apache-2.0; only the trained checkpoint carries the NC restriction.

## Image tag

`ghcr.io/bradleylab/dofa-clip:v1` (also `:latest`, `:torch2.5-cpu`).

Multi-arch: `linux/amd64` + `linux/arm64`.

## Stack

- Base: `python:3.11-slim`
- PyTorch 2.5.1 + torchvision 0.20.1 (CPU wheels)
- Vendored open_clip fork from [`xiong-zhitong/DOFA-CLIP`](https://github.com/xiong-zhitong/DOFA-CLIP) — installed editable from `/opt/DOFA-CLIP/open_clip`
- `timm`, `einops`, `transformers>=4.40,<5`, `huggingface_hub<1.0`, Pillow, safetensors

### Why a vendored open_clip?

Upstream open_clip does not support a `wavelengths=` argument to the image trunk. The fork at `xiong-zhitong/DOFA-CLIP/open_clip` adds the dynamic hypernetwork that converts per-band wavelength tensors into input projection weights. The image trunk is called as `model.visual.trunk(image, wavelengths)` rather than the standard `model.encode_image(image)`.

## Weights

Baked at build time (~1.7 GB safetensors). Single variant — the so400m-384-EO checkpoint:

- Image trunk: ViT-L/14 with wavelength-conditioning hypernetwork
- Text encoder: SigLIP so400m
- Embedding dim: 1152
- Image resolution: 384×384
- Context length (text): 64 tokens

Bake-at-build means the runtime container is offline-capable. To use a different open_clip-format checkpoint (e.g. a future re-release), pass `--model-id "hf-hub:..."` and bind-mount `$HF_HOME=/opt/hf-cache` so the lazy download is persistent.

## Inference

### RGB

```bash
docker run --rm \
  -v "$PWD:/work" \
  ghcr.io/bradleylab/dofa-clip:v1 \
  python /opt/scripts/dofa_clip_score.py \
    --image /work/photo.jpg \
    --rgb \
    --prompt "a satellite image of an active mine" \
    --prompt "a satellite image of dense forest" \
    --prompt "a satellite image of farmland" \
    --out /work/scores.csv
```

### Multispectral (Sentinel-2 12-band)

```bash
docker run --rm \
  -v "$PWD:/work" \
  ghcr.io/bradleylab/dofa-clip:v1 \
  python /opt/scripts/dofa_clip_score.py \
    --image /work/s2_tile.npy \
    --sentinel2-12band \
    --prompt "a satellite image of cloudy weather" \
    --prompt "a clear satellite image of forest" \
    --out /work/scores.csv
```

Multispectral input must be a `.npy` or `.pt` file with a `(C, H, W)` float32 tensor. The script resizes spatially to 384×384 with bilinear interpolation. Number of bands `C` must match the wavelength list length.

### Wavelength conventions

| flag | bands |
|---|---|
| `--rgb` (default) | RGB at [0.665, 0.560, 0.490] µm |
| `--sentinel2-12band` | S2 L1C without B10: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 |
| `--sentinel2-10band` | S2 land-only: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 |
| `--sentinel1` | S1 VV+VH (both at C-band 5.405 µm) |
| `--wavelengths 0.49 0.56 0.665 ...` | arbitrary user-supplied list (micrometers) |

For Landsat / Gaofen / hyperspectral, pass `--wavelengths` explicitly with the per-band central wavelengths in micrometers.

## Output

CSV columns: `prompt`, `cosine`, `sigmoid_prob`.

- `cosine` — raw L2-normalised dot product between image and text embeddings. Use absolute thresholds for screening.
- `sigmoid_prob` — `sigmoid(cosine * logit_scale + logit_bias)`, the SigLIP-style score. **Per-prompt independent** (does NOT sum to 1 across the panel) — different from RemoteCLIP / CLIP softmax. Each prompt is its own binary classifier.

## What happened to Path A?

The `BiliSakura/DOFA-CLIP-{ViT-B-16,VIT-L-14}` HF transformers mirrors look loadable via standard `transformers.CLIPModel`, and we attempted that path first. Two failures showed up:

1. **B-16 image-size mismatch.** `preprocessor_config.json` was uploaded with `size=384 / crop_size=384`, but the model's `vision_config.image_size` is 224. Loading raises `ValueError: Input image size (384*384) doesn't match model (224*224)`. Workaround: override `processor.image_processor.{size,crop_size}` to match the model's declared image size.

2. **Text encoder weights silently dropped (both variants).** The checkpoints store text encoder self-attention as combined `in_proj.{weight,bias}` (open_clip / SigLIP convention — Q+K+V in a single projection matrix). HF's `CLIPModel` expects separate `q_proj`/`k_proj`/`v_proj`. The mismatched names trigger a "Some weights of the model checkpoint were not used" warning and **every text encoder attention layer is left randomly initialized**. Inference then produces text embeddings that collapse to nearly identical vectors across unrelated prompts (pairwise cosine ~1.00), making image-text scoring meaningless.

The vendored open_clip fork at `xiong-zhitong/DOFA-CLIP/open_clip` accepts the `in_proj.*` naming directly and produces non-degenerate text embeddings (verified at build time — the smoke test asserts `pair_cos < 0.95` and that an airport image scores higher on "a busy airport" than on "a forest" / "a stadium"). That's why this container ships Path B instead.

## Use cases

- **Multispectral text-prompt screening** of Sentinel-2 / Sentinel-1 / Gaofen tiles. The headline differentiator from `bradleylab/remoteclip` (which is RGB-only).
- **EO scene classification by text prompt** on RGB. Sister to RemoteCLIP — different training corpus (GeoLangBind-2M vs RET-3 + SEG-4 + DET-10), useful for ensemble screening.

For permissively-licensed multispectral embeddings (no CLIP / text alignment), use `bradleylab/dofa` (CC-BY-4.0). For permissive RGB CLIP, use `bradleylab/remoteclip` (Apache-2.0).

## Run on Compute2

CPU works for one-shot scoring (~5-10 s/image after model load). For batch screening across many tiles, submit a CPU job array on `general-cpu` and bind-mount the so400m-384-EO HF cache to RIS storage. GPU H100 batch variant deferred until a workload warrants it.

## Caveats

- **NC license forces compliance burden on downstream users.** Make sure anyone integrating this knows.
- **Single variant** in v1 (so400m-384-EO). The smaller B-16 / L-14 mirrors are unusable as published.
- **No fine-tuning hooks** in the inference script — pure forward pass, embedding-only output.
- **Multispectral preprocessing is naive** — bilinear resize to 384×384 with no per-band normalization. For best results, normalize bands to similar dynamic ranges before passing in.
- The vendored open_clip pyproject is missing `einops`; we add it explicitly. If upstream re-pins, this should be unaffected.
