# remoteclip

Vision-language foundation model for remote sensing — CLIP
architecture fine-tuned on a 12× larger remote-sensing pre-training
corpus (Liu, Chen et al. 2024, IEEE TGRS). Three OpenCLIP-format
checkpoints are distributed via Hugging Face Hub: `RN50`, `ViT-B-32`,
`ViT-L-14`.

RemoteCLIP itself is a checkpoint, not a package. This container
bundles the [`open-clip-torch`](https://github.com/mlfoundations/open_clip)
architecture and `huggingface_hub`; checkpoints are pulled at runtime
into `$HF_HOME=/opt/hf-cache`.

Laptop-runnable; CLIP-sized models are very fast on CPU. The Compute2
path is for building searchable embedding indices across large
archives of tiles (embarrassingly parallel batch job where H100
throughput matters); a CUDA variant can be added then.

## Image tag

`ghcr.io/bradleylab/remoteclip:v1` (also `:latest`, `:torch2.5-cpu`)

## Stack

- Base: `python:3.11-slim`
- PyTorch 2.5.1 + torchvision 0.20.1 (CPU wheels)
- `open-clip-torch >= 2.20`
- `huggingface_hub >= 0.25`
- `Pillow >= 10`

## Weights

Not baked. Three RemoteCLIP variants live at
[`chendelong/RemoteCLIP`](https://huggingface.co/chendelong/RemoteCLIP)
on Hugging Face Hub:

| Variant | File | Approx. size |
|---|---|---|
| RN50 | `RemoteCLIP-RN50.pt` | ~400 MB |
| ViT-B-32 | `RemoteCLIP-ViT-B-32.pt` | ~600 MB |
| ViT-L-14 | `RemoteCLIP-ViT-L-14.pt` | ~1.7 GB |

The first call to `hf_hub_download(...)` downloads the chosen
variant. Bind-mount a persistent host directory at
`/opt/hf-cache` so each variant only downloads once per host:

```bash
docker run --rm -it \
  -v "$PWD/hf-cache:/opt/hf-cache" \
  -v "$PWD/data:/data" \
  ghcr.io/bradleylab/remoteclip:v1
```

## Inference

```python
import torch, open_clip
from PIL import Image
from huggingface_hub import hf_hub_download

variant = "ViT-L-14"  # or "RN50" or "ViT-B-32"

ckpt = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{variant}.pt")
model, _, preprocess = open_clip.create_model_and_transforms(variant)
tokenizer = open_clip.get_tokenizer(variant)

state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("missing:", missing, "| unexpected:", unexpected)
model.eval()

# Zero-shot scene classification:
image = preprocess(Image.open("/data/tile.png")).unsqueeze(0)
text = tokenizer(["a satellite image of a forest",
                  "a satellite image of a city",
                  "a satellite image of farmland"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features  = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features  /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).softmax(dim=-1)
print(similarity)
```

For image-text retrieval, build embedding indices and run nearest-
neighbour queries; see the upstream
[`retrieval.py`](https://github.com/ChenDelong1999/RemoteCLIP/blob/main/retrieval.py).

## Inputs

- RGB image readable by Pillow (PNG, JPEG, GeoTIFF via Pillow + rasterio
  if added). Default OpenCLIP transforms expect 224×224 (RN50, ViT-B-32)
  or 224×224 + ViT-L-14 patch size 14.
- Text prompts of arbitrary length (tokenized to 77 tokens by default).

## Run on Compute2

Inference is CPU-trivial; for batch embedding across many tiles,
submit a CPU job array on `general-cpu`. A future CUDA variant would
give better throughput — not on the roadmap until a use case lands.

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=4 \
       --mem=8G \
       --time=04:00:00 \
       --array=0-99 \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+remoteclip+v1.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/opt/hf-cache,/scratch2/fs1/alexander.s.bradley/tiles:/data \
         bash -lc "export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/embed_tile_batch.py"'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — see
`~/.claude/rules/research-infrastructure.md`.

## Limitations

- CPU-only by design. Building embedding indices over 100k+ tiles
  benefits from GPU; add a CUDA variant when that workload lands.
- The fine-tuning corpus is RGB-only; multispectral tiles need a
  band-selection step before encoding.
- Default OpenCLIP transforms resize to 224×224 — not appropriate for
  large-AOI mosaics without tiling.
