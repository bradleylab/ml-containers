# sam2

Portable Docker image of [SAM 2](https://github.com/facebookresearch/sam2) (Segment Anything Model 2) for bradleylab research compute.

`docker run` (or `apptainer run`) on any GPU host — Compute2, EC2, local — and segment any RGB image. Weights are downloaded from the Hugging Face Hub on first use and cached.

## Pull

```bash
# Docker
docker pull ghcr.io/bradleylab/sam2:latest

# Apptainer (Compute2, TGI RAILS)
apptainer pull docker://ghcr.io/bradleylab/sam2:latest
```

Tags follow the GitHub Actions metadata convention:

- `latest` — head of `main`
- `v0.1.0` — released versions (push a git tag to publish)
- `sha-<short>` — every successful CI build

## Run

### Automatic mask generation (no prompts)

The default mode. SAM 2 segments every plausible object in the image. Best for "what's in this scene" workflows like boulder fields, tree crowns, cells.

```bash
docker run --gpus all --rm \
  -v $PWD:/work \
  -v $PWD/.hf-cache:/home/runner/.cache/huggingface \
  ghcr.io/bradleylab/sam2:latest \
    --image  /work/scene.jpg \
    --output /work/masks.json \
    --mode   amg
```

### Box-prompted segmentation

```bash
docker run --gpus all --rm -v $PWD:/work \
  ghcr.io/bradleylab/sam2:latest \
    --image  /work/scene.jpg \
    --output /work/masks.json \
    --mode   box \
    --boxes  '[[10,10,200,200],[300,50,500,400]]'
```

### Point-prompted segmentation

```bash
docker run --gpus all --rm -v $PWD:/work \
  ghcr.io/bradleylab/sam2:latest \
    --image        /work/scene.jpg \
    --output       /work/masks.json \
    --mode         point \
    --points       '[[100,150],[250,150]]' \
    --point-labels '[1,1]'
```

### On Compute2

```bash
# One-time pull (login node):
apptainer pull docker://ghcr.io/bradleylab/sam2:latest

# In an SLURM batch script:
srun --gres=gpu:1 \
  apptainer run --nv \
    --bind /scratch2/fs1/.../work:/work \
    --bind /scratch2/fs1/.../.hf-cache:/home/runner/.cache/huggingface \
    sam2_latest.sif \
      --image /work/tile_001.tif \
      --output /work/tile_001_masks.json \
      --mode amg
```

`--nv` exposes the GPU; `--bind` mounts host paths.

## Output format

A single JSON file:

```json
{
  "image": "scene.jpg",
  "image_shape": [3000, 4000, 3],
  "mode": "amg",
  "model_id": "facebook/sam2.1-hiera-large",
  "device": "cuda",
  "n_masks": 47,
  "elapsed_s": 6.42,
  "masks": [
    {
      "id": 0,
      "bbox": [x, y, w, h],
      "area": 1234,
      "predicted_iou": 0.91,
      "stability_score": 0.97,
      "point_coords": [[x, y]],
      "crop_box": [x, y, w, h],
      "rle": {"size": [H, W], "counts": "..."}
    }
  ]
}
```

`rle` is a [COCO RLE](https://cocodataset.org/#format-results) object — decode with `pycocotools.mask.decode` (or any COCO-compatible library). Pass `--save-masks-dir DIR` to also write one binary PNG per mask.

## Choosing a model variant

Override `--model-id` with any of:

| HF id | Params | RAM (fp16) | Notes |
|---|---:|---:|---|
| `facebook/sam2.1-hiera-tiny` | 39 M | ~1 GB | Fast, lower quality. |
| `facebook/sam2.1-hiera-small` | 46 M | ~1.2 GB | |
| `facebook/sam2.1-hiera-base-plus` | 81 M | ~1.7 GB | Good quality / speed balance. |
| `facebook/sam2.1-hiera-large` | 217 M | ~2.4 GB | **Default**, best quality. |

## Design notes

- Base image: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (no compiled extensions in SAM 2 — runtime variant is enough).
- `TORCH_CUDA_ARCH_LIST` includes `9.0` for H100 support.
- Non-root user `runner` (uid 1000) so it works rootless under Apptainer on Compute2 / TGI RAILS.
- Weights are *not* baked into the image. They download on first run from HF Hub and cache under `$HF_HOME` (= `/home/runner/.cache/huggingface` by default). Bind a host path there to persist the cache across runs.
- Output is application-agnostic: raw masks in COCO RLE format. Project-specific code (georeferencing, filtering, polygonisation) lives in the consuming repo, not here.

## Build locally

```bash
docker build -t sam2:dev .
docker run --gpus all --rm -v $PWD:/work sam2:dev \
    --image /work/test.jpg --output /work/test_masks.json --mode amg
```

## License

Apache 2.0. SAM 2 itself is also Apache 2.0.
