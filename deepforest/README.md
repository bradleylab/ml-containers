# deepforest

Aerial RGB tree-crown detection via DeepForest 2.x (Weinstein et al.).
RetinaNet-style detector trained on NEON aerial imagery; the
pretrained `weecology/deepforest-tree` checkpoint is fetched from
Hugging Face Hub on first call.

## Image tag

`ghcr.io/bradleylab/deepforest:v1` (also `:latest`,
`:torch2.5-cu121`)

## Stack

- CUDA 12.1 + cuDNN 8 (H100 sm_90 supported)
- PyTorch 2.5.1 + torchvision 0.20.1 (cu121 wheels)
- `deepforest >= 2.1.0`
- `opencv-python-headless` (avoids libGL conflicts inside container)
- `huggingface_hub` for model fetch

## Weights

Not baked into the image. The first call to
`deepforest.main.deepforest()` pulls
`weecology/deepforest-tree:main` from Hugging Face Hub. Cache lives
at `$HF_HOME` (set to `/opt/hf-cache` in the image). Bind-mount a
persistent host directory there so the model only downloads once per
host:

```bash
srun --container-image=/storage1/.../bradleylab+deepforest+latest.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/opt/hf-cache,/scratch2/.../inputs:/inputs \
     --container-writable \
     bash -lc 'python -c "from deepforest import main; m = main.deepforest(); m.use_release()"'
```

(The first run will download ~50-100 MB; subsequent runs reuse the
cache.)

## Inference

```python
from deepforest import main
m = main.deepforest()
m.use_release()                       # load pretrained NEON checkpoint
boxes = m.predict_image("/inputs/tile.tif")
boxes.to_csv("/outputs/predictions.csv")
```

For tiled inference on a large mosaic:

```python
m.predict_tile(
    "/inputs/big_mosaic.tif",
    patch_size=400, patch_overlap=0.05,
    return_plot=False,
)
```

See https://deepforest.readthedocs.io for the full API.

## Inputs

- 3-band RGB GeoTIFF (or any rasterio-readable raster).
- Default `patch_size=400` px assumes ~10-cm GSD; rescale for other
  resolutions.

## Limitations

- Pretrained on NEON sites — temperate broadleaf North America. Has
  not been calibrated against Tyson aerial RGB; first-run results are
  exploratory.
- 2D crown bounding-boxes only. For 3D crown geometry, pair with a
  point-cloud-based segmenter (AMS3D, SegmentAnyTree).

## Run on Compute2

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-gpu \
       --gpus=1 \
       --mem=32G \
       --time=02:00:00 \
       --wrap='srun --container-image=/storage1/.../bradleylab+deepforest+latest.sqsh \
                    --container-mounts=/scratch2/.../hf-cache:/opt/hf-cache,/scratch2/.../inputs:/inputs,/scratch2/.../outputs:/outputs \
                    --container-writable \
                    python /scratch2/.../scripts/run_deepforest.py'
```
