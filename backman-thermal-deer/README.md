# backman-thermal-deer

Runtime container for the Backman et al. 2025 thermal animal detection
model. Frame-level detection in thermal video using a recurrent (LSTM)
ONNX model with built-in `frameSkip`.

CPU-only by default; the model is small enough to run comfortably on
a `general-cpu` slot.

## Image tag

`ghcr.io/bradleylab/backman-thermal-deer:v1` (also `:latest`)

## What ships in the image

- Python 3.11
- `onnxruntime` for ONNX inference
- `opencv-python-headless` + `ffmpeg` for video I/O
- `numpy`, `pandas`, `tqdm`

## What does NOT ship

The `inferenceExample/` directory containing `model.onnx` and
`generateVideoPredictions.py` is distributed by the paper authors and
is **not baked into the image**. Bind-mount it at runtime and point
the wrapper at it.

## Run pattern (Compute2)

```bash
sbatch --container-image=/storage1/.../bradleylab+backman-thermal-deer+latest.sqsh \
       --container-mounts=/scratch2/...:/scratch2/...,/storage1/...:/storage1/... \
       --no-container-entrypoint \
       run_backman.sh
```

Where `run_backman.sh` invokes the upstream script with paths to
mounted volumes:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

MODEL_DIR=/scratch2/.../backman/inferenceExample
VIDEO_DIR=/scratch2/.../mp4
OUT_DIR=/scratch2/.../predictions

python "${MODEL_DIR}/generateVideoPredictions.py" \
    --videoCSVFile "${OUT_DIR}/videos.csv" \
    --saveDir "${OUT_DIR}" \
    --onnxFile "${MODEL_DIR}/model.onnx" \
    --imageWidth 640 --imageHeight 512 \
    --frameSkip 5 \
    --display False
```

The Tyson wrapper at
`tyson-deer-survey/tyson-thermal-deer-survey/scripts/run_backman_inference.py`
generates the `videos.csv` from the project's segment list and invokes
`generateVideoPredictions.py` automatically. The wrapper itself is the
canonical entry point — the manual `python` invocation above is the
fallback.

## Outputs

- `predictions/{segment_label}.csv` — per-frame detection rows with
  `frame, class, x, y, w, h, score` (class 0 = arboreal, 1 = ground)
- `video/{segment_label}.mp4` — annotated visualization video

## Inputs

- 640×512 thermal MP4 segments (DJI XT2 native resolution).
- Videos must be processed as complete sequences — do NOT pre-sample
  frames. The model uses LSTM hidden state for temporal context, and
  `frameSkip=5` is applied internally.

## Provenance

- Upstream model + script: distributed by Backman et al. as the
  `inferenceExample/` directory accompanying the 2025 paper. Trained
  zero-shot on Australian survey data.
- First end-to-end run at Tyson: 2026-03-23 flights (6 flights, 30
  segments, 128,638 frames, 21 deer detected after filtering).
