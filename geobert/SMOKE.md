# geobert — smoke test

## 1. Build (GitHub Actions)

The build ends with `geobert-selftest`, run on the CPU with the network
disabled. It saves a tiny randomly initialized BERT token classifier with
GeoBERT's nine labels and runs `geobert-ner` on three lines: a short passage, a
blank line and a 600-word passage. It checks that the output has records for
lines 1 and 3 only, that every entity label is one of GeoBERT's four, and that
line 3 is marked `"skipped"`. The real weights are not in the image, so the
build cannot test them.

## 2. The real weights

`geobert-ner` with `--cache-dir` downloads GeoBERT at the pinned revision and
checks `tf_model.h5` against its SHA-256 before loading it. Save the example
sentence from the model card as one line in `card.txt`; it should return
these eight spans:

| Label | Text | Characters | Score |
|---|---|---|---|
| GeoLoc | North America | 3–16 | 0.9113 |
| GeoMeth | pore pressure | 108–121 | 0.9965 |
| GeoPetro | drainage | 139–147 | 0.9994 |
| GeoMeth | resistivity | 211–222 | 0.9997 |
| GeoPetro | Sandstone | 288–297 | 0.9999 |
| GeoPetro | Sediment | 298–306 | 0.9999 |
| GeoMeth | seismic | 399–406 | 0.9998 |
| GeoTime | Paleozoic | 445–454 | 0.9999 |

These come from the recipe's wrapper and pinned versions (TensorFlow 2.15.1,
Keras 2.15.0, transformers 4.44.2) installed on an Apple-silicon Mac, on the
CPU, before the image was built.

## 3. Compute2, H100

Import, then run the same sentence on a GPU node with the lab's Hugging Face
cache, from a folder under Storage3 that holds `card.txt`:

```bash
srun -A compute2-alexander.s.bradley \
  -p general-gpu \
  --gpus=1 \
  --container-image=ghcr.io#bradleylab/geobert:v1 \
  --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
  --container-workdir="$PWD" \
  bash -c 'python -c "import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))" && geobert-ner card.txt --out card.jsonl --cache-dir /storage3/fs1/alexander.s.bradley/Active/hf-cache'
```

Check that:

- TensorFlow lists one GPU.
- The hash check passes and the eight spans match the table in section 2.
  Scores may differ in the last digit between CPU and GPU.
