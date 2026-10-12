# hunting-pits — smoke test

## 1. Build (GitHub Actions)

The build ends with `hunting-pits-selftest`, run on the CPU with the network
disabled: upstream's inference script on upstream's test chip, compared with
the polygons upstream committed for that chip. Record the printed polygon
counts and IoU here after the first build.

| Build | This image | Upstream's committed prediction | IoU |
|---|---|---|---|
| | | | |

## 2. Compute2, H100

Import, then run the same self-test on a GPU node, where TensorFlow should
place the model on the GPU:

```bash
srun -A compute2-alexander.s.bradley \
  -p general-gpu \
  --gpus=1 \
  --container-image=ghcr.io#bradleylab/hunting-pits:v1 \
  bash -c 'python -c "import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))" && hunting-pits-selftest'
```

Check that:

- TensorFlow lists one GPU. If it lists none, the TensorFlow 2.15 CUDA wheels
  do not run on the H100 driver and the image is CPU-only until that is fixed.
- The self-test prints the same figures as the build. The image filesystem is
  read-only under enroot, so this also proves that WhiteboxTools and the
  `/workspace/temp_inference` link work without writing into the image.

## 3. A real tile (optional)

Upstream's test labels are for one chip only. A fuller check needs a 0.5 m DEM
over known, mapped pits; the paper's data are archived at
[doi:10.5878/en98-1b29](https://doi.org/10.5878/en98-1b29).
