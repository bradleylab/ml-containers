# n2n4m — Compute2 smoke test

The smallest job that proves the image landed on Compute2 intact. It
touches no CRISM data, no Zenodo data, and no network: it loads the baked
weights and scaler and runs one forward pass. If this passes, the image is
good and any later failure is about *your* data or paths, not the
container.

## 0. One-time: import the image

On a login node (this is a download, not compute — it is the one thing
here that does not need a job):

```bash
enroot import \
  -o /storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+n2n4m+v1.sqsh \
  'docker://ghcr.io#bradleylab/n2n4m:v1'
```

## 1. Input

None. No mounts, no files, no arguments. That is the point — everything
the test needs ships inside the image.

## 2. The job

CPU partition, one core, five minutes:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=1 \
       --mem=4G \
       --time=00:05:00 \
       -J n2n4m-smoke \
       -o n2n4m-smoke-%j.out \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+n2n4m+v1.sqsh \
         bash -lc "export PYTHONNOUSERSITE=1; python -c \"
import importlib.metadata as im, warnings, numpy as np, torch
warnings.simplefilter(\\\"ignore\\\")
from n2n4m.n2n4m_denoise import (DEFAULT_MODEL_FILEPATH, DEFAULT_SCALER_FILEPATH,
                                 instantiate_default_model, load_scaler, create_dataloader)
from n2n4m.model_functions import predict, check_available_device
assert DEFAULT_MODEL_FILEPATH.is_file() and DEFAULT_SCALER_FILEPATH.is_file()
model, scaler = instantiate_default_model(), load_scaler()
out = predict(model, create_dataloader(scaler.transform(np.full((4, 350), 0.2)), batch_size=4),
              torch.device(check_available_device()))
assert tuple(out.shape) == (4, 350) and bool(torch.isfinite(out).all())
print(\\\"n2n4m\\\", im.version(\\\"n2n4m\\\"), \\\"| torch\\\", torch.__version__,
      \\\"| device\\\", check_available_device())
print(\\\"params\\\", sum(p.numel() for p in model.parameters()))
print(\\\"SMOKE OK\\\")
\""'
```

If quoting that through `--wrap` is more trouble than it is worth — and it
often is — put the Python in a file on scratch and call it:

```bash
srun -A compute2-alexander.s.bradley -p general-cpu --cpus-per-task=1 \
     --mem=4G --time=00:05:00 \
     --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+n2n4m+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/scripts:/scripts \
     bash -lc 'export PYTHONNOUSERSITE=1; python /scripts/n2n4m_smoke.py'
```

## 3. Output that proves success

```
n2n4m 0.0.2 | torch 2.9.1+cu128 | device cpu
params 1092945
SMOKE OK
```

The pass criterion is the `SMOKE OK` line and a zero exit status. The two
lines above it are what make a failure diagnosable:

- **`params 1092945`** — the checkpoint loaded into a matching
  architecture. A shape mismatch raises before this prints, so seeing the
  exact count means the baked weights are the ones the code expects.
- **`torch 2.9.1+cu128`** — the pinned CUDA build, not a CPU wheel and not
  something a stray `pip install --user` shadowed in. If the `+cu128`
  suffix is missing, `PYTHONNOUSERSITE=1` was dropped or the wrong image
  was mounted.
- **`device cpu`** — expected on `general-cpu`. Not a failure.

## 4. Expected runtime

Seconds of compute. The four-spectrum forward pass is negligible; wall
time is the squashfs mount plus interpreter start. Queue wait dominates
everything and is not predictable — request five minutes and treat
anything that runs longer than about a minute of *compute* as a problem
with the mount, not the model.

## 5. Optional: confirm the GPU path

Same test on `general-gpu`. The only difference in the output should be
the device.

```bash
srun -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
     --cpus-per-task=1 --mem=8G --time=00:05:00 \
     --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+n2n4m+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/scripts:/scripts \
     bash -lc 'export PYTHONNOUSERSITE=1; python /scripts/n2n4m_smoke.py'
```

Expect `device cuda`. If it still says `cpu`, the GPU was not visible to
the container — check `--gpus=1` reached `srun` — and note that this is
a performance problem, not a correctness one. The model produces the same
answer either way.

## 6. If it fails

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: n2n4m` | Wrong `.sqsh`, or the import never completed. Re-run the `enroot import`. |
| `FileNotFoundError` on `trained_model_weights.pt` | Upstream stopped shipping the weights as package data. This should be impossible — the build-time smoke test asserts it — so suspect a hand-built image instead. |
| Version numbers that do not match this README | `$HOME` site-packages shadowing the container's. `PYTHONNOUSERSITE=1` was not exported. |
| `InconsistentVersionWarning` about `RobustScaler` | Expected, and suppressed above. The scaler was pickled under scikit-learn 1.2.2. See the README caveat. |
