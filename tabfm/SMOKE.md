# tabfm — Compute2 smoke test

The smallest run that proves the image works on an H100: **a real fit and
predict on a small table**, checking that predictions come back with the right
shape and that probabilities sum to one. The build-time smoke test already
covers imports and that both estimator classes resolve; what it cannot cover is
weight loading, which needs the network and 13.1 GB of cache.

## 0. One-time: import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+tabfm+v1.sqsh \
    'docker://ghcr.io#bradleylab/tabfm:v1'
```

`enroot import` can exit 0 after its `mksquashfs` child is OOM-killed, so check
the artifact rather than the exit status:

```bash
file -b bradleylab+tabfm+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. One-time: stage the weights

Compute nodes have no outbound network, so the cache is populated from a login
node into the Storage3 HF cache. 13.1 GB, both checkpoints.

```bash
HF_HOME=/storage3/fs1/alexander.s.bradley/Active/hf-cache \
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/tabfm-1.0.0-pytorch')
"
```

The weights stay on Storage3 and are never copied into an image or a published
artifact — the license forbids redistributing them.

## 2. The test

`HF_HOME` is set inside the container command rather than via `--export`,
because the image's own `ENV` wins over `--export` and a forwarded cache path
would send the offline loader somewhere the weights are not.

```bash
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=48G --time=00:30:00 \
       -J tabfm-smoke -o tabfm-smoke-%j.out --wrap='
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+tabfm+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley/Active/hf-cache:/hf \
     --container-workdir=/tmp \
     bash -c "export HF_HOME=/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1; python - <<PY
import numpy as np, torch
from sklearn.datasets import make_classification
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

print(\"gpu:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")

X, y = make_classification(n_samples=200, n_features=8, n_informative=5,
                           n_classes=3, random_state=42)
Xtr, ytr, Xte, yte = X[:150], y[:150], X[150:], y[150:]

model = tabfm_v1_0_0.load(model_type=\"classification\")
clf = TabFMClassifier(model=model)
clf.fit(Xtr, ytr)
probs = clf.predict_proba(Xte)

print(\"probs:\", probs.shape, probs.dtype)
assert probs.shape == (50, 3), probs.shape
assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), probs.sum(axis=1)[:5]
acc = (probs.argmax(axis=1) == yte).mean()
print(f\"accuracy on a separable synthetic set: {acc:.3f}\")
assert acc > 0.5, \"no better than chance on a separable problem\"
print(\"SMOKE OK\")
PY"'
```

The accuracy floor is a sanity check, not a benchmark: `make_classification`
with five informative features is separable, so a model that loaded correctly
should clear 0.5 comfortably. A model that loaded garbage weights will not.
