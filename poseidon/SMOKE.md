# poseidon — Compute2 smoke test

The smallest run that proves the image works on an H100: **a real forward pass
through Poseidon-T**, checking that a solution field comes back at the shape the
model's own configuration implies. The build-time test already loads the baked
checkpoint; what it cannot cover is an sm_90 forward pass, which is exactly the
thing that failed silently for `crossearth` on the cu117 wheel.

## 0. One-time: import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+poseidon+v1.sqsh \
    'docker://ghcr.io#bradleylab/poseidon:v1'
```

`enroot import` can exit 0 after its `mksquashfs` child is OOM-killed, so check
the artifact rather than the exit status:

```bash
file -b bradleylab+poseidon+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. The test

Poseidon-T is baked, so this needs no staged weights and no network.

```bash
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=48G --time=00:30:00 \
       -J poseidon-smoke -o poseidon-smoke-%j.out --wrap='
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+poseidon+v1.sqsh \
     --container-workdir=/tmp \
     bash -c "export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1; python - <<PY
import torch
from scOT.model import ScOT

assert torch.cuda.is_available(), \"no GPU visible to the job\"
print(\"gpu:\", torch.cuda.get_device_name(0), \"| capability:\", torch.cuda.get_device_capability(0))

model = ScOT.from_pretrained(\"camlab-ethz/Poseidon-T\").cuda().eval()
cfg = model.config
c_in = cfg.num_channels
size = cfg.image_size
print(\"config: num_channels\", c_in, \"| image_size\", size)

x = torch.randn(1, c_in, size, size, device=\"cuda\")
t = torch.tensor([0.5], device=\"cuda\")
with torch.no_grad():
    out = model(pixel_values=x, time=t)

field = out.output if hasattr(out, \"output\") else out[0]
print(\"output:\", tuple(field.shape), field.dtype)
assert field.shape[0] == 1 and field.shape[-1] == size, field.shape
assert torch.isfinite(field).all(), \"non-finite values in the solution field\"
print(\"SMOKE OK\")
PY"'
```

Random input is deliberate: the test asks whether the graph runs on sm_90 and
returns a finite field of the right shape, not whether the solution means
anything. A model that loaded but could not execute a Hopper kernel fails here.

If the output attribute name differs from `output` on this release, the first
run will say so — read the printed `ScOTOutput` fields and adjust rather than
guessing.
