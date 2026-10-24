# seislm

SeisLM (Liu, Münchmeyer, Laurenti, Marone, de Hoop and Dokmanić, 2024) is a
self-supervised foundation model for seismic waveforms. It follows Wav2Vec2:
two strided convolutions turn a single-station, 3-component record at 100 Hz
into a feature sequence, a convolutional position embedding and pre-norm
transformer blocks read it, and pretraining asks the transformer to pick out
the quantized features of masked time steps from distractors. The pretraining
data are the unlabeled training folds of eight SeisBench datasets: ETHZ,
INSTANCE, Iquique, STEAD, GEOFON, MLAAPDE, PNW and OBST2024. Fine-tuned with a
small convolutional head, it is evaluated on event detection, phase
identification and onset regression (ETHZ, GEOFON, STEAD) and on
foreshock-aftershock classification at station NRCA in the 2016 Norcia
sequence; the paper reports its advantage over PhaseNet as largest when
labeled data are scarce.

Paper: [arXiv:2410.15765](https://arxiv.org/abs/2410.15765) ·
Code: [liutianlin0121/seisLM](https://github.com/liutianlin0121/seisLM) @ `ae1d27a` ·
Weights: Google Drive folder linked from upstream's README, also published as
[ltlltl2006/seislm-base](https://huggingface.co/ltlltl2006/seislm-base) and
[ltlltl2006/seislm-large](https://huggingface.co/ltlltl2006/seislm-large) ·
License: MIT (code and weights)

Pull: `ghcr.io/bradleylab/seislm:v1`

## License and the weights

Upstream added an MIT LICENSE (copyright 2024, Tianlin Liu) on 2026-10-15, at
commit `e339c7b`, and its `docs/huggingface.md` states that the author has
confirmed the pretrained weights use the MIT license. The code this image pins,
`ae1d27a`, predates that commit, so the build copies the LICENSE text from
`e339c7b` to `/opt/seislm/LICENSE` and checks its SHA-256.

The weights are not in the image. Upstream's fine-tuning code loads the original
Lightning checkpoints, so the image uses those, mounted at `/weights` at run
time. Upstream's Hugging Face bundles hold the same weights as safetensors for
its newer inference API, which this image does not carry.

### Lab copies of the checkpoints

The lab's copy is on the NAS at `/mnt/nas/datasets/ml_model_weights/seislm/`;
the Compute2 working copy is
`/storage3/fs1/alexander.s.bradley/Active/seislm/weights/`, with the same
layout. Upstream's `docs/installation.md` lists the same SHA-256 values.

| Model | Path under the weights directory | Size | SHA-256 |
|---|---|---|---|
| SeisLM-base | `pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt` | 136,471,912 bytes | `6e97c53c562e9ba6bc96df85d8d208b89ff84e80a25468825f29bb5bace5eacb` |
| SeisLM-large | `pretrained_seislm_large/checkpoints/epoch=39-step=701720.ckpt` | 1,088,481,576 bytes | `db44b1a46512de2cdf5c310c7ccfdac0ef04931cc89b7768fad41c978266df4b` |

The paths are the ones upstream's notebooks use. Both scripts in the image
check the size and SHA-256 before loading a file, because a Lightning
checkpoint is a pickle and loading it runs code.

## The two models

| | SeisLM-base | SeisLM-large |
|---|---|---|
| Transformer blocks | 6 | 12 |
| Hidden size | 240 | 768 |
| Parameters (counted from the checkpoint) | 11,359,568 | 90,683,648 |
| Parameters (paper) | 11.4 M | 90.7 M |
| Upstream pretraining config | `pretrain_config_std_norm_single_ax_8_datasets_sample_pick_false.json` | `pretrain_config_std_norm_single_ax_8_datasets_32bit_scaleup_samp_false.json` |

Both share the convolutional encoder (two layers, 256 channels, kernel 3,
stride 2), two codebooks of 320 code vectors, 12 attention heads and a
3072-unit feed-forward layer. The paper finds the two close on event detection
and SeisLM-large ahead on phase identification and on the foreshock-aftershock
task.

## Input contract

These follow upstream's pretraining pipeline and demo notebook:

- Three components ordered Z, N, E, channels first: shape `(batch, 3, samples)`.
- 100 Hz; resample other rates first, as the paper does.
- Each channel demeaned and divided by its standard deviation, per window.
- Pretraining and the demo use 3001-sample windows (30 s). The encoder reduces
  3001 samples to 749 feature frames.

## Using the encoder from Python

Upstream's source is on `PYTHONPATH` (`/opt/seislm`). Loading follows upstream's
demo notebook, with `weights_only=False` because the saved hyperparameters are
an `ml_collections.ConfigDict` that torch's restricted unpickler refuses; check
the SHA-256 first, as `seislm_checkpoints.verified_checkpoint` does.

```python
from pathlib import Path

import torch
from seislm_checkpoints import verified_checkpoint
from seisLM.model.foundation.pretrained_models import LitMultiDimWav2Vec2

ckpt = verified_checkpoint(Path("/weights"), "base")
model = LitMultiDimWav2Vec2.load_from_checkpoint(ckpt, map_location="cuda", weights_only=False).model.eval()
with torch.no_grad():
    out = model.wav2vec2(input_values=x.cuda(), output_hidden_states=True)  # x: (batch, 3, 3001)
features = out.last_hidden_state  # (batch, 749, 240) for base, (batch, 749, 768) for large
```

The paper's trace embedding for its t-SNE figure is the time average of the
last layer's output. For fine-tuning, upstream's task classes in
`seisLM.model.task_specific` (phase picking and foreshock-aftershock
classification) load a checkpoint through `pretrained_ckpt_path` in their
configs under `seisLM/configs/`.

Two constraints come from upstream's code. `seisLM.utils.project_path` runs
`git rev-parse --show-toplevel` from the working directory when imported and
reads data from `<root>/data`; the data pipelines and run scripts import it, so
run them from a git repository laid out that way (`seislm-verify-shock` sets one
up). And upstream's own `load_from_checkpoint` calls pass no `weights_only`, so
under torch 2.14 they need `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in the
environment, set only after the checkpoint is verified.

## Scripts

- `seislm-selftest [--device cpu|cuda] [--weights /weights]` loads both
  checkpoints the way upstream's demo does, runs each on a synthetic 30 s
  window, and checks parameter counts, feature shapes and finite output.
  `--random-init` builds both from the upstream configs instead and needs no
  weights; the image build runs it.
- `seislm-verify-shock` reruns the paper's foreshock-aftershock result on an
  H100: `stage` downloads upstream's NRCA data, `run` fine-tunes and tests one
  model, and `selfcheck` runs the whole pipeline offline on synthetic data,
  which the image build also does. [`SMOKE.md`](SMOKE.md) has the Compute2 jobs.

## Stack

- Base: `python:3.12-slim-bookworm`, with `git` for upstream's `project_path`.
- `torch==2.14.0+cu130` from the PyTorch CUDA 13.0 index. Upstream's README
  names Python 3.9 and `pytorch-cuda=11.8` but pins nothing, and both
  checkpoints load and run on this stack with Lightning 2.6.
- `torchtune==0.6.1` for the `RMSNorm` class upstream imports at module level
  (neither checkpoint uses it), with `torchao==0.17.0`, the newest release
  torchtune 0.6.1 can import.
- Lightning, ml-collections, SeisBench, einops, scikit-learn, torchmetrics,
  GitPython, pandas 2.x and gdown. [`constraints.txt`](constraints.txt) freezes
  the full resolved set (114 pins, resolved 2026-10-02) and records the command.
- Upstream at `ae1d27a` in `/opt/seislm` (the last commit, 2024-10-15), from
  the GitHub tarball; the shared checkpoint module in `/opt/seislm-lab`.

The build test is offline: torch is the CUDA 13.0 build, upstream's modules
import, `seislm-selftest --random-init` passes, and
`seislm-verify-shock selfcheck` fine-tunes a random SeisLM-base for one epoch
on synthetic NRCA-format data and evaluates it.

## Related images

- `seisbench` runs PhaseNet and EQTransformer from the SeisBench model zoo for
  picking out of the box. SeisLM is not a ready picker: its released weights are
  the pretrained encoder, and picking or detection needs fine-tuning on labeled
  data first. SeisBench's datasets are what SeisLM was pretrained and evaluated
  on.
- `seist` covers polarity, magnitude, back-azimuth and distance with supervised
  checkpoints. The paper's case for SeisLM is tasks with few labels.

## License

MIT, code and weights. See [License and the weights](#license-and-the-weights).
