# poseidon

Foundation models for solving partial differential equations.

Herde, Raonić, Rohner et al., [Poseidon: Efficient Foundation Models for
PDEs](https://arxiv.org/abs/2405.19101) (NeurIPS 2024), ETH Zurich CAMLab.
Upstream: [camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon).
Weights: [camlab-ethz on the Hub](https://huggingface.co/camlab-ethz).

## What this is

A scalable Operator Transformer (scOT) with time-conditioned layer
normalization, pretrained across families of PDEs — compressible Euler,
Navier-Stokes, wave, Poisson, Helmholtz — and designed to be finetuned onto a
downstream operator using far fewer samples than training from scratch. Given
an input state, and a time for time-dependent problems, it returns the solved
field rather than a scalar.

| Checkpoint | Weights |
|---|---|
| `Poseidon-T` | 83 MB, **baked into this image** |
| `Poseidon-B` | 1.26 GB, runtime fetch |
| `Poseidon-L` | 5.03 GB, runtime fetch |

## The license situation, first

**The weights are CC-BY-NC-4.0** — non-commercial, ungated, the same footing as
`sonata`, `concerto` and `utonia` here.

**The code carries no license at all.** There is no LICENSE file anywhere in the
upstream repository (whole tree checked at `b8fa28f`, 2026-09-21) and the README
states no terms. Absent a license, no permission to use, modify or redistribute
has formally been granted, and this image necessarily redistributes the code
because scOT must be installed to run.

Against that, the project page states under "Usage": *"We encourage using our
pretrained models on your own datasets. To that end, you can directly plug your
dataset into our code and then finetune using our scripts."* The paper describes
the models and datasets as open sourced, and the weights carry a deliberate
license tag. The missing file reads as an oversight rather than a withholding.
That is a judgment, not a license, and it is why this image exists.

Two consequences stand until upstream adds a license:

- **Non-commercial only**, from the weights.
- **This image is not listed on the public model catalog**, the same treatment
  `forainet` and `backman-thermal-deer` receive for the same reason.

If upstream adds a LICENSE, revisit both the label and that decision.

## Stack, and the H100 trap in it

scOT pins `torch == 2.0.1`, whose default wheel is cu117 — which predates
sm_90. `crossearth` shipped exactly that way and **hung** on an H100 rather
than failing. CUDA 11.8 is the first release with Hopper support and
`torch 2.0.1+cu118` exists, so this image honors the upstream pin and still
reaches the GPU.

- Base `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`, Python 3.10
- `torch 2.0.1+cu118` / `torchvision 0.15.2+cu118` from the cu118 index
- `transformers==4.29.2`, `accelerate==0.31.0`, `wandb==0.14.2` (disabled at
  runtime), `numpy<2` for the torch 2.0 ABI
- scOT installed `--no-deps` from the pinned commit: its bare `torch == 2.0.1`
  requirement would otherwise pull the cu117 wheel over the cu118 one

The build asserts the CUDA version rather than `torch.cuda.get_arch_list()`,
which is empty on a driver-less CI runner.

## Usage

```python
from scOT.model import ScOT

model = ScOT.from_pretrained("camlab-ethz/Poseidon-T")   # or -B, -L
```

`Poseidon-B` and `-L` download on first use; point `HF_HOME` at a mounted cache
so compute nodes read them offline afterwards. Upstream also ships a CLI:

```bash
python -m scOT.inference --help
```

Finetuning onto your own dataset means implementing a dataset class and
registering it in `scOT/problems/base.py`, then running upstream's training
script — see their README. That is a project, not a single run.

Pull: `ghcr.io/bradleylab/poseidon:v1`

## Scope

The pretrained models learned on PDEgym, a synthetic benchmark collection.
Solute transport, groundwater flow and heat flow all sit in the families
covered, but applying these checkpoints to real lab data means finetuning.
Nothing here has been run on lab data.
