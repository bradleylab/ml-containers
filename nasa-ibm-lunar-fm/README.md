# nasa-ibm-lunar-fm

The NASA-IBM Lunar Foundation Model (NASA-IBM LFM), released 2026-09-10 by NASA
IMPACT and IBM Research: a ViT-B encoder–decoder pretrained from scratch on
SomBench, about two million co-registered lunar tile bundles spanning 11
modalities at two scales, LROC NAC (~1 m/px) and LROC WAC (~100 m/px). It adapts
TerraMind's masked-token recipe to the Moon in two ways: per-tile illumination
geometry is an explicit encoder input, and NAC and WAC tiles train in one loop,
so one set of weights covers a 100× range in resolution. Fine-tuning and
inference run through TerraTorch, as for `terramind` and `prithvi-eo`.

Paper: [arXiv:2609.13283](https://arxiv.org/abs/2609.13283) ·
Code: [NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model) @ `d54c67a` ·
Weights: [Hugging Face collection](https://huggingface.co/collections/nasa-ibm-ai4science/nasa-ibm-lunar-fm-and-downstream-models-6a96f44c430791be886ca331)

Pull: `ghcr.io/bradleylab/nasa-ibm-lunar-fm:v1`

## Released checkpoints

Upstream released the backbone and three fine-tuned tasks. The test scores are
upstream's, mean ± standard deviation over five seeds, from each model card.

| Task | Checkpoint | Data | Published test score |
|---|---|---|---|
| Crater detection, WAC (~100 m/px, Robbins catalog) | `WAC_ni_lfm_ps8_lora_s46.ckpt`, LoRA | 800 / 100 / 100 tiles | mAP 0.2581 ± 0.0017 |
| Crater detection, NAC (~1 m/px, hand-labeled) | `NAC_ni_lfm_ps8_s44.ckpt`, LoRA | 645 / 59 / 62 tiles | mAP 0.1543 ± 0.0098 |
| Irregular mare patch segmentation, NAC | `ni_lfm_ps8_frozen_s44.ckpt`, frozen encoder + UNet decoder | 100 / 20 / 10 tiles | IoU₁ 0.5709 ± 0.0114, F1₁ 0.7268 ± 0.0093 |
| Polar ice prospectivity, 240 m/px | `ni_lfm_ps8_all_modalities_s42.ckpt`, full fine-tune | 108 / 23 / 25 patches | RMSE 0.0293 ± 0.0013, R² 0.9884 ± 0.0010 |

**The ice-prospectivity target is a model, not a measurement.** It regresses the
knowledge-driven fuzzy-overlay prospectivity map of Coyan et al. (2025), built
from eight thermophysical, illumination and terrain layers. Its output is not
measured ice, not a resource estimate, and not evidence of ice at any location;
upstream states this, and it should travel with any number this model produces.

**No geodetic reference frame.** Upstream notes that the backbone recovers local
structure but not absolute values: a generated DTM has the right shape at a
shifted elevation, and generated latitude and longitude can be off by tens of
degrees. The fine-tuned tasks inherit this.

## Weights and data, staged beside the image

Nothing is baked. Everything is Apache-2.0 (weights) or CC-BY-4.0 (SomBench
benchmarks) and ungated, so size is the only reason. The lab's copy is on
Storage3 at `Active/nasa-ibm-lunar-fm`, produced by
[`scripts/stage_weights.py`](scripts/stage_weights.py) with every repository
pinned to a Hugging Face revision:

```
Active/nasa-ibm-lunar-fm/
├── hf/NASA-IBM-Lunar-Foundation-Model/backbone/   checkpoint.pt (2.4 GB) + config.yaml
├── hf/Crater-Detection-NASA-IBM-Lunar-Foundation-Model/
├── hf/IMP-Segmentation-NASA-IBM-Lunar-Foundation-Model/
├── hf/Ice-Prospectivity-NASA-IBM-Lunar-Foundation-Model/
└── downstream_dataset/
    ├── prospectivity_dataset/   imp_dataset/   nac_craters_dataset/   wac_craters_dataset/
```

The nine VQ-VAE tokenizers (10.3 GB) are not staged. They serve any-to-any
generation, which upstream offers as a qualitative probe; fine-tuning and the
downstream tasks do not use them.

## Running it

The upstream configs resolve three paths relative to the working directory:
`terratorch_integration/` (via `custom_modules_path`), `backbone/` and `data/`.
Mount the staged copy at `/weights`, change to a scratch directory and run
`lfm-link`, which creates all three and says which one is missing if any is:

```bash
lfm-link
terratorch test \
  --config /weights/hf/Ice-Prospectivity-NASA-IBM-Lunar-Foundation-Model/config.yaml \
  --ckpt_path /weights/hf/Ice-Prospectivity-NASA-IBM-Lunar-Foundation-Model/ni_lfm_ps8_all_modalities_s42.ckpt
```

`LFM_BACKBONE_DIR` and `LFM_DATA_DIR` override the two staged paths. Every
downstream config needs the backbone directory, including the ice config, whose
checkpoint upstream describes as self-contained: each config still names
`backbone/checkpoint.pt` and `backbone/config.yaml`, and the backbone wrapper
refuses to build without the config. `terratorch fit` with the configs
under `/opt/ni-lfm/terratorch_integration/configs/` reproduces the fine-tunes;
TerraTorch writes its merged `config.yaml` and `tb_logs/` into the working
directory, which is the other reason to run from scratch space.

**The model cards' usage paths do not exist in the repository.** They show
`configs/finetune/<task>.yaml`; the configs are under
`terratorch_integration/configs/<task>/`, and each Hugging Face repo carries the
config its checkpoint was trained with. The crater card also lists
`NAC_ni_lfm_ps9_s44.ckpt`; the file is `NAC_ni_lfm_ps8_s44.ckpt`.

## Stack

- Base: `python:3.12-slim-bookworm`
- `torch==2.12.1` and `torchvision==0.27.1` from the CUDA 12.9 index. Upstream
  requires torch ≥ 2.12 with torchvision 0.27, which fixes the pair; of the CUDA
  12.6, 12.8, 12.9 and 13.0 indexes, 2.12.1 is on 12.9 only, and Compute2's
  driver (580.105.08) supports it.
- The upstream package installed editable at `/opt/ni-lfm`, as upstream's README
  does, with the rest of `requirements.txt` resolved against that torch:
  TerraTorch, Lightning, torchgeo, timm, diffusers, vector-quantize-pytorch.
- [`constraints.txt`](constraints.txt) freezes that resolution — 146 pins,
  resolved 2026-09-27 — because upstream pins only ranges, and floating ranges
  are what broke `terramind` and `prithvi-eo` months after their last good
  build. Regenerating it is a deliberate step, followed by
  `lfm-verify-benchmarks`.

The build test is offline and weight-free: it imports torch and TerraTorch
separately under faulthandler, confirms the three `ni_lfm_v1_*` backbones
register with TerraTorch, and checks the configs are in the source tree.
[`SMOKE.md`](SMOKE.md) is the Compute2 test, which reruns two published
benchmarks on an H100. With the released checkpoints this image gives ice-prospectivity
RMSE 0.0277 and R² 0.9896 (published 0.0293 ± 0.0013 and 0.9884 ± 0.0010) and IMP
IoU₁ 0.5824 (published 0.5709 ± 0.0114).

## Related images

- `lunarfm` is a different model: FDL / Trillium's MultiMAE encoder over orbital
  instrument channels, under PolyForm Strict 1.0.0 and registered in `ml-jobs`
  only. The two share a subject, not code, data or license.
- `crater-detection` detects lunar craters and matches them to a catalog for
  absolute position fixing. This image's crater checkpoints are benchmark
  detectors without catalog matching; neither replaces the other.

## License

Apache-2.0 for the code and for the backbone and all released downstream
checkpoints. The repository's NOTICE credits 4M/MultiMAE (EPFL and Apple) and
other sources, all Apache-2.0 or MIT. The SomBench benchmark datasets are
CC-BY-4.0.
