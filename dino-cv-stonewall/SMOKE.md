# dino-cv-stonewall — smoke test

## 1. Build (GitHub Actions)

`stonewall-selftest` runs on the CPU with the network disabled: all three
two-view models built as upstream builds them, random weights saved in
upstream's checkpoint format, and `stonewall-predict` run on synthetic tiles.
Record the printed parameter counts here after the first build; the paper's
two-view totals are 138 M (WideResNet-50-2), 46 M (ResNet-50) and 42 M
(ViT-S/16).

| Backbone | Parameters (this image) | Paper |
|---|---|---|
| wrn50 | | 138 M |
| rn50 | | 46 M |
| vitsmall16 | | 42 M |

## 2. Compute2, H100, with the lab's checkpoints

Needs the checkpoints staged as in `README.md` (lab note). Load each backbone's
six fold checkpoints strictly, then predict on one held-out area of upstream's
BudjBimArea dataset with the checkpoint that held that area out, and compute
pixel F1 and IoU per tile against the labels. The comparison is the paper's
per-area mIoU (Fig. 6).

```bash
srun -A compute2-alexander.s.bradley \
  -p general-gpu \
  --gpus=1 \
  --container-image=ghcr.io#bradleylab/dino-cv-stonewall:v1 \
  --container-mounts=/storage3/fs1/alexander.s.bradley/Active/dino-cv-stonewall/weights:/weights,/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
  stonewall-predict --tiles <area_dir> --out <out_dir> --backbone wrn50 --area area3
```

`stonewall-predict` writes masks only; scoring them against the labels is a separate step.
