#!/usr/bin/env python
"""Score the image's semseg path (forpt_lib, as forpt-semseg uses it) on the FOR-instanceV2 test split.

    python forpt_lib_miou.py <forinstancev2_sem>/test

Each scene folder holds coord.npy and segment.npy (0 ground, 1 wood, 2 leaf,
-1 ignored). Intersections and unions are summed over all scenes before the
ratio, as upstream's SemSegSimpleTester does, so the numbers are comparable with
its log and with Table 1 of the paper. Not copied into the image; run it with
this folder mounted.
"""

import sys
from pathlib import Path

import forpt_lib as fl
import numpy as np
import torch

IGNORE_INDEX = -1


def main() -> None:
    test_dir = Path(sys.argv[1])
    device = fl.require_cuda()
    model = fl.build_task_model(fl.SEMSEG, device)
    transform = fl.test_transform(fl.SEMSEG)
    names = fl.class_names(fl.SEMSEG)
    intersection = np.zeros(len(names))
    union = np.zeros(len(names))
    target = np.zeros(len(names))
    scenes = sorted(p for p in test_dir.iterdir() if (p / "coord.npy").exists())
    for scene in scenes:
        # Upstream's dataset loads coordinates as float32.
        coord = np.load(scene / "coord.npy").astype(np.float32)
        truth = np.load(scene / "segment.npy").reshape(-1)
        fl.seed_everything()
        batch = fl.model_input(coord, transform, device)
        with torch.no_grad():
            pred = model(batch)["seg_logits"].argmax(1)[batch["inverse"]].cpu().numpy()
        valid = truth != IGNORE_INDEX
        for c in range(len(names)):
            p, t = (pred == c) & valid, (truth == c) & valid
            intersection[c] += (p & t).sum()
            union[c] += (p | t).sum()
            target[c] += t.sum()
        print(f"{scene.name}: {len(coord):,} points")
    iou = intersection / union
    print(f"{len(scenes)} scenes: mIoU {100 * iou.mean():.1f}, allAcc {100 * intersection.sum() / target.sum():.1f}")
    print("per-class IoU:", ", ".join(f"{n} {100 * v:.1f}" for n, v in zip(names, iou)))


if __name__ == "__main__":
    main()
