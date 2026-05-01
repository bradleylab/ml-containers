# forainet

ForAINet (Xiang et al., ETH PRS) — semantic + instance panoptic
segmentation of airborne lidar point clouds. PointGroup-style
architecture trained on FOR-Instance.

> **EXPERIMENTAL.** Upstream ships PyTorch 1.9 / CUDA 11.1 (no sm_90
> support). This container ports the stack to PyTorch 2.2.2 / CUDA
> 12.1 by reusing the H100-proven recipe from
> `segment-any-tree-h100`. The combination of (a) ForAINet code
> targeting torch 1.9, (b) running it on torch 2.2.2, and (c)
> MinkowskiEngine on H100 means runtime breakage is plausible. First
> end-to-end run is the test.

## Image tag

`ghcr.io/bradleylab/forainet:v1` (also `:latest`)

## Stack

- `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel` base
- MinkowskiEngine from `CiSong10/MinkowskiEngine@cuda12-installation`
  (same fork as SAT; native sm_90)
- torchsparse 1.4.0 with PyTorch-2.2 `.type()` → `.scalar_type()`
  patches
- torch-geometric 2.5.3 + PyG ecosystem (cu121 wheels)
- torch-points-kernels with CUDA 12 + numpy<2 patches
- torch-points3d compat patches (`data.keys` method form,
  region_grow device alignment)
- hydra-core 1.0.7 + omegaconf 2.0.6 (matches the legacy
  torch-points3d code path)

## Weights

NOT baked into the image. The authors distribute
`PointGroup-PAPER.pt` from Dropbox under no clear license. Fetch
once into a persistent host directory and bind-mount at runtime:

```bash
# One-time fetch on the target host (Compute2 scratch, pliny, etc.):
mkdir -p /scratch2/fs1/alexander.s.bradley/forainet_weights
cd /scratch2/fs1/alexander.s.bradley/forainet_weights
wget 'https://www.dropbox.com/scl/fi/mv4nxe60cco86fd2u9f3z/PointGroup-PAPER.pt?rlkey=ua6093kehk0youpo8g3a6g0nm&dl=1' \
     -O PointGroup-PAPER.pt
```

## Run on Compute2

ForAINet has two inference paths. For UAV-airborne tiles that fit on
one H100 in one pass:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-gpu \
       --gpus=1 --mem=64G --time=04:00:00 \
       --wrap='srun --container-image=/storage1/.../bradleylab+forainet+latest.sqsh \
                    --container-mounts=/scratch2/.../forainet_weights:/opt/ForAINet/PointCloudSegmentation/model_file,/scratch2/.../inputs:/opt/ForAINet/PointCloudSegmentation/eval_data,/scratch2/.../outputs:/opt/ForAINet/PointCloudSegmentation/eval_out \
                    --container-writable \
                    bash -lc "cd /opt/ForAINet/PointCloudSegmentation && python eval.py"'
```

For large airborne tiles that need split → infer-per-tile → merge:

```bash
# (same SBATCH block as above, replacing the inner command:)
bash -lc 'cd /opt/ForAINet/PointCloudSegmentation && bash large_PC_predict.sh'
```

`eval.yaml` and `exampleeval.yaml` (for the large-tile path) live
inside the container at
`/opt/ForAINet/PointCloudSegmentation/conf/`. To override paths or
parameters, mount your own config over the upstream one or edit the
file in a writable container layer before running.

## Why we built this

The lab decided 2026-04-06 NOT to build a ForAINet container, on
the rationale that:

- ForAINet's training set has higher point density (~75 pts/m²)
  than Tyson UAV (~28 pts/m²) — expected to underperform at our
  density.
- Identical CUDA rebuild effort to SAT, with no expected quality
  improvement.
- The over-merging seen on SAT v1 turned out to be a hardcoded
  `block_merge_th=0.1` bug, fixed in SAT v2 — not a fundamental
  model limitation.

Reversed 2026-05-01 to validate that conclusion empirically on
Tyson data. Treat this image as a baseline-comparison tool, not a
production segmenter.

## Limitations

- Trained on European airborne lidar at higher density than Tyson —
  inference at lower density is out-of-distribution.
- No license on the upstream repo root or the Dropbox-hosted
  weights — for in-lab use only; do not redistribute publicly.
- The torch 2.2 port has not been validated end-to-end yet. If
  inference fails, look first at the PyG 2.x compatibility layer
  (`data.keys` patches), the torch_points_kernels device
  alignment, and the hydra Python-3.10 patch — all are cribbed
  from the SAT v1 fixes and may need tuning for ForAINet's slightly
  different code paths.
