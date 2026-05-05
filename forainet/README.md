# forainet

ForAINet (Xiang et al., ETH PRS) — semantic + instance panoptic
segmentation of airborne lidar point clouds. PointGroup-style
architecture trained on FOR-Instance.

> **WORKING (with caveats).** Upstream ships PyTorch 1.9 / CUDA 11.1
> (no sm_90 support). This container ports the stack to PyTorch
> 2.2.2 / CUDA 12.1 by reusing the H100-proven recipe from
> `segment-any-tree-h100`. End-to-end inference verified on H100
> after two H100 fixes landed (see "H100 fixes" below). Out-of-
> distribution behavior on closed-canopy airborne lidar at low
> density (<100 pts/m²) is a separate, real concern — see "Known
> caveats."

## H100 fixes

Five issues blocked end-to-end inference on H100 (sm_90); all are
fixed in this image. Two were initial blockers (the model never ran);
three more were latent bugs masked by the multiprocessing deadlock
and only surfaced once the model could complete a forward pass.

1. **`torch-points-kernels` build arch.** Upstream `setup.py`
   hardcodes `-arch=sm_35` in the nvcc flag list, which overrides the
   per-arch `-gencode` flags PyTorch's CUDAExtension builder injects
   from `TORCH_CUDA_ARCH_LIST`. Result: kernels run only on the
   hardcoded arch and fail on H100. Fixed by stripping the hardcoded
   flag at build time and letting `TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0
   8.6 8.9 9.0"` drive the build.
2. **PointGroup3heads clustering deadlock.**
   `torch_points3d/utils/meanshift_cluster.py::cluster_single` spawns
   a `multiprocessing.Pool` with the default `fork` start method.
   Forked workers inherit the parent's already-initialized CUDA
   context and deadlock on the first sklearn-side
   numpy/threadpool touch (the classic PyTorch+fork CUDA hang).
   `PointGroup3heads` (cluster_type=7, the active path for the
   PointGroup-PAPER checkpoint) calls this inside `model.forward()`,
   so the freeze hangs inference end-to-end. Fixed by patching
   `meanshift_cluster.py` to use `multiprocessing.get_context("spawn")`,
   which starts workers with a fresh interpreter.
3. **Numpy 1.24+ deprecated aliases.** ForAINet predates the numpy
   1.20 deprecation of `np.float`, `np.int`, `np.bool`, etc.; numpy
   1.24+ raises `AttributeError` on these. The aliases appear in the
   panoptic tracker, the `treeins_set1` dataset class, and the
   panoptic metrics code. Fixed by walking every `.py` under
   `torch_points3d/` and substituting the builtin (`int`, `float`,
   `bool`, `complex`, `object`, `str`).
4. **PointGroup3heads `_compute_score` device alignment.** The
   `self.input.<attr>` tensors live on CPU, but cluster index tensors
   come back from `cluster_single` on the device that PointGroup ran
   on. Every `self.input.<attr>[cluster]` site needs a device align.
   Fixed via regex covering all `self.input.<attr>[cluster]` forms.
5. **Panoptic tracker device alignment.** Same family: tracker
   buffers (`self._test_area[...].pos`, `originids`, etc.) are CPU
   but cluster indices arrive from the model on GPU. Fixed by forcing
   `cluster.cpu()` at every `[cluster]` indexing site in the tracker.

## Known caveats

- **Density regime.** ForAINet was trained on FOR-Instance (TLS/MLS /
  dense ULS, ~500–10,000 pts/m²). On low-density airborne UAV lidar
  (~100 pts/m²) the model runs end-to-end and produces non-zero output,
  but it can return mega-cluster instances that span multiple trees
  rather than discrete crowns. On the Tyson UAV tile_-10_10 evaluation
  (89 pts/m², closed-canopy oak-hickory) ForAINet returned 110
  instances with median bbox area ~6,500 m² — roughly 80x the median
  bbox area returned by classical methods on the same tile. This is a
  training-distribution issue, not a container bug.
- **`merge_tiles.py` output path.** Upstream's `merge_tiles.py` assumes
  per-tile predictions land under `<hydra_run_dir>/eval/<timestamp>/`,
  but ForAINet's tracker actually writes them flat in `<hydra_run_dir>/`
  as `Instance_Results_forEval_<i>.ply`. `merge_tiles.py` will fail to
  find the inputs. Workaround: bypass `merge_tiles.py` and concatenate
  the per-tile PLYs directly (an example helper is in
  `tyson-forest-linkage/method_comparison/sbatch/recover_forainet_output.sbatch`).
  We may patch this upstream-side later; for now it is a known caveat.

## Image tag

- `ghcr.io/bradleylab/forainet:v2` (also `:latest`) — pretrained
  weights baked at `/opt/ForAINet/PointCloudSegmentation/model_file/PointGroup-PAPER.pt`
- `ghcr.io/bradleylab/forainet:v1` — preserved for rollback; weights
  NOT baked, runtime bind-mount required

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

For `:v2` and later: the `PointGroup-PAPER.pt` checkpoint is baked
into the image at `/opt/ForAINet/PointCloudSegmentation/model_file/`.
No runtime bind-mount required.

The build pulls from a release asset on this repo
(`forainet-weights-v1`) which was mirrored from the upstream Dropbox
link so GHA builds are deterministic. SHA-256
`97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861`
is verified at build time.

Canonical lab archive lives on NAS at
`/mnt/nas/datasets/ml_model_weights/forainet/PointGroup-PAPER.pt`
in case the GitHub release is ever lost.

For `:v1` (legacy, no baked weights), the original bind-mount
workflow still applies — see git history for that variant of this
README.

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
