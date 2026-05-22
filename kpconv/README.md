# kpconv — H100 container

Kernel Point Convolution (KPConv) point-cloud **semantic segmentation**
(S3DIS / SemanticKITTI). Thomas et al., ICCV 2019. The most build-friendly
alt-architecture comparison in the AEC set — no transformer, no custom
CUDA op.

- Upstream code: https://github.com/HuguesTHOMAS/KPConv-PyTorch (MIT)
- Paper: [arXiv:1904.08889](https://arxiv.org/abs/1904.08889) (ICCV 2019)
- Architecture: KPConv operator (pure PyTorch) + CPU C++ grid-subsampling
  / radius-neighbor preprocessing.

## Image tag

`ghcr.io/bradleylab/kpconv:latest` (also `:v1`, `:torch2.2-cu121`).
GPU at runtime; built amd64.

## Contents

- PyTorch 2.2.2 + CUDA 12.1 (sm_90 via stock torch — no custom CUDA op)
- **numpy < 1.26** (the C++ wrappers' setup.py use `numpy.distutils`,
  removed in 1.26; the base's 1.26 is downgraded)
- The two CPU C++11 wrappers compiled in place
  (`cpp_subsampling.grid_subsampling`, `cpp_neighbors.radius_neighbors`)
- `matplotlib` (a hard import dep of `kernels/kernel_points.py`),
  `MPLBACKEND=Agg` for headless
- KPConv-PyTorch source at `/opt/KPConv-PyTorch` (run via `PYTHONPATH`)

## Weights are NOT baked — fetch + mount at runtime

Pretrained **S3DIS** checkpoints live on Google Drive; a live Drive pull
during the image build is throttling-prone, so they're fetched separately
and mounted. **No SemanticKITTI pretrained weights exist upstream**
(train-only). `gdown` is installed in the image for convenience.

| Model | mIoU | Drive file ID |
|-------|------|---------------|
| Light_KPFCNN | 65.4% | `14sz0hdObzsf_exxInXdOIbnUTe0foOOz` |
| Heavy_KPFCNN | 66.4% | `1ySQq3SRBgk2Vt5Bvj-0N7jDPi0QTPZiZ` |
| Deform_KPFCNN | 67.3% | `1ObGr2Srfj0f7Bd3bBbuQzxtjf0ULbpSA` |

```bash
# Fetch once (on the Mac or in a job), unzip into results/:
gdown 14sz0hdObzsf_exxInXdOIbnUTe0foOOz -O light_kpfcnn.zip
unzip light_kpfcnn.zip -d results/    # -> results/Log_YYYY-MM-DD_.../
```

Each `Log_*` folder holds `parameters.txt` (architecture, parsed
automatically) + `checkpoints/*.tar` (`model_state_dict`).

## Inference

`test_models.py` has **no CLI args** — edit the `__main__` block (or sed
it) to point `chosen_log` at the mounted `Log_*` folder:

```python
chosen_log = 'results/Log_2024-05-14_21-04-36'   # the unzipped checkpoint folder
chkp_idx = -1                                     # last checkpoint
on_val = True                                     # validation split
```

then `python test_models.py`. It reconstructs the net from
`parameters.txt`, loads the checkpoint, and runs `cloud_segmentation_test`
over the S3DIS validation split (expects the **raw S3DIS dataset** at
`../../Data/S3DIS` — the CPU wrappers do grid-subsampling + neighbor
search at data-load time). Output predictions land in `test/<log>/`.

S3DIS 13 classes: ceiling, floor, wall, beam, column, window, door,
chair, table, bookcase, sofa, board, clutter.

## Running on Compute2 (Pyxis/enroot)

```bash
cd /storage1/fs1/alexander.s.bradley/Active/c2_jobs
enroot import -o bradleylab+kpconv+v1.sqsh \
  'docker://ghcr.io#bradleylab/kpconv:v1'
```

Mount the unzipped checkpoint folder + the S3DIS dataset; `srun` with
`PYTHONNOUSERSITE=1`, `--container-workdir=/opt/KPConv-PyTorch`, and run
`python test_models.py` (with `chosen_log` pointed at the mount).

## Validation status

**`experimental`.** Build-smoke validated: numpy<1.26, headless
matplotlib, the two compiled CPU wrappers import, and the `KPFCNN` model
class is reachable. Runtime gate (S3DIS Area-5 mIoU, target ~65% for
Light_KPFCNN) needs the mounted weights + S3DIS dataset on Compute2.

## License

MIT (code; weights are the authors' release under the same repo). No
non-commercial restriction (unlike the ScanNet-bound `octformer`).

## Notes

- **No custom CUDA op** — the only compiled code is CPU C++; GPU use is
  stock PyTorch, so this is the most portable build in the set.
- `test_models.py` is edit-in-place (hardcoded config), not a CLI.
- Same base as the sibling images for parity.
