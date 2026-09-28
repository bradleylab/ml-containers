# panopticon — Compute2 smoke test

The risk this test addresses is cost at hyperspectral band counts. Panopticon's
patch embedding keeps a 2304-D token for every band of every patch, so memory and
time grow linearly with the band count (README, "Memory and time as the band
count grows"). Tanager-1 carries 426 bands and EMIT 285, against at most 13 in
any pretraining view. The test runs the baked teacher on real
224 × 224 patches from one Tanager scene and one EMIT granule, each band labeled
with the center wavelength stored in the file, and records peak GPU memory and
forward time as the band count rises.

## 0. Import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+panopticon+v1.sqsh \
    'docker://ghcr.io#bradleylab/panopticon:v1'

file -b bradleylab+panopticon+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. Stage one Tanager scene and one EMIT granule

Both live on the NAS, which only pliny mounts, and Compute2 cannot reach pliny,
so the copy runs from pliny over its `c2` ControlMaster alias, the route
TanagerFM's `scripts/stage_to_c2.sh` uses. Two files, about 2 GB for the EMIT
granule plus one Tanager scene:

- Tanager `20241121_183741_33_4001` (2024-11-21, southern Arizona), the
  desert/mineral scene of TanagerFM's C3 probe pairs, from
  `/mnt/nas/geospatial_data/2026-04-18_tanager/ortho_sr_hdf5/`. Planet Open Data
  ortho surface reflectance, HDF-EOS5; band centers, band widths and the
  `good_wavelengths` flags are attributes of the reflectance dataset.
- EMIT `EMIT_L2A_RFL_001_20240926T104903_2427007_029`, from
  `/mnt/nas/geospatial_data/2026-04-20_emit_l2a/`: native L2A reflectance,
  285 bands, 255 flagged good in TanagerFM's native-EMIT metadata audit. Only
  the native granule carries EMIT's own band centers; the EMIT WebDataset shards
  are resampled onto the Tanager grid and would test Tanager wavelengths twice.

```bash
ssh pliny
D=/scratch2/fs1/alexander.s.bradley/panopticon_smoke
ssh c2 "mkdir -p $D"
rsync -av \
  /mnt/nas/geospatial_data/2026-04-18_tanager/ortho_sr_hdf5/20241121_183741_33_4001_ortho_sr_hdf5.h5 \
  "c2:$D/"
rsync -av \
  /mnt/nas/geospatial_data/2026-04-20_emit_l2a/EMIT_L2A_RFL_001_20240926T104903_2427007_029/EMIT_L2A_RFL_001_20240926T104903_2427007_029.nc \
  "c2:$D/"
```

## 2. Measure memory and time against band count on an H100

[`scripts/panopticon_band_scaling.py`](scripts/panopticon_band_scaling.py) takes
224 × 224 grid windows that are complete on every good band (for Tanager, also
free of nodata, cloud and cirrus flags), standardizes each band over the window,
and runs the teacher with these band sets: 12, 50, 100 and 200 bands spread
evenly across the good bands, every good band, and every native band (426 for
Tanager, 285 for EMIT). Each set runs at batch 1 and batch 8, the batch filled
with distinct windows where the file has them. For every case it writes one JSON
line: median and minimum forward time over five timed passes after a warm-up,
peak allocated and reserved GPU memory, the bytes of the input and of the model
weights, the band count and wavelength range, and whether every output value is
finite. An out-of-memory case is recorded as `"status": "oom"` and the ladder
continues.

```bash
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=64G --time=01:00:00 \
       -J pano-bands -o pano-bands-%j.out --wrap='
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+panopticon+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/panopticon_smoke:/data \
     --container-workdir=/data \
     bash -c "
python -c \"import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)\" &&
for dtype in fp32 bf16; do
  python /opt/scripts/panopticon_band_scaling.py --sensor tanager \
    --path /data/20241121_183741_33_4001_ortho_sr_hdf5.h5 \
    --dtype \$dtype --out /data/results/tanager_\${dtype}_\$SLURM_JOB_ID.jsonl &&
  python /opt/scripts/panopticon_band_scaling.py --sensor emit \
    --path /data/EMIT_L2A_RFL_001_20240926T104903_2427007_029.nc \
    --dtype \$dtype --out /data/results/emit_\${dtype}_\$SLURM_JOB_ID.jsonl
done"'
```

**Reading the result.** Every `ok` record should show `finite: true` and a class
token of shape `[batch, 768]`. Peak allocated memory minus the model weights and
the input should grow in a straight line with `n_bands`. Reading the code gives at
least four fp32 blocks of 2.25 MiB per band and image alive at the peak, about
9 MiB per band per image, or about 3.7 GiB per image at 426 bands, plus whatever
layout copies the attention matmuls make; bf16 should come in near half. The
measured slope per band and per image, divided by 2.25 MiB, is the number of
band-sized blocks the patch embedding actually holds (fp32). Forward time should grow linearly
in `n_bands` from a constant trunk cost, with the patch embedding's arithmetic
passing the trunk's at roughly 8 bands. The records that decide how TanagerFM can use the model are the
all-band cases at batch 8: whether they fit in the H100's 80 GB, and the time per
image, which sets the cost of embedding a full scene. `filled_values` is nonzero
only for the all-band cases, in the bands the file flags as bad; those
embeddings serve the timing only.

## 3. A published benchmark

No published Panopticon number is reproduced here. Every result the paper
tabulates for the released model comes from a trained head: linear probes swept
over 78 configurations (two feature extractions × three poolings × 13 learning
rates, 50 epochs; paper §F.2, with flips and random resized crops in geobreeze's
configuration), UPerNet segmentation heads, or re-initialized patch embeddings.
The only training-free results, kNN accuracies, appear as curves in Fig. 3 and,
in the ablation tables, for 30-epoch ablation runs (paper §5) rather than the
released model. The hyperspectral results in Table 2 use SpectralEarth
Corine, which DLR distributes only after sign-up (upstream README), and Hyperview.

The closest candidate is GEO-Bench m-eurosat: the data are on Hugging Face
(`recursix/geo-bench-1.0`, not gated), and the paper reports 96.4 % top-1
linear-probing accuracy (Waldmann, Shah et al., 2025, Table 3 and Table 10).
Reproducing that value needs the paper's evaluation harness,
[geobreeze](https://github.com/geobreeze/geobreeze), with its GEO-Bench loader,
kornia augmentations and the 78-head sweep, none of which this image ships. A
simpler probe written for this image would measure a different protocol, and
its agreement or disagreement with 96.4 % would not show whether the image
computes what upstream's code computes.
