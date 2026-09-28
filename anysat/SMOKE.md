# anysat — Compute2 smoke test

Four steps. Importing the image and fetching the data happen once. Step 2 is a
forward pass on real imagery; step 3 asks whether the features carry land-cover
information that raw band values do not, because an encoder that returns
finite numbers of the right shape can still return nothing useful.

## The area of interest

A 3840 m square centered on the Tyson Research Center ForestGEO plot, whose
site page ([forestgeo.si.edu](https://forestgeo.si.edu/sites/north-america/tyson-research-center))
gives 38.5178 N, 90.5575 W. In WGS 84 / UTM 15N (EPSG:32615, the CRS of
Sentinel-2 tile 15SYC, which covers it) the center snaps to the 10 m grid at
712940 E, 4266100 N, so the AOI is

    711020 – 714860 E,  4264180 – 4268020 N

It is sixteen 960 m tiles. With 30 m patches each tile is 32×32 patches, the
most upstream advises per tile, and the AOI is 128×128 = 16,384 patches. The
square reaches past the plot on purpose: over the 20 ha plot alone NLCD is
almost entirely deciduous forest, which leaves nothing to classify.

| Layer | Source | What is fetched |
|---|---|---|
| NAIP | Planetary Computer `naip` | 2020 quarter-quads touching the AOI (0.6 m, R G B NIR), area-averaged to 1.25 m |
| Sentinel-2 | Planetary Computer `sentinel-2-l2a` | 2021 scenes with scene cloud cover ≤ 20 %, kept where ≤ 1 % of AOI pixels are no data, cloud, cloud shadow or cirrus in the SCL layer; one item per date; `BOA_ADD_OFFSET` removed per item |
| NLCD 2021 Land Cover | MRLC WCS, `NLCD_2021_Land_Cover_L48` | 30 m Albers, resampled nearest-neighbor onto the 10 m grid |

Planetary Computer is read anonymously through the `planetary_computer`
signer; MRLC needs no key. NLCD 2021 is not on Planetary Computer. NAIP has
no 2021 flight here; the nearest are July 2020 and June 2022, one year either
side of NLCD 2021, and 2020 is the default (`--naip-year 2022` switches).
`aoi.json` records the bounds and every STAC item ID used.

## 0. Import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+anysat+v1.sqsh \
    'docker://ghcr.io#bradleylab/anysat:v1'

file -b bradleylab+anysat+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. Fetch the AOI

On `general-cpu`, so the image needs the GPU override or enroot's driver hook
stops the container before it starts.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/anysat
mkdir -p $D
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=4 --mem=16G --time=02:00:00 \
       -J anysat-fetch -o $D/fetch-%j.out --wrap="
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+anysat+v1.sqsh \
     --container-mounts=$D:/data \
     fetch_aoi.py --out /data/tyson"
```

## 2. Forward pass on real imagery

NAIP and the Sentinel-2 series together, 960 m tiles, 30 m patches, with the
`dense` output on the Sentinel-2 grid as well as `tile` and `patch`. The
check after the run asserts the CUDA build and GPU, the output shapes, finite
values, and that no feature is constant across the AOI.

The check is a short script written next to the data first:

```bash
D=/storage3/fs1/alexander.s.bradley/Active/anysat
cat > $D/check_forward.py <<'EOF'
import numpy as np
import torch

assert torch.cuda.is_available(), "no GPU visible in the container"
assert torch.version.cuda.startswith("12.9"), torch.version.cuda
print(torch.cuda.get_device_name(0), "| CUDA build", torch.version.cuda)

f = np.load("features.npz")
expected = {"tile": (4, 4, 768), "patch": (128, 128, 768), "dense": (384, 384, 1536)}
for key, shape in expected.items():
    assert f[key].shape == shape, (key, f[key].shape)
    assert np.isfinite(f[key]).all(), f"non-finite values in {key}"
    spread = f[key].reshape(-1, shape[-1]).std(axis=0)
    assert (spread > 0).all(), f"a {key} feature is constant across the AOI"
    print(key, shape, "| median per-feature std", float(np.median(spread)))
print("FORWARD OK")
EOF

sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=64G --time=01:00:00 \
       -J anysat-fwd -o $D/fwd-%j.out --wrap="
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+anysat+v1.sqsh \
     --container-mounts=$D:/data \
     --container-workdir=/data/tyson \
     bash -c 'anysat_features.py --vhr naip.tif --vhr-key naip \
                --s2 s2.tif --s2-dates s2_dates.txt \
                --tile-m 960 --patch-m 30 --dense s2 --out features.npz \
              && python /data/check_forward.py'"
```

## 3. Linear probe against NLCD

Every 30 m patch is labeled with its modal NLCD class — at Level I (the first
digit of the code: water, developed, barren, forest, shrubland, herbaceous,
planted/cultivated, wetlands), then at the full Level II codes. Four feature
sets go through the same classifier:

- **`anysat_patch`**: the fused `patch` output, 768 values.
- **`anysat_dense`**: the `dense` output for `s2`, averaged over the patch's
  nine 10 m sub-patches, 1536 values. Its first half is the fused patch
  token repeated; its second half is the sub-patch feature.
- **`anysat_subpatch`**: that second half alone, 768 values.
- **`baseline`**: raw band means per patch, 4 NAIP bands over the patch's
  24×24 pixels and 10 Sentinel-2 bands over its 3×3 pixels and every date,
  14 values.

The three AnySat sets are scored separately because they carry different
information. The fused patch tokens of one tile are close to identical to
each other, already on upstream's own demo sample; the local detail sits
in the sub-patch half, which is why upstream sends segmentation through
`dense`.

The classifier is a linear probe: standardized features into multinomial
logistic regression (scikit-learn defaults, L2 with C = 1) with class weights
balanced by frequency. Evaluation is four-fold spatial cross-validation: each
fold holds out one 1920 m quadrant (NW, NE, SW, SE) and trains on the other
three, so train and test patches are never interleaved. Test patches of a
class absent from the fold's training quadrants are left out of that fold and
counted in the report. The primary metric is **macro-F1** over the classes
present in the held-out quadrant, because deciduous forest dominates the AOI
and overall accuracy would reward predicting forest everywhere; overall
accuracy is reported alongside. Step 3 needs no GPU.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/anysat
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=8 --mem=32G --time=01:00:00 \
       -J anysat-probe -o $D/probe-%j.out --wrap="
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+anysat+v1.sqsh \
     --container-mounts=$D:/data \
     --container-workdir=/data/tyson \
     bash -c 'nlcd_probe.py --data . --features features.npz --level 1 --out probe_level1.json \
              && nlcd_probe.py --data . --features features.npz --level 2 --out probe_level2.json'"
```

**Reading the result.** There is no published number to reproduce here; the
test is the comparison. Each report gives macro-F1 and accuracy per held-out
quadrant for every feature set. An AnySat set passes if its macro-F1 is above
the baseline's in most quadrants and on the mean; each of the three is judged
on its own. One that does not pass adds nothing a 14-number summary of the
same pixels does not already carry, at least for NLCD classes at 30 m. Before
blaming the model for that, check the normalization (`channel_stats`) and the
date handling. The report also gives class counts per quadrant and the median
label purity (the share of a patch's pixels that hold its modal class): a
class found in only one quadrant is never scored, because it is missing from
training whenever its quadrant is held out, and low purity bounds what any
classifier can reach.

Caveats that apply to every feature set equally: NLCD is itself a classified
product with its own errors, so these are weak labels; NAIP is from 2020,
NLCD and Sentinel-2 from 2021; and quadrant borders still leave neighboring
patches on either side of a split.
