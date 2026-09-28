# satclip — Compute2 smoke test

Four steps. Importing the image and downloading the benchmark data are one-off
downloads on a login node. The GPU steps first repeat the build's self-test with
the encoder on an H100, then rerun one of the paper's published downstream
results, because an image whose encoders load but compute something different
would pass the lighter check.

## 0. Import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+satclip+v1.sqsh \
    'docker://ghcr.io#bradleylab/satclip:v1'

file -b bradleylab+satclip+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. Self-test on an H100

`satclip-selftest --device cuda` loads all six baked checkpoints onto the GPU,
embeds St. Louis, Kirkwood (MO), the Sahara and the Amazon basin, and checks
that the embeddings are finite, 256-dimensional, and place St. Louis nearer to
Kirkwood than to the other two. It fails at once if no GPU is visible.

```bash
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=2 --mem=8G --time=00:15:00 \
       -J satclip-selftest -o satclip-selftest-%j.out --wrap='
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+satclip+v1.sqsh \
     satclip-selftest --device cuda'
```

## 2. Download the elevation benchmark

The authors released the elevation dataset of the paper's downstream
experiments in a public Google Drive folder linked from upstream's README
("Downstream evaluation datasets"). It needs no credentials. The file is
`labels_elevation.csv`, 5.1 MB: 100,000 points (columns `elevation`, `Lat`,
`Lon`) subsampled from the global dataset of Rolf et al. (2021), of which 99,995
have an elevation, matching the paper's Table 5. The folder states no license,
so the file is used here for the test only and not redistributed.

On a login node:

```bash
D=/storage3/fs1/alexander.s.bradley/Active/satclip-benchmark
mkdir -p $D && cd $D
curl -fsSL -o labels_elevation.csv \
  'https://drive.usercontent.google.com/download?id=1YpbNV6k-hgCXfhOQQ5H-dOJVa7_W-tQr&export=download&confirm=t'
echo '32d3879981e9cffa6b441d0da1c2c9aab636fd03f6d97b4b0615d5769e302746  labels_elevation.csv' | \
  sha256sum -c -
```

A checksum mismatch means the file in the folder has changed; stop there rather
than compare a different dataset with the published numbers.

## 3. Rerun the published elevation regression on an H100

The published result is the Elevation row of Table 2 in Klemmer et al.,
"SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery",
arXiv:2311.17179v3 (AAAI 2025): test R² over 10 independently initialized MLP
runs of

| Location embedding | Published test R² |
|---|---|
| SatCLIP, L = 40 | 0.88 ± 0.00 |
| SatCLIP, L = 10 | 0.83 ± 0.01 |
| Identity (raw lon/lat) | 0.25 ± 0.08 |

Table 7 repeats these values. Its caption gives the backbone as ViT16, whereas
Table 2's caption gives ResNet50, so the default run scores both L = 40
checkpoints against the L = 40 row, plus the Identity baseline.

`satclip-verify-elevation` follows the paper's Appendix E.3 for what it
specifies: the five rows without an elevation dropped (and listed), a random
30/10/60 train/validation/test split, an MLP on the embedding trained with MSE
loss, the weights from the epoch with the lowest validation loss kept, 10 runs.
The split is drawn once and shared by the 10 runs, which differ in MLP
initialization; the paper does not say whether its runs shared a split. The
paper tuned each MLP by random search and does not report the values it
chose, so the MLP (64 hidden units, with B01's hidden layer applied twice with
shared weights, as B01 writes it), Adam at a learning rate of 0.001, 3,000
full-batch epochs, and scaling of the target by its maximum are taken from
upstream's notebook B01, the authors' own downstream example.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/satclip-benchmark
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=4 --mem=32G --time=01:00:00 \
       -J satclip-elev -o $D/satclip-elev-%j.out --wrap="
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+satclip+v1.sqsh \
     --container-mounts=$D:/bench \
     satclip-verify-elevation /bench/labels_elevation.csv --device cuda"
```

`--configs vit16-l10 resnet50-l10` scores the L = 10 checkpoints against the
L = 10 row.

**Reading the result.** Each line prints the mean and standard deviation of
test R² over the 10 runs beside the published value. Because the MLP settings
come from B01 rather than the paper's unreported tuned values, agreement is
judged against the published row and against the gap to Identity, not to the
last digit. A SatCLIP score near 0.88 and far above the Identity score
reproduces the paper. A SatCLIP score near the Identity score means the
embeddings carry no more information than the raw coordinates, which points to
the encoder or its checkpoint rather than to the regression.
