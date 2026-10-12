# hunting-pits

Detection of hunting pits in airborne-lidar DEMs (Lidberg, Westphal, Brax,
Sandström and Östlund, 2024). Hunting pits are dug hollows used to trap game,
common in the forests of northern Scandinavia. The model is a TensorFlow U-Net
that reads one terrain index, the profile curvature of a 0.5 m DEM, and labels
each pixel pit or background; the predictions are merged into polygons.

It was pretrained on lunar impact craters and trained on 2,519 hunting pits
mapped by hand in northern Sweden from Swedish national lidar (1–2 points/m²),
with 20% of the 250 × 250 px image chips held out at random for testing. The
paper compared ten terrain indices, two DEM resolutions (0.5 m and 1 m) and
three architectures; this U-Net on 0.5 m profile curvature was the best. On the
test chips, counted object by object, it found 70% of the pits (recall), 15% of
its detections were not pits (precision 85%), and F1 was 0.76. The results
file stored beside the weights in upstream's repository gives the counts: 342
pits found, 62 false detections and 149 missed.

Paper: [Journal of Field Archaeology 49:395–405](https://doi.org/10.1080/00934690.2024.2364428) (open access, CC BY 4.0) ·
Code: [williamlidberg/Detection-of-hunting-pits-using-airborne-laser-scanning-and-deep-learning](https://github.com/williamlidberg/Detection-of-hunting-pits-using-airborne-laser-scanning-and-deep-learning) @ `e45f3ca` ·
License: MIT (code and weights)

Pull: `ghcr.io/bradleylab/hunting-pits:v1`

## Run it

```bash
hunting-pits-infer /path/to/dem_tiles /path/to/output
```

`dem_tiles` is a folder of 0.5 m DEM GeoTIFFs in a metric projected CRS. Each
tile is processed whole, so tiles of a few thousand pixels on a side are the
practical size (upstream works with 2.5 × 2.5 km tiles, 5,000 px). The output
folder receives one shapefile per tile that contains at least one predicted pit;
tiles without pits produce nothing. Add `--output_type raster` for a binary
GeoTIFF per tile instead.

On Compute2:

```bash
srun -A compute2-alexander.s.bradley \
  -p general-gpu \
  --gpus=1 \
  --container-image=ghcr.io#bradleylab/hunting-pits:v1 \
  --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
  hunting-pits-infer <dem_dir> <out_dir>
```

## What the image does to a DEM

All of this is upstream's `semantic_segmentation/inference_unet_from_dem.py`,
run unmodified:

1. Profile curvature from the DEM with WhiteboxTools, clipped to [-2, 1] and
   rescaled to [0, 1].
2. U-Net prediction on 250 × 250 px windows with a 100 px margin, stitched back
   together keeping only each window's center.
3. Pixels of the pit class merged into polygons; polygons with an area of 30
   square map units or less dropped (m² in a metric CRS). Upstream's script
   also filters on perimeter-to-area ratio, with a threshold of -1 that keeps
   every polygon.

## Choices made in this recipe

- **No re-hosting.** Upstream publishes a ready image,
  `williamlidberg/hunting_pits:v1` (13.5 GB, built 2023-08-31 from the
  repository's `semantic_segmentation/Dockerfile`). This recipe builds from the
  repository at a pinned commit instead and keeps only the files inference
  needs.
- **Weights.** The weights are the repository's own
  `semantic_segmentation/trained_models/UNet/05m/profile_curvature1/trained.h5`,
  the file upstream's README passes to its inference command. Upstream's image
  copies the repository, and this file has not changed since the commit before
  that image was built. They are baked at
  `/opt/hunting-pits/weights/unet_05m_profile_curvature.h5` and checked against
  SHA-256 `b058c491…8614c` at build. The repository's top-level `models/`
  folder holds other files (named for lunar pretraining and transfer learning)
  and is not used.
- **WhiteboxTools 2.3.0, pinned.** Upstream's script downloads the
  WhiteboxTools binary on import, and the download serves the current release,
  so the version that computes the curvature would drift. The image installs
  v2.3.0, the release current when upstream built its image, as one fixed
  conda-forge build (`py311hfe55011_0`, compiled 2023-03-29), and sets
  `WBT_PATH`, which makes the whitebox package skip its download. v2.3.0 can
  no longer be compiled from its source tag: its lock file pins the Rust crate
  `tsp-rs` 0.1.0, which was removed from crates.io and whose name another
  author registered again in January 2026 for an unrelated crate.
- **TensorFlow 2.15.** The paper used TensorFlow 2.6 and upstream's image 2.8.
  2.15 is the last release on Keras 2, which reads these HDF5 weights through
  the same call upstream uses. PyTorch (CPU) is installed only because
  upstream's model module imports it for a class this model does not use.
- **The fixed working folder.** Upstream writes intermediates to
  `/workspace/temp_inference/`, which in this image points into `/tmp`, and
  deletes every `.tif` there when it finishes. The wrapper empties that folder
  and holds a lock on it, so two runs on one node take turns rather than
  deleting each other's files.

## Limits stated by the authors

- Not every pit is a hunting pit: other human-made and natural depressions can
  be detected as pits, and the authors judge that 1–2 points/m² was probably
  too sparse to tell them apart. Denser lidar would likely help.
- Some small features, including individual pits, were missed.
- Training used 1,275 km² of northern Sweden; other regions and other pit types
  are untested. The training locations came partly from earlier surveys, which
  can carry their biases.
- The 20% test set was drawn at random by chip, not by area, so test chips can
  sit next to training chips.

## Build-time check

The build runs, with the network disabled, upstream's script on upstream's
test chip (`data/test_chip/dem/12714.tif`) with the baked weights, and prints
how the polygons compare with the prediction upstream committed for the same
chip. It fails if nothing is predicted. The comparison is printed rather than
enforced because the WhiteboxTools and TensorFlow versions differ from those
in upstream's image. See `SMOKE.md` for the GPU run on Compute2.
