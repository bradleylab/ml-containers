# dino-cv-stonewall

Segmentation of historic dry-stone walls in airborne-lidar terrain models
(Huang, Islam, Armstrong, Bell, Khoshelham and Tomko, 2026). The input is two
relief visualizations of a 10 cm DEM, the Multi-directional Hillshade (MHS)
and the Visualization for Archaeological Topography (VAT). Each view has its
own encoder, and one decoder (a U-Net for the ResNets, a DPT head for the
ViT) outputs a wall probability per pixel.

The encoders were pretrained with DINO-CV, the paper's self-distillation
scheme across the two views, on 188,006 unlabeled tiles from the 301 km² lidar
survey of the Budj Bim Cultural Landscape in Victoria, Australia, a UNESCO
World Heritage site on Gunditjmara country. They were then fine-tuned on 7,102
tiles with hand-drawn masks of the European historic dry-stone walls near Tae
Rak (Lake Condah). The labels are colonial-era walls; the paper does not
claim to map Gunditjmara stone structures.

Evaluation is six-fold leave-one-area-out cross-validation over six areas of
the labeled region, with pixel metrics averaged over tiles:

| Backbone (two-view) | Parameters (paper) | F1 | mIoU | mIoU, 10% of labels |
|---|---|---|---|---|
| WideResNet-50-2 | 138 M | 81.4% | 68.6% | 63.8% |
| ResNet-50 | 46 M | 81.2% | 68.3% | 63.6% |
| ViT-S/16 | 42 M | 80.1% | 66.8% | 58.9% |

The models as upstream's code builds them are larger than the paper's
parameter counts (183.4 M for WideResNet-50-2, the size of the released
checkpoints); see `SMOKE.md`.

Paper: [Science of Remote Sensing 13:100372](https://doi.org/10.1016/j.srs.2026.100372) (open access, CC BY 4.0) ·
Code: [MLinArcheaomatics/BudjBimStoneWall-DINO-CV](https://github.com/MLinArcheaomatics/BudjBimStoneWall-DINO-CV) @ `fa4e65a` ·
License: CC BY-NC 4.0, culturally sensitive, see below

Pull: `ghcr.io/bradleylab/dino-cv-stonewall:v1`

## Restricted use

- **License.** The repository's `LICENSE` file is the Creative Commons
  Attribution-NonCommercial 4.0 International legal code (GitHub reports
  NOASSERTION because it does not classify NC licenses). The README's "Usage &
  Ethics" section says the repository contains culturally sensitive
  information, that the data and models are for academic research and
  educational purposes only, and that commercial use is strictly prohibited,
  including training proprietary models and selling derived works.
- **Ownership.** The paper states that all data and derived results are owned
  by the Gunditj Mirring Traditional Owners Corporation (GMTOC), which provided
  the data and oversaw its use under a memorandum of understanding. A GMTOC
  member is a co-author.
- **What this image does.** It carries upstream's code, under CC BY-NC 4.0,
  with the license text at
  `/opt/licenses/LICENSE.dino-cv-stonewall.CC-BY-NC-4.0`. **The fine-tuned
  checkpoints are not in the image** and are never fetched into it. The lab
  holds its own copy and mounts it at `/weights`.
- **Outputs.** Wall masks from this model are derived results of a model
  trained on GMTOC data. Keep them out of public buckets and public catalogs
  unless that use has been cleared.

## Input contract: the image does not make MHS or VAT

The paper says both views were made with the Relief Visualization Toolbox
(RVT) but gives no parameters for either (sun elevation, number of
directions, which VAT blend, the stretch to 8 bits), and the repository has
no code that makes them. Computing them here would mean choosing those
parameters, and a model fed differently stretched views returns worse masks
without failing. So the caller supplies the views:

- A folder with `mhs/` and `vat/` subfolders of GeoTIFF tiles, the same file
  name in each.
- Training tiles were 400 × 400 px at 10 cm (40 m), resized to 256 × 256 inside
  the model. Other tile sizes run, but are off the training distribution.
- Pixel values are read with upstream's `tiff_to_arr`: values in [0, 1] are
  scaled to [0, 255], anything else is clipped to [0, 255], and a single band
  is copied to three.

Upstream's released BudjBimArea dataset contains MHS and VAT tiles made by
the authors. Matching RVT settings against those tiles is how a pipeline for
new DEMs would be checked; until that is done, treat masks on other terrain as
unverified.

## Run it

```bash
stonewall-predict --tiles /path/to/tiles --out /path/to/masks --backbone wrn50
```

| Option | Default | Meaning |
|---|---|---|
| `--backbone` | `wrn50` | `wrn50`, `rn50` or `vitsmall16` |
| `--area` | `area3` | Which fold's checkpoint: each was fine-tuned with that area held out. `area3` is upstream's default in `main_pred.py` |
| `--conf` | `0.5` | Probability threshold, upstream's default |
| `--sieve` | `100` | `gdal_sieve` threshold in pixels, 8-connected, upstream's default |
| `--weights` | `/weights` | Mounted checkpoint folder |

The output is one binary GeoTIFF per tile in which walls remain after sieving,
with the input tile's CRS and transform.

`stonewall-predict` follows upstream's `main_pred.py` step for step. It
differs in three ways that leave predictions unchanged: it needs no `rgb/`
folder (upstream's dataset class requires one that the model never reads), it
builds the encoders without loading the pretrained backbones (the fine-tuned
checkpoint carries every weight), and it refuses a checkpoint whose keys do not
all match, where upstream loads with `strict=False`.

## Lab note: the private weights

The fine-tuned checkpoints are zip files behind University of Melbourne
Mediaflux share links in upstream's README. All five links resolved on
2026-10-11:

| Archive | Size (bytes) |
|---|---|
| `ft_wrn50.zip` (WideResNet-50-2) | 8,160,767,543 |
| `ft_rn50.zip` (ResNet-50) | 4,316,814,442 |
| `ft_vitsmall16.zip` (ViT-S/16) | 4,750,521,084 |
| `pretrained_weights.zip` (DINO-CV backbones, not needed for prediction) | 4,054,877,222 |
| `dataset.zip` (BudjBimArea tiles and labels) | 10,260,648,210 |

The share page loads a download link on port 8443 with `browser=false`; the
server ignores HTTP range requests. Stage the archives on the NAS
(`ml_model_weights/dino-cv-stonewall/`) and Storage3, unzip each backbone into
its own folder (`/weights/wrn50`, `/weights/rn50`, `/weights/vitsmall16`), and
write `/weights/SHA256SUMS` with a path relative to `/weights` for every
checkpoint. `stonewall-predict` looks for
`<backbone>/…/area<N>/ckpt/best_val_epoch.pth`, the path upstream's trainer
writes, and loads nothing that is not listed in `SHA256SUMS`, because a
checkpoint is a pickle and loading it runs code. Keep every copy private: no
public bucket, public Hugging Face repository or public image.

`ft_wrn50.zip` (SHA-256 `14a17c9c…a99bbba8`) holds six checkpoints of
1,468,433,514 bytes each, one per held-out area, at
`dino_cv_wrn50_budjbim_siamese/area<N>/ckpt/best_val_epoch.pth`, and the
training logs. Each checkpoint is a dict of `epoch`, `model_state_dict` and
`optimizer_state_dict`. The lab's copy is unpacked at
`/storage3/fs1/alexander.s.bradley/Active/dino-cv-stonewall/weights/wrn50/`
(owner-only), with `SHA256SUMS` beside it; the archive is on the NAS. All six
pass the hash check and load into this image's model with every key matched.

## Build-time check

With the network disabled, the build constructs all three two-view models as
upstream does, writes a checkpoint in upstream's format from random weights,
lists it in a `SHA256SUMS`, and runs `stonewall-predict` end to end on two
synthetic 400 × 400 tile pairs. This proves the code path, the checkpoint
check and the GeoTIFF round trip, not accuracy. See `SMOKE.md` for the check
with the real checkpoints.
