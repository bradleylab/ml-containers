# fraxtex

fraXteX segments geological fractures in outcrop imagery pixel by pixel: given a
top-down photograph or orthomosaic of an outcrop, it returns the probability
that each pixel lies on a fracture trace. The four models are baselines from the
FraXet benchmark of Fatihi, Caldeira, Beucler, Thiele and Samsu, which compares
classical edge filters, CNNs and transformers for fracture mapping. FraXet
combines three published UAV outcrop datasets (Ovaskainen22, granite in
southeastern Finland; Matteo21, granite at the Granite Dells, Arizona; Samsu19,
Lower Cretaceous siliciclastic strata in southeastern Australia) into 8,953
co-registered 256×256 RGB+DEM patches with traced fractures, split spatially
into train, validation and test regions.

Paper: [Fatihi et al., EGUsphere preprint, doi:10.5194/egusphere-2026-1097](https://doi.org/10.5194/egusphere-2026-1097) ·
Weights: [ayoubft/fraXteX](https://huggingface.co/ayoubft/fraXteX) @ `890ccb4` ·
Data: [FraXet v0.1, doi:10.5281/zenodo.17069947](https://doi.org/10.5281/zenodo.17069947)

Pull: `ghcr.io/bradleylab/fraxtex:v1`

## The four models

All four ONNX files are baked into the image at `/opt/fraxtex/models`, pinned
to a Hugging Face revision and checked by sha256, with the upstream model card
beside them as `MODEL_CARD.md`. Each emits one channel: a sigmoid fracture
probability in [0, 1].

| Name | File | Architecture (model card) | Input | Tile |
|---|---|---|---|---|
| `unet-rgb` | `rgb/unet-rgb.onnx` | U-Net from scratch, 31.03 M parameters | RGB | any multiple of 16 |
| `sam2-rgb` | `rgb/sam2-rgb.onnx` | SAM 2.0 hiera-tiny encoder, frozen; mask decoder fine-tuned; prompt-free; 33.12 M | RGB | 512×512 |
| `unet-rgbdem` | `rgbdem/unet-rgbdem.onnx` | U-Net from scratch, 31.03 M | RGB + DEM | 256×256 |
| `segformer-rgbdem` | `rgbdem/segformer-rgbdem.onnx` | SegFormer decoder on an ImageNet ResNet-34, 21.87 M | RGB + DEM | 256×256 |

**The two directories do not share preprocessing.** Each file records its input
contract in its ONNX `metadata_props` (15 keys, the same schema in all four).
The values embedded in the baked files are:

| Key | `unet-rgb` | `sam2-rgb` | `unet-rgbdem` | `segformer-rgbdem` |
|---|---|---|---|---|
| `model.name` | unet | sam2_seg | unet | segformer |
| `input.channels` / `input.channel_order` | 3 / RGB | 3 / RGB | 4 / RGB+DEM | 4 / RGB+DEM |
| `input.range` | 0-255 | 0-255 | 0-1 | 0-1 |
| `preprocessing.in_graph` | true | true | false | false |
| `preprocessing.normalize` | dataset | imagenet | none | none |
| `input.dynamic_hw` / `input.fixed_size` | true / none | false / 512x512 | false / 256x256 | false / 256x256 |
| `input.dtype`, `input.layout` | float32, NCHW | float32, NCHW | float32, NCHW | float32, NCHW |
| `output.type`, `output.range` | sigmoid_probability, 0-1 | same | same | same |
| `citation.doi` | 10.5194/egusphere-2026-1097 | same | same | same |

The RGB models take raw 0–255 values and normalize inside the graph, with FraXet
statistics for the U-Net and ImageNet statistics for SAM 2. The RGB+DEM models
take values the caller has already scaled to 0–1 and apply no normalization.
Feeding a model the other directory's input produces plausible masks that are
wrong, not an error, so the scripts here read the contract from the file and
refuse any input they cannot make conform: wrong band count, a tile size the
model does not take, a value range or normalization scheme they do not
recognize, or metadata that disagrees with the graph's own input shape. The two
directories were also exported at different times with different toolchains
(torch 2.7.1 for `rgbdem/`, 2.13 for `rgb/`), so the model card cautions that a
difference between their outputs is not purely architectural.

## The DEM channel

How the DEM becomes the fourth channel is stated three different ways, and the
released weights reflect only one of them:

- The preprint (sec. 2.2) and the model card say the DEM is min–max scaled per
  256 px patch to 0–1, and the embedded `input.range` is 0-1.
- The upstream training loader (`src/dataset_benchm.py` in fractex2D.pt at
  `f5edf43`) reads the fourth band of the image file when it has one, else the
  DEM file; casts it to 8-bit integers; min–max scales it per patch; and then
  divides all four channels by 255. On FraXet that yields a DEM channel between
  0 and 1/255 for Matteo21, whose image TIFFs carry the DEM as band 4. Samsu19
  images are RGBA PNGs, so the loader takes their alpha band, constant at 255 in
  the patches inspected, and the channel is zero. Ovaskainen22 DEMs hold relative heights below 1 m in
  the test patches inspected (0.40–0.67 m), which truncate to zero.
- The authors' demonstration Space feeds the DEM without min–max scaling,
  divided by 255.

If the released weights came from that loader, the RGB+DEM models saw a DEM
channel that was zero or nearly so for most training patches, and they likely
rely on the RGB bands. The FraXet check in [`SMOKE.md`](SMOKE.md) scores both
models with the loader's treatment and with per-patch min–max; the one that
reproduces the published scores identifies what the weights expect.
`fraxtex-predict` therefore has no default: `--dem-scaling` must be one of

- `patch-minmax`: per-tile min–max to 0–1, as the preprint and model card state;
- `patch-minmax-255`: the same, divided by 255, the magnitude the training loader
  delivered where the DEM carried information (it omits the loader's 8-bit
  truncation, which would wrap absolute elevations above 255 m);
- `zero`: an all-zero fourth channel, what the loader delivered for two of the
  three datasets.

## Imagery it suits

Top-down UAV orthomosaics or photographs of rock outcrops, 8-bit RGB. FraXet's
ground sampling distance runs from 0.5–1.3 mm (Matteo21) through 5–6 mm
(Ovaskainen22) to 29–32 mm (Samsu19) (preprint Table 3), so one 256 px tile
covers 0.13–0.33 m, 1.28–1.54 m and 7.4–8.2 m of ground in the three
(authors' response to reviewers). Imagery much coarser than a few centimeters
per pixel lies outside the training data. A DEM is optional: the RGB models need
none, and the RGB+DEM models take one on the same pixel grid as the image.

## Running it

```bash
fraxtex-predict --model unet-rgb ortho.tif fractures_prob.tif

fraxtex-predict --model unet-rgbdem \
  --dem dem_on_ortho_grid.tif --dem-scaling patch-minmax \
  ortho.tif fractures_prob.tif \
  --mask fractures_mask.tif --threshold 0.1
```

The input is any raster GDAL reads (GeoTIFF, JPEG, PNG). Bands tagged red, green
and blue are used; an image that does not tag them has bands 1–3 used, and the
script says so. RGB must be 8-bit; other data types need `--rgb-range LOW HIGH`,
a linear map onto 0–255 chosen for the sensor. The DEM must have the same width
and height as the image and, when both are georeferenced, the same CRS and
geotransform; resampling is left to the user (for example `gdalwarp` with the
image's extent and resolution).

**Tiling and stitching.** Tiles lie on one global grid with a stride of
`tile × (1 − overlap)`, default overlap 0.5, the last row and column pushed flush
with the image edge. The fixed-size models use their own tile; `unet-rgb`
defaults to 256 px, the FraXet patch size it was trained on, and `--tile` takes
any multiple of 16. Each tile's output is weighted by a separable Hann taper
(sin² evaluated at pixel centers, so never exactly zero) and every pixel gets
the weighted mean of all tiles covering it, which suppresses the weak context at
tile edges without leaving seams. `--overlap 0` gives the non-overlapping tiling
the model card describes. The image is read and written in 4096 px blocks, each
running every grid tile that touches it, so orthomosaics of any size stream
through bounded memory and block boundaries produce the same values as
whole-image processing. An image smaller than one tile is mirror-padded and the
padding cropped.

**Nodata and alpha.** A pixel is invalid where the RGB dataset mask says so
(nodata value, alpha band or internal mask) or where the DEM is nodata or not
finite. Invalid pixels are nodata (−1) in the probability map and 255 in the
mask. Before inference, invalid pixels inside a tile are filled with the mean of
the tile's valid pixels, because a black fill is itself an edge: the preprint
attributes false positives in Matteo21 to NaNs that became black pixels. The
boundary between filled and real pixels is still artificial, so probabilities
within a tile's width of a nodata edge deserve suspicion. Tiles with no valid
pixel are skipped. The script prints how many tiles ran, how many were skipped
and how many pixels are nodata.

**Outputs.** A float32 GeoTIFF on the input's pixel grid and georeference
(taken from the DEM when only the DEM is georeferenced), deflate-compressed and
tiled, with tags recording the model file, its contract, the tiling and the DEM
scaling. `--mask` adds a uint8 GeoTIFF: 1 fracture, 0 background, 255 nodata,
from `--threshold`. The upstream evaluation thresholds at 0.1, also the
preprint's default for FracSim, and the preprint treats the threshold as a user
choice that trades precision against recall.

**Device.** `--device auto` (the default) uses CUDA when it initializes and
otherwise runs on CPU, printing which. `--device cuda` fails instead of falling
back. On a CPU partition, set `NVIDIA_VISIBLE_DEVICES=void` so enroot starts the
container at all.

## Scoring on FraXet

`fraxet-stage DEST` downloads FraXet v0.1 from Zenodo (2.64 GB), checks it
against the record's md5 and unpacks it. `fraxtex-eval-fraxet` then scores a
model on the 1,812 test patches with the preprint's protocol, reimplemented from
the upstream evaluation code: the label refinement of preprint sec. 2.2
(widening of dark wide fractures, then smoothing into rings out to 9 px, any
pixel of which counts as fracture), a 0.1 probability threshold, pixels pooled
over all patches with background included, and FracSim on the `list2.txt`
subset. Its docstring states each metric's definition. Two upstream behaviors
are kept because the published numbers include them: ROC-AUC is computed on the
thresholded map, which makes it (recall + specificity) / 2, and SegFormer
probabilities above 0.99 are set to zero before scoring. Upstream's random
test-time flips are not repeated. On a sample of FraXet patches the label
refinement matches upstream's functions exactly and every metric matches
torchmetrics to within 1e-5.

The preprint reports test scores for the two RGB+DEM models (Table D1; per
dataset in Table 6):

| Model | Prec | Rec | Spec | F1 | IoU | ROC | CK | SSIM | PSNR | FracSim |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net (`unet-rgbdem`) | 0.50 | 0.45 | 0.92 | 0.48 | 0.31 | 0.69 | 0.39 | 0.49 | 14.7 | 3.40 |
| SegFormer (`segformer-rgbdem`) | 0.38 | 0.52 | 0.84 | 0.44 | 0.28 | 0.68 | 0.32 | 0.40 | 13.2 | 3.43 |

It reports none for `unet-rgb` or `sam2-rgb`; the evaluation script scores them
on the same protocol without a published reference. SAM 2 takes 512 px tiles, so
each 256 px FraXet patch is enlarged 2× by pixel repetition and the output
averaged back over 2×2 blocks, a choice made here because the upstream
materials do not say how SAM 2 met 256 px patches. The authors' Space also
records that `rgbdem/unet-rgbdem.onnx` and the earlier PyTorch checkpoint
(`pytorch/unet.pt`, tag `v1-pytorch`) give different outputs on the same patch,
so reproducing Table D1 is the evidence that the ONNX weights are the ones
scored in the paper.

## Limitations

From the model card and the preprint:

- Thin or poorly illuminated fractures may be missed.
- Shadows, veins, vegetation, weathering and strong rock textures produce false
  positives; the preprint's error analysis also traces confident disagreements
  to traces drawn offset from the fracture pixels.
- Performance depends on annotation quality, illumination and lithology. The
  multi-site models reach F1 0.73 on the Ovaskainen22 test granite but 0.34 on
  Samsu19 and 0.39 on Matteo21 (U-Net, Table 6); the preprint notes these sit
  well below the F1 above 0.8 typical of deep-learning segmentation studies.
- The outputs are assistive probability maps for pre-annotation and review.
  They are not a structural interpretation and need expert validation.

## Stack

- Base: `python:3.12-slim-bookworm`.
- `onnxruntime-gpu==1.26.0` with its `[cuda,cudnn]` extras: the last PyPI release
  built for CUDA 12 (CUDA 12.8 and cuDNN 9 per the ONNX Runtime provider table;
  1.27 moved to CUDA 13), with the CUDA 12.9 runtime and cuDNN 9.26 as pip
  wheels. Compute2's driver, 580.105.08, supports it. The CUDA and cuDNN
  libraries are preloaded from site-packages with `onnxruntime.preload_dlls`
  before a CUDA session is created.
- rasterio, NumPy, scikit-image and SciPy for raster I/O and the evaluation.
- [`constraints.txt`](constraints.txt) freezes the full resolved set (25 pins,
  resolved 2026-09-28).

No PyTorch: the models are ONNX graphs. The build test runs `fraxtex-selftest`
offline on CPU: it prints package versions, asserts that the CUDA execution
provider is compiled in (no GPU is present at build time to run it), then opens
each baked model, prints its embedded contract, runs a batch of noise shaped to
that contract (320 px for the dynamic-size U-Net) and checks that the output has
the input's size and lies in [0, 1]. [`SMOKE.md`](SMOKE.md) runs the same test
on an H100 with CUDA required, and then the FraXet scoring.

## License

The weights are MIT, as declared on the Hugging Face repository; the RGB+DEM
U-Net and SegFormer are also released on Zenodo
([doi:10.5281/zenodo.17866853](https://doi.org/10.5281/zenodo.17866853)) under
CC-BY-4.0. `sam2-rgb.onnx` contains Meta's SAM 2.0 hiera-tiny image encoder
unchanged, which Meta releases under Apache-2.0; that license is shipped at
`/opt/licenses/SAM2-LICENSE`. FraXet is CC-BY-4.0, as are its three source
datasets, and is downloaded at run time rather than baked.

The upstream code repository, [ayoubft/fractex2D.pt](https://github.com/ayoubft/fractex2D.pt),
has no license, so no permission to copy or redistribute it has been granted and
none of it is in this image. It was read to learn the preprocessing and
evaluation protocol. The inference, evaluation, staging and self-test scripts
here are the lab's own.

Cite the preprint (its DOI is also embedded in every model file under
`citation.doi`) and the Zenodo model release:

> Fatihi, A., Caldeira, J., Beucler, T., Thiele, S. T., and Samsu, A.: Towards
> robust fracture mapping: benchmarking automatic fracture mapping in 2D outcrop
> imagery, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2026-1097, 2026.
