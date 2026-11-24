# forpt

ForPT (Yue, Puliti, Robert and others, 2026) is a self-supervised foundation
model for forest lidar point clouds. A LitePT-S backbone (12.4 M parameters) was
pretrained on about 33 billion unlabeled points from airborne, UAV and mobile
laser scans, covering about 160 km² of forest across several regions, and then
fine-tuned for three forestry tasks. This image holds the pretrained encoder and
the three fully fine-tuned models:

- **Semantic segmentation** into ground, wood and leaf, trained on
  FOR-instanceV2 (ULS, MLS and TLS plots from Norway, the Czech Republic,
  Austria, French Guiana, New Zealand and Australia). On the FOR-instanceV2 test
  set the paper reports 88.3% mIoU and 95.6% overall accuracy, with per-class
  IoU of 99.1% for ground, 94.9% for leaf and 71.0% for wood (Table 1).
- **Tree species classification** of single-tree clouds into the 33 species of
  FOR-species20K, which was collected mostly in Europe with additional trees
  from Canada, Australia and New Zealand. The paper reports 83.2% overall
  accuracy and an F1 of 83.0% on the FOR-species20K test set (Table 5).
- **Tree age regression** on single-tree clouds, trained on FOR-age. The paper
  reports an RMSE of 20.03 years and an R² of 0.75 on the FOR-age test set
  (Table 7).

Paper: [arXiv:2609.24787](https://arxiv.org/abs/2609.24787) ·
Code: [prs-eth/ForPT](https://github.com/prs-eth/ForPT) @ `8304f57` ·
Weights: [prs-eth/ForPT](https://huggingface.co/prs-eth/ForPT) @ `6b9047e` ·
License: MIT (see [License](#license))

Pull: `ghcr.io/bradleylab/forpt:v1` (also `:latest`, `:torch2.4-cu124`).
amd64 only; inference needs a GPU, and the CUDA extensions are compiled for the
H100 (sm_90).

## Run it

### Ground, wood and leaf for every point

```bash
forpt-semseg plot.laz plot_forpt.laz
```

The output is the input file with one extra dimension, `forpt_class`: 0 ground,
1 wood, 2 leaf. Every other field is unchanged. Coordinates must be in meters.
The whole file is one forward pass, as in upstream's tester, so GPU memory sets
the largest file that fits; split large tiles before running. The FOR-instanceV2
training plots were ULS, MLS and TLS scans, so sparse airborne data is outside
what the segmentation model was trained on.

### Species of single trees

```bash
forpt-species trees/ --out species.csv --top-k 5
```

Each file holds one segmented tree in meters: LAS/LAZ, `.npy`, or
comma-separated `x,y,z` text, which is FOR-species20K's own format. Folders are
searched for those files, not recursively. The CSV has one row per tree with the
point count and the top-k species and scores. Scores are softmax values of the
logits, not calibrated probabilities.

The model knows only FOR-species20K's 33 species, and a tree of any other
species still gets one of those names. Eight of the 33 are North American:
*Acer saccharum*, *Picea glauca*, *Pinus contorta*, *P. resinosa*, *Populus
deltoides*, *P. tremuloides*, *Pseudotsuga menziesii* and *Quercus rubra*.
There is no hickory, and the other four oaks and both ashes are European.

### On Compute2

```bash
srun -A compute2-alexander.s.bradley \
  -p general-gpu \
  --gpus=1 \
  --container-image=ghcr.io#bradleylab/forpt:v1 \
  --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
  forpt-semseg <input.laz> <output.laz>
```

### From Python

`forpt_lib` (on the image's `PYTHONPATH`) builds each model from upstream's
config and loads its weights strictly. The pretrained encoder gives
self-supervised features for fine-tuning or for probing other tasks; the age
model has no command of its own.

```python
from pathlib import Path

import forpt_lib as fl

device = fl.require_cuda()
encoder = fl.build_pretrained_encoder(device)       # LitePT-S, encoder only
points = fl.read_points(Path("tree.laz"))           # (N, 3) float64, meters
fl.seed_everything()
batch = fl.model_input(points, fl.test_transform(fl.SPECIES), device)
features = encoder(batch).feat                      # coarsest-stage features, 504 wide

age_model = fl.build_task_model(fl.AGE, device)
```

Upstream's age test transform does no centering, so it takes the coordinates as
FOR-age stores them in `pc/<tree>/coord.npy`. Check how a cloud compares with
those before trusting an age.

## What the image does to a point cloud

Each model is built from upstream's config file, and each input passes through
that config's test-split transforms, so the preprocessing is upstream's:

1. Shift the cloud so its x–y center is at the origin and its lowest point at
   z = 0 (semseg and species).
2. Keep one point per 5 cm voxel; the x, y, z coordinates are the only input
   feature.
3. For semseg, center x and y again, run the model, and give every input point
   the label of its voxel.

The only change from upstream's transforms is that steps reading ground-truth
labels are dropped, since the inputs here are unlabeled.

## Choices made in this recipe

- **The fully fine-tuned models.** Upstream releases, for each task, a model
  trained from scratch, a linear probe and a fine-tuned model (and a decoder
  probe for segmentation). The fine-tuned ones score highest in the paper's
  100%-label rows, so they are the ones baked. The other checkpoints are on the
  same Hugging Face repository.
- **Weights reduced to state dicts.** Each checkpoint is checked against its
  Git LFS SHA-256 at build, then stored as its state dict and epoch. Upstream's
  fine-tuned checkpoints also pickle their optimizer and learning-rate scheduler,
  which needs `torch.load(weights_only=False)`; the reduced files load with
  `weights_only=True`, and upstream's `tools/test.py` still reads them. For the
  pretraining checkpoint the image keeps the student backbone, which is what
  upstream's fine-tuning configs start from.
- **Fixed seed.** Upstream's test transforms keep a random point per voxel, and
  the configs shuffle the serialization order on each forward pass. The
  commands seed both before each cloud, so a rerun gives the same answer.
- **Stack.** `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`, the torch and CUDA
  versions LitePT's install instructions name; spconv-cu124 2.3.8 and the PyG
  wheels for torch 2.4; FlashAttention 2.7.4.post1 from its prebuilt wheel,
  because LitePT's attention has no other code path. Upstream's pointops and
  PointROPE CUDA extensions are compiled in the image for sm_90. Every other
  package is pinned in `requirements.lock.txt`, resolved as of the day after
  ForPT's release.
- **Not included:** instance segmentation, which upstream lists as coming soon,
  and the training data.

## License

The code repository is MIT. The Hugging Face weights repository states no
license; this image treats the weights as covered by the code's MIT license.
The image contains none of the pretraining or benchmark data.

## Build-time check

The build runs `forpt-selftest` on the CPU with the network disabled. It builds
the pretrained encoder and the three task models from upstream's configs, loads
the baked weights strictly, checks the class lists (3 semseg classes, 33
species), and runs each task's test transforms on a synthetic tree placed at
projected-coordinate magnitudes. A forward pass needs the GPU, so on a GPU node
the same command also runs every model on that tree and checks the outputs are
finite. See `SMOKE.md` for the run on Compute2.

## Citation

```bibtex
@article{yue2026forpt,
  title   = {Toward a foundation model for forest point clouds},
  author  = {Yue, Yuanwen and Puliti, Stefano and Robert, Damien and Topalo{\u{g}}lu, Atakan and Xiang, Binbin and Wielgosz, Maciej and Wegner, Jan Dirk and Astrup, Rasmus and Rupprecht, Christian and Schindler, Konrad},
  journal = {arXiv preprint arXiv:2609.24787},
  year    = {2026}
}
```
