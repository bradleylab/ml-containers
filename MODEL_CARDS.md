# Model Cards

Provenance + status catalog for every container in this repo. One
entry per image; format borrowed from the model-card community to
make license, weights, and lab-status auditable in one place.

**Sync rule.** When a new container is added or an existing one moves
between status buckets (experimental → production, or vice versa),
update its card *in the same PR*. Top-level `README.md` and
`DEPRECATIONS.md` should agree with this file.

**Status levels.**

- **production** — used end-to-end on lab data, results referenced
  in deliverables.
- **production-capable** — built and validated; the lab has not yet
  used it on a deliverable but it works.
- **experimental** — first-run or known-fragile; results to be
  validated.
- **exploratory** — out-of-distribution for our data by construction;
  results expected to be informative more than reliable.
- **utility** — general-purpose; not tied to a specific lab task.

---

## segment-any-tree-h100

| | |
|--|--|
| Task | Tree-instance segmentation (UAV airborne lidar) |
| Sensor | UAV airborne lidar (ALS); also tested at TLS density |
| Upstream repo | [SmartForest-no/SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree) |
| Upstream license | Apache-2.0 |
| Paper | Wielgosz et al. (2024), *Remote Sensing of Environment* — [doi:10.1016/j.rse.2024.114367](https://doi.org/10.1016/j.rse.2024.114367) |
| Weights source | Ships in upstream repo via `git clone` during build; v2 patches the checkpoint's run_config in-place |
| Weights license | Same as upstream (Apache-2.0) per repo metadata |
| Container stack | PyTorch 2.2.2 + CUDA 12.1 + MinkowskiEngine (CiSong10 sm_90 fork) + torchsparse 1.4.0 + torch-geometric 2.5.3 |
| H100 status | Native sm_90 |
| Lab status | **production** |
| First-run / current behavior | Tyson 20-ha leaf-on (745 tiles, 2025-11-04): 221 instances/tile — over-fragments broadleaf canopy; tuned-clustering v2 reduces this. Leaf-off collapses (out-of-distribution; trained on Nordic conifer apexes). |
| Tags | `:v2` (= `:latest`, UAV-tuned clustering); `:v2-defaults` (paper-default clustering); `:v1` (pre-bug-fix) |

## ams3d-crownseg

| | |
|--|--|
| Task | Adaptive mean-shift crown segmentation (classical) |
| Sensor | UAV airborne lidar (ALS) |
| Upstream repo | [Lenostatos/crownsegmentr](https://github.com/Lenostatos/crownsegmentr) (R wrapper around AMS3D C++) |
| Upstream license | GPL-3.0+ |
| Paper | Ferraz et al. (2016), *Remote Sensing of Environment* 183, 318–333 — [doi:10.1016/j.rse.2016.05.028](https://doi.org/10.1016/j.rse.2016.05.028) |
| Weights source | None (classical algorithm; no learned components) |
| Weights license | N/A |
| Container stack | rocker/geospatial (R) + crownsegmentr + PDAL conda-forge for COPC writer |
| H100 status | N/A (CPU only) |
| Lab status | **production** — UAV-canonical at Tyson |
| First-run / current behavior | Tyson 20-ha leaf-on: 1,999 crowns segmented; baseline against which SAT is compared |
| Tags | `:v1` (= `:latest`) |

## fsct

| | |
|--|--|
| Task | TLS DL stem-point semantic segmentation → classical cylinder fit for DBH |
| Sensor | TLS / MLS |
| Upstream repo | [philwilkes/FSCT](https://github.com/philwilkes/FSCT) (archived; active fork at [tls-tools-ucl/TLS2trees](https://github.com/tls-tools-ucl/TLS2trees)) |
| Upstream license | **No LICENSE file in repo root** — treat as research use only; do not redistribute |
| Paper | Krisanski et al. (2021), *Remote Sensing* 13(8), 1413 — [doi:10.3390/rs13081413](https://doi.org/10.3390/rs13081413) |
| Weights source | Ships in upstream repo |
| Weights license | Inherits unclear-license posture from upstream |
| Container stack | PyTorch 1.9 + CUDA 11.1 (FSCT pins, predates sm_90) |
| H100 status | **NO** — CPU-only on Compute2 (porting to cu118 not worth the effort given it works on `general-cpu`) |
| Lab status | **production** |
| First-run / current behavior | BP7 TLS voxel02cm, 2026-04-24, Compute2 job 609998: 184 stems, median DBH 25.8 cm. UAV inputs fail by sensor geometry, not training distribution |
| Tags | `:v1` (= `:latest`) |

## sam2

| | |
|--|--|
| Task | General-purpose image segmentation (auto + point + box prompts) |
| Sensor | Any 2D RGB image; bundled `samgeo` adds georeferenced raster I/O |
| Upstream repo | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| Upstream license | Apache-2.0 (code); BSD-3-Clause (weights) — separate licenses, verify if redistributing weights externally |
| Paper | Ravi et al. (2024), *SAM 2: Segment Anything in Images and Videos* — [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) |
| Weights source | Hugging Face Hub on first run (`facebook/sam2.1-hiera-large` default; tiny / small / base-plus / large variants selectable via `--model-id`) |
| Weights license | BSD-3-Clause (Meta) |
| Container stack | PyTorch 2.5.1 + CUDA 12.1 (runtime variant); `SAM2_BUILD_CUDA=0` skips optional connected-components extension |
| H100 status | Native sm_90 |
| Lab status | **utility** — no specific Tyson task yet; bundled `samgeo` enables georeferenced workflows on demand |
| First-run / current behavior | Generic; deployed but no specific evaluation on lab data |
| Tags | `:latest` |

## treelearn

| | |
|--|--|
| Task | DL tree-instance segmentation (offset prediction + clustering) |
| Sensor | Ground-based lidar (TLS / MLS) |
| Upstream repo | [ecker-lab/TreeLearn](https://github.com/ecker-lab/TreeLearn) |
| Upstream license | MIT |
| Paper | Henrich et al. (2024), *Ecological Informatics* — [doi:10.1016/j.ecoinf.2024.102888](https://doi.org/10.1016/j.ecoinf.2024.102888) |
| Weights source | Göttingen dataverse [doi:10.25625/VPMPID](https://doi.org/10.25625/VPMPID) — fetched at runtime via bundled `download_weights.sh` (dataverse is too flaky for build-time fetch); 3 variants attempted, ≥1 success required |
| Weights license | MIT (per upstream repo) |
| Container stack | PyTorch 2.0.0 + CUDA 11.8 + spconv-cu118 (sparse-conv backbone with sm_90 wheels); `setuptools<80` pin so `munch==2.5.0` survives |
| H100 status | Native sm_90 (via cu118) |
| Lab status | **production** — the model that actually works at Tyson |
| First-run / current behavior | BP7 plot 2025-09-02: 101 instances, ~93 stems/ha |
| Tags | `:v1` (= `:latest`, `:torch2.0-cu118`) |

## pointstowood

| | |
|--|--|
| Task | DL semantic leaf-wood segmentation (per-point binary classification) |
| Sensor | High-resolution TLS |
| Upstream repo | [harryjfowen/PointsToWood](https://github.com/harryjfowen/PointsToWood) (default branch `version1.0-paper`) |
| Upstream license | **AGPL-3.0** (strong copyleft; container redistribution requires source availability — link to upstream repo satisfies that) |
| Paper | Owen, Allen, Grieve, Wilkes & Lines (2025, in review) — [arXiv:2503.04420](https://arxiv.org/abs/2503.04420) |
| Weights source | In-tree at `version1.0-paper` (`pointstowood/model/global.pth`, 73 MB); verified loadable at build time via `torch.load` |
| Weights license | AGPL-3.0 (per upstream repo) |
| Container stack | PyTorch 2.5.1 + CUDA 12.1 + PyG ecosystem (cu121 wheels) |
| H100 status | Native sm_90 |
| Lab status | **exploratory** — out-of-distribution at Tyson |
| First-run / current behavior | BP7 leaf-on TLS: collapses to 99.996% wood. Two OOD axes: training set is leaf-off + RIEGL VZ-calibrated dB; ours is leaf-on + raw linear intensity. Confidently wrong. |
| Tags | `:v1` (= `:latest`, `:torch2.5-cu121`) |

## 3dfin

| | |
|--|--|
| Task | Deterministic TLS stem detection + DBH estimation |
| Sensor | TLS |
| Upstream repo | [3DFin/3DFin](https://github.com/3DFin/3DFin) (PyPI: `3DFin`) |
| Upstream license | GPL-3.0 |
| Paper | Laino et al. (2024), *Forestry: An International Journal of Forest Research* — [doi:10.1093/forestry/cpae020](https://doi.org/10.1093/forestry/cpae020) |
| Weights source | None (classical algorithm; no learned components) |
| Weights license | N/A |
| Container stack | python:3.11-slim + 3DFin pip + `laspy[lazrs]` + geopandas/shapely/pyproj |
| H100 status | N/A (CPU only) |
| Lab status | **production-capable** — ready for use, no Tyson deliverable yet |
| First-run / current behavior | Wrapper script smoke-tests the CLI; no production output yet |
| Tags | `:v1` (= `:latest`) |

## backman-thermal-deer

| | |
|--|--|
| Task | Frame-level animal detection in thermal video (recurrent ONNX) |
| Sensor | DJI XT2 thermal video (640×512) |
| Upstream | No GitHub repo; code + ONNX model distributed via Zenodo: [doi:10.5281/zenodo.14799290](https://doi.org/10.5281/zenodo.14799290) |
| Upstream license | See Zenodo deposit — license lives in the deposit, not surfaced via repo metadata |
| Paper | Backman et al. (2025), *Methods in Ecology and Evolution* — [doi:10.1111/2041-210X.70006](https://doi.org/10.1111/2041-210X.70006) |
| Weights source | NOT bundled into image — `inferenceExample/` (model.onnx + generateVideoPredictions.py) bind-mounted at runtime because redistribution rights are not established. Wrapper at `tyson-deer-survey/tyson-thermal-deer-survey/scripts/run_backman_inference.py` |
| Weights license | Unclear; bundling deferred until license is verified |
| Container stack | python:3.11-slim + onnxruntime + opencv-python-headless + ffmpeg |
| H100 status | N/A (CPU runtime; ONNX recurrent model is small) |
| Lab status | **production** |
| First-run / current behavior | Tyson 2026-03-23 thermal flights: 6 flights / 30 segments / 128,638 frames → 21 deer after size + persistence filtering; 73 deer site-population estimate by strip transect (CI 45–104) |
| Tags | `:v1` (= `:latest`) |

## deepforest

| | |
|--|--|
| Task | Aerial RGB tree-crown detection (RetinaNet) |
| Sensor | Aerial RGB GeoTIFF (~10 cm GSD assumed by default `patch_size`) |
| Upstream repo | [weecology/DeepForest](https://github.com/weecology/DeepForest) |
| Upstream license | MIT (code); software DOI [doi:10.5281/zenodo.2538143](https://doi.org/10.5281/zenodo.2538143) |
| Papers | Weinstein et al. (2020), *MEE* — [doi:10.1111/2041-210X.13472](https://doi.org/10.1111/2041-210X.13472); model citation Weinstein et al. (2019), *Remote Sensing* 11(11), 1309 — [doi:10.3390/rs11111309](https://doi.org/10.3390/rs11111309) |
| Weights source | Hugging Face Hub on first call: `weecology/deepforest-tree:main`. Cache at `$HF_HOME=/opt/hf-cache` (bind-mount for persistence) |
| Weights license | Per HF model card |
| Container stack | nvidia/cuda:12.1.0-cudnn8 + python 3.11 + PyTorch 2.5.1 + torchvision 0.20.1 + `deepforest>=2.1.0` + `opencv-python-headless` |
| H100 status | Native sm_90 |
| Lab status | **exploratory** — NEON-pretrained checkpoint not Tyson-calibrated |
| First-run / current behavior | First Tyson run scheduled 2026-05-01; results TBD |
| Tags | `:v1` (= `:latest`, `:torch2.5-cu121`) |

## forainet

| | |
|--|--|
| Task | Panoptic (semantic + instance) segmentation of airborne lidar — PointGroup-style 3-head architecture, 7-level sparse U-Net backbone |
| Sensor | Airborne lidar; 5 semantic classes (ground, low veg, stems, live branches, dead branches) |
| Upstream repo | [prs-eth/ForAINet](https://github.com/prs-eth/ForAINet) |
| Upstream license | **No LICENSE file in repo root** — default copyright; redistribution and derivative use not granted by upstream. The lab's `:v2` image baking the weights is a research-use posture parallel to `segment-any-tree-h100`. Do not push this image to public GHCR with intent to redistribute outside the lab without upstream contact |
| Paper | Xiang et al. (2024), *Remote Sensing of Environment* 305, 114078 — [doi:10.1016/j.rse.2024.114078](https://doi.org/10.1016/j.rse.2024.114078) (DOI to be re-verified before any publication-grade citation) |
| Weights source | Upstream Dropbox link → mirrored to GitHub Release [`forainet-weights-v1`](https://github.com/bradleylab/ml-containers/releases/tag/forainet-weights-v1) on this repo. Canonical lab archive on NAS at `/mnt/nas/datasets/ml_model_weights/forainet/PointGroup-PAPER.pt`. SHA-256 `97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861` verified at build time |
| Weights license | Not stated by upstream — treat same as upstream code |
| Container stack | PyTorch 2.2.2 + CUDA 12.1 + MinkowskiEngine (CiSong10 sm_90 fork) + torchsparse 1.4.0 + torch-geometric 2.5.3 + torch-points-kernels (CUDA-12 patches) + torch_points3d (PyG-2.x compat patches, mirroring SAT v1) + hydra 1.0.7 / omegaconf 2.0.6 |
| H100 status | Native sm_90 (experimental forward-port from upstream's torch 1.9 / CUDA 11.1 stack which cannot target sm_90) |
| Lab status | **experimental** — first end-to-end run 2026-05-01; expected to underperform SAT at Tyson density (training set requires >75 pts/m², Tyson UAV is ~28 pts/m²) per `tyson-forest-linkage/.claude/memory/forainet_evaluation.md`. This run is empirical confirmation, not production segmentation |
| First-run / current behavior | TBD (2026-05-01) |
| Tags | `:v2` (= `:latest`, weights baked); `:v1` (legacy, weights bind-mounted at runtime) |

## forestformer3d

| | |
|--|--|
| Task | Transformer-panoptic 3D forest instance segmentation (OneFormer3D-based) — replaces PointGroup-style clustering with learned instance queries; no post-hoc clustering parameters |
| Sensor | Airborne / UAV lidar; trained on FOR-instanceV2 (extends FOR-instance with TU_WIEN deciduous alluvial leaf-off + BlueCat broadleaf temperate) |
| Upstream repo | [SmartForest-no/ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D) |
| Upstream license | CC BY-NC 4.0 (inherited from OneFormer3D base) — academic use OK; commercial requires upstream permission |
| Paper | Xiang et al. (2025), *Proceedings of ICCV* (Oral) — [arXiv:2506.16991](https://arxiv.org/abs/2506.16991) |
| Weights source | Zenodo record [16742708](https://zenodo.org/records/16742708): `clean_forestformer.zip` (~198 MB, md5 `553d67379331966509076f3fbb409e57`) → `epoch_3000_fix.pth`. Runtime fetch via `download_weights.sh` (Zenodo can be flaky) |
| Weights license | CC BY-NC 4.0 |
| Container stack | nvidia/cuda 11.8.0-cudnn8-devel-ubuntu22.04 + pip torch 1.13.1+cu117 + mmengine 0.7.3 / mmcv 2.0.0 / mmdet 3.0.0 / mmsegmentation 1.0.0 / mmdet3d @ 22aaa47 + MinkowskiEngine NVIDIA @ 02fc608 (rebuilt sm_90) + spconv-cu118 2.3.6 + cumm-cu118 0.4.11 + segmentator @ 76efe46 + torch-scatter 2.0.9. `replace_mmdetection_files/` overlay applied at build time |
| H100 status | Native sm_90 (Plan B build — lowest deviation from upstream pinned stack; Plan A fallback to torch 2.2 / cu121 documented in `forestformer3d/README.md`) |
| Lab status | **experimental** — first end-to-end run pending. Realistic Tyson F1 expectation: 60-70% (89 pts/m² is below FOR-instanceV2 training distribution; closed-canopy leaf-on broadleaf not in training). Published per-site F1: TU_WIEN 76.7%, Wytham 75.0%, BlueCat 61.7% |
| First-run / current behavior | TBD |
| Tags | `:v1` (= `:latest` = `:torch1.13-cu118-planB`); weights NOT baked, fetched via `download_weights.sh` |

## seisbench

| | |
|--|--|
| Task | Seismic phase picking (P/S onset detection) |
| Sensor | Seismic waveforms; 3-component preferred for EQTransformer, single-component acceptable for PhaseNet |
| Upstream repo | [seisbench/seisbench](https://github.com/seisbench/seisbench) |
| Upstream license | GPL-3.0 |
| Paper | Woollam et al. (2022), *Seismological Research Letters* — [doi:10.1785/0220210324](https://doi.org/10.1785/0220210324). Bundled architectures: PhaseNet (Zhu & Beroza 2019, *GJI*, [doi:10.1093/gji/ggy423](https://doi.org/10.1093/gji/ggy423)); EQTransformer (Mousavi et al. 2020, *Nature Communications*, [doi:10.1038/s41467-020-17591-w](https://doi.org/10.1038/s41467-020-17591-w)) |
| Weights source | SeisBench model zoo (S3-hosted) on first call to `Model.from_pretrained(...)`. Cache at `$SEISBENCH_CACHE_ROOT=/opt/seisbench-cache` (bind-mount for persistence) |
| Weights license | Varies by checkpoint in the model zoo; verify per pretrained model before redistribution |
| Container stack | python:3.11-slim + PyTorch 2.5.1 (CPU wheels) + `seisbench>=0.7` + `obspy>=1.4` + `h5py` + `pandas` |
| H100 status | N/A (CPU runtime by design; CUDA variant can be added later if needed) |
| Lab status | **utility** — no specific lab seismic deliverable; SeisBench enables phase picking workflows on demand, parallel to sam2's role for image segmentation |
| Architecture | **Multi-arch from v2** — `linux/amd64` + `linux/arm64`. Apple-Silicon Macs pull native arm64; Compute2 / EC2 pull amd64. v1 was amd64-only |
| First-run / current behavior | v2 build smoke test passes (2026-05-02); first run-time validation captured in PR review |
| Tags | `:v2` (= `:latest`, `:torch2.5-cpu`) — current; `:v1` retained for rollback (amd64-only) |

## neuralhydrology

| | |
|--|--|
| Task | Rainfall-runoff / streamflow prediction (LSTM) |
| Sensor | Time-series: meteorological forcings (precipitation, temperature, radiation, etc.) + streamflow observations |
| Upstream repo | [neuralhydrology/neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) |
| Upstream license | BSD-3-Clause |
| Paper | Kratzert, Gauch, Nearing & Klotz (2022), *JOSS* — [doi:10.21105/joss.04050](https://doi.org/10.21105/joss.04050) |
| Weights source | User-supplied checkpoint directory bind-mounted at runtime (`run_dir/` with `model_epochXXX.pt` + `config.yml`). Pretrained CAMELS checkpoints linked from the NeuralHydrology research blog |
| Weights license | Per checkpoint provenance — verify before redistribution |
| Container stack | python:3.11-slim + PyTorch 2.5.1 (CPU wheels) + `neuralhydrology>=1.13` + xarray/netcdf4/numba/h5py/pandas/scipy |
| H100 status | N/A (CPU runtime; this image targets inference. Training would need a separate CUDA variant) |
| Lab status | **utility** — no specific lab hydrology deliverable; the library enables LSTM rainfall-runoff workflows on demand |
| First-run / current behavior | Build smoke test passes (2026-05-01); `nh-run --help` resolves; no production inference output yet |
| Tags | `:v1` (= `:latest`, `:torch2.5-cpu`) |

## remoteclip

| | |
|--|--|
| Task | Remote-sensing image-text retrieval / zero-shot scene classification (CLIP architecture) |
| Sensor | RGB satellite imagery (Sentinel-2, NAIP, UAV — anything CLIP ingests as RGB at ~224 px) |
| Upstream repo | [ChenDelong1999/RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) |
| Upstream license | Apache-2.0 |
| Paper | Liu, Chen, Guan, Zhou, Zhu, Ye, Fu, Zhou (2024), *IEEE TGRS* — preprint [arXiv:2306.11029](https://arxiv.org/abs/2306.11029); IEEE Xplore record [10504785](https://ieeexplore.ieee.org/document/10504785) |
| Weights source | Hugging Face Hub: [`chendelong/RemoteCLIP`](https://huggingface.co/chendelong/RemoteCLIP). Three OpenCLIP-format checkpoints: `RemoteCLIP-RN50.pt` (~400 MB), `RemoteCLIP-ViT-B-32.pt` (~600 MB), `RemoteCLIP-ViT-L-14.pt` (~1.7 GB). Cache at `$HF_HOME=/opt/hf-cache` for bind-mount persistence |
| Weights license | Per HF model card; verify before redistribution |
| Container stack | python:3.11-slim + PyTorch 2.5.1 + torchvision 0.20.1 (CPU wheels) + `open-clip-torch>=2.20` + `huggingface_hub>=0.25` + Pillow |
| H100 status | N/A (CPU runtime by design; CUDA variant deferred until a batch-embedding workload lands) |
| Lab status | **utility** — no specific lab task; the container enables zero-shot scene classification and embedding workflows on demand, parallel to sam2 |
| First-run / current behavior | Build smoke test passes (2026-05-01); `open_clip.create_model_and_transforms` and `huggingface_hub.hf_hub_download` reach; no production inference output yet |
| Tags | `:v1` (= `:latest`, `:torch2.5-cpu`) |

## satlas

| | |
|--|--|
| Task | Pre-trained foundation-model backbones for remote-sensing imagery (downstream: classification, detection, segmentation, regression) |
| Sensor | Sentinel-2 RGB + 9-band MS; Sentinel-1 VH+VV; Landsat 8/9 all-bands; 0.5–2 m/px aerial RGB |
| Upstream repo | [allenai/satlaspretrain_models](https://github.com/allenai/satlaspretrain_models) |
| Upstream license | Apache-2.0 (code); [ODC-BY](https://github.com/allenai/satlas/blob/main/DataLicense) (weights) — separate licenses |
| Paper | Bastani et al. (2023), *ICCV* — *SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding*. [Open access PDF](https://openaccess.thecvf.com/content/ICCV2023/html/Bastani_SatlasPretrain_A_Large-Scale_Dataset_for_Remote_Sensing_Image_Understanding_ICCV_2023_paper.html); arXiv: [2211.15660](https://arxiv.org/abs/2211.15660) |
| Weights source | [`allenai/satlas-pretrain`](https://huggingface.co/allenai/satlas-pretrain) on HF Hub. The upstream `Weights().get_pretrained_model(...)` fetches via `requests.get` + `BytesIO` and does NOT cache on disk; for repeated jobs, pre-download to a host dir |
| Weights license | ODC-BY |
| Container stack | nvidia/cuda:12.1.0-cudnn8 + python 3.11 + PyTorch 2.5.1 + torchvision 0.20.1 (cu121) + `satlaspretrain-models>=0.3.1` |
| H100 status | Native sm_90 |
| Lab status | **utility** — pretrained backbones, no specific lab task; downstream fine-tuning + heads required for any actual prediction. Recommended tier: Compute2 H100 |
| First-run / current behavior | Build smoke test passes (2026-05-01); 14 checkpoint IDs reachable via `SatlasPretrain_weights`; no production inference output yet |
| Tags | `:v1` (= `:latest`, `:torch2.5-cu121`) |

## clay

| | |
|--|--|
| Task | Multi-sensor Earth-observation foundation model — per-patch embeddings, similarity search, clustering, lightweight downstream classification |
| Sensor | Sentinel-2 (multi-spectral), Sentinel-1 SAR, Landsat, NAIP, MODIS |
| Upstream repo | [Clay-foundation/model](https://github.com/Clay-foundation/model) |
| Upstream license | Apache-2.0 (code + weights) |
| Paper / docs | [clay-foundation.github.io/model](https://clay-foundation.github.io/model) (project Jupyter Book; papers in progress); model card on HF Hub |
| Weights source | [`made-with-clay/Clay`](https://huggingface.co/made-with-clay/Clay) on HF Hub: `v1.5/clay-v1.5.ckpt` (~3 GB). Cache at `$HF_HOME=/opt/hf-cache` for bind-mount persistence |
| Weights license | Apache-2.0 (per upstream README) |
| Container stack | nvidia/cuda:12.1.0-cudnn8 + python 3.11 + PyTorch 2.5.1 + torchvision 0.20.1 (cu121) + `claymodel==1.5.0` (pinned to upstream commit `f14e698`) + Lightning / timm / vit-pytorch / geopandas / scikit-image |
| H100 status | Native sm_90 |
| Lab status | **utility** — pretrained foundation model, no specific lab task; downstream embedding + adapter required for any prediction. Recommended tier: Compute2 H100 (batch embedding is the killer use) |
| First-run / current behavior | Build smoke test passes (2026-05-01); `ClayMAEModule` and `ClayDataModule` import cleanly; no production embedding output yet |
| Tags | `:v1` (= `:latest`, `:torch2.5-cu121`) |

## xrd-classifier

| | |
|--|--|
| Task | Powder X-ray diffraction phase identification — probabilistic multi-phase ID via 1D CNN trained on simulated patterns with physics-informed augmentation |
| Sensor | 1D XRD pattern (intensity vs 2θ; standard 5–90°, two-column ASCII) |
| Upstream repo | [njszym/XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer) (note: `PV-Lab/autoXRD` is a different project with the same package name — do not conflate) |
| Upstream license | MIT |
| Paper | Szymanski et al. (2021), *Chem. Mater.* — *Probabilistic Deep Learning Approach to Automate the Interpretation of Multi-phase Diffraction Spectra*, [doi:10.1021/acs.chemmater.1c01071](https://doi.org/10.1021/acs.chemmater.1c01071); follow-up Adaptive XRD, *npj CompMat* 2023 |
| Weights source | Upstream `Example/Model.h5` (Li-Mn-Ti-O-F demo system, 73 MB) baked in at `/opt/xrd-autoanalyzer/Example/Model.h5` via pinned `git clone` (commit `bf32082`). For other chemistries: retrain via the bundled `Novel-Space/` pipeline |
| Weights license | MIT (per upstream LICENSE) |
| Container stack | python:3.11-slim + TensorFlow >=2.16 (CPU) + `autoXRD` (installed from cloned repo at pinned SHA `bf32082`, version 0.0.6 to match Example/Model.h5) + pymatgen + pyxtal + scipy + scikit-image |
| H100 status | N/A (CPU runtime by design; autoXRD inference is ~10 s/pattern on CPU and parallelises on `general-cpu` job arrays better than it scales up a single GPU) |
| Lab status | **utility** — multi-phase XRD ID toolkit; bundled demo is chemistry-specific (Li battery cathodes), so general mineralogy use requires retraining on user CIFs |
| Architecture | **AMD64-only.** Upstream's prediction pipeline calls into BGMN (Rietveld refinement, Linux x86_64 binary). No arm64 path exists. Apple-Silicon Mac users run via `docker pull --platform linux/amd64` + qemu |
| First-run / current behavior | v2 build smoke test passes (2026-05-02); BGMN baked at build time; bundled demo `Example/run_CNN.py` runs offline on first pull |
| Tags | `:v2` (= `:latest`, `:autoxrd-tf2.16-cpu`) — current; `:v1` retained for rollback (had the BGMN runtime-fetch bug + version-skew between pip and cloned autoXRD) |

## prithvi-eo

| | |
|--|--|
| Task | Geospatial foundation model — ViT pre-trained on HLS for downstream burn-scar / flood / crop classification / segmentation; container ships TerraTorch's `BACKBONE_REGISTRY` + Lightning task scaffolding |
| Sensor | image:multi (Harmonized Landsat-Sentinel-2 — 6 bands B2/B3/B4/B5/B6/B7, multi-temporal) |
| Upstream repo | [IBM/terratorch](https://github.com/IBM/terratorch) (the toolkit); model weights at [`ibm-nasa-geospatial`](https://huggingface.co/ibm-nasa-geospatial) on HF Hub |
| Upstream license | Apache-2.0 (TerraTorch); Apache-2.0 (Prithvi weights — verify per HF model card) |
| Paper | Roy, Carney, Castaldi, et al. (2024) — *Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications*, [arXiv:2412.02732](https://arxiv.org/abs/2412.02732). Earlier 1.0 paper Jakubik et al. (2023), arXiv:2310.18660 |
| Weights source | HF Hub: [`ibm-nasa-geospatial/Prithvi-EO-1.0-100M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M), [`Prithvi-EO-2.0-300M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M), [`Prithvi-EO-2.0-300M-TL`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL), [`Prithvi-EO-2.0-600M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M), [`Prithvi-EO-2.0-600M-TL`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL). Cache at `$HF_HOME=/opt/hf-cache` for bind-mount persistence |
| Weights license | Apache-2.0 |
| Container stack | nvidia/cuda:12.1.0-cudnn8 + python 3.11 + PyTorch 2.5.1 + torchvision 0.20.1 (cu121) + `terratorch>=1.2.5` + Lightning + torchgeo + segmentation-models-pytorch + diffusers + timm + geopandas |
| H100 status | Native sm_90 |
| Lab status | **utility** — pretrained backbones; downstream task heads + fine-tuning required for any actual prediction. Recommended tier: Compute2 H100 for the 300M/600M models |
| First-run / current behavior | Build smoke test passes (2026-05-01); `BACKBONE_REGISTRY` reachable; no production embedding output yet |
| Tags | `:v1` (= `:latest`, `:torch2.5-cu121`) |

## treex

| | |
|--|--|
| Task | Tree-instance segmentation — unsupervised / classical, multi-platform (TLS / PLS / ULS) |
| Sensor | lidar:tls,pls,uls (point clouds in LAS / LAZ / PLY) |
| Upstream repo | [ai4trees/pointtree](https://github.com/ai4trees/pointtree) (Python package `pointtree`, author Josafat-Mattias Burmeister) |
| Upstream license | MIT |
| Paper | Burmeister, Tockner, Reder, Engel, Richter, Mund, Döllner (2025), *treeX: Unsupervised Tree Instance Segmentation in Dense Forest Point Clouds*, [arXiv:2509.03633](https://doi.org/10.48550/arXiv.2509.03633) |
| Weights source | None (classical / unsupervised algorithm; no learned components) |
| Weights license | N/A |
| Container stack | python:3.11-slim + numpy>=2.3 + pointtree==1.0.1 (PyPI; pybind11 + scikit-build-core compile C++ extensions at install) + pointtorch + circle_detection (force-reinstalled from upstream `main` to mirror the upstream Dockerfile) + cloth-simulation-filter + pyclesperanto-prototype<0.24.5 + numba + rasterio + pygam + scikit-learn |
| H100 status | N/A (CPU-only by design; `TreeXAlgorithm` is the unsupervised path and does not call torch) |
| Lab status | **experimental** — first end-to-end run scheduled against Tyson UAV `tile_-10_10` (89 pts/m², 100×100 m, leaf-on closed-canopy hardwood) on Compute2 `general-cpu`. Reported ULS F1 = 0.58 on Wytham + FOR-instance (Burmeister et al. 2025); expect noticeable recall gaps on suppressed stems |
| First-run / current behavior | Smoke test passes at build time (`TreeXAlgorithm` + `TreeXPresetULS` instantiate); no production output yet |
| Tags | `:v1` (= `:latest`) |
| Notes | Container supports only the `TreeXAlgorithm` (unsupervised) path. The companion `CoarseToFineAlgorithm` from the same package needs torch + torch-scatter + a learned semantic-segmentation checkpoint and would require a separate, much larger container variant |

## raman-classifier

| | |
|--|--|
| Task | Raman mineral identification via nearest-neighbour matching against the RRUFF reference library |
| Sensor | Raman spectrum (1D, 2-column wavenumber/intensity text input) |
| Upstream repo | [barahona-research-group/RamanSPy](https://github.com/barahona-research-group/RamanSPy) |
| Upstream license | BSD-3 (ramanspy code). RRUFF reference data: cite Lafuente et al. 2015 — no explicit Creative Commons license posted by the project |
| Paper | Georgiev, Pedersen, Xie, Fern, Barahona (2024), *Anal. Chem.* — *RamanSPy: An Open-Source Python Package for Integrative Raman Spectroscopy Data Analysis*, [doi:10.1021/acs.analchem.4c00383](https://doi.org/10.1021/acs.analchem.4c00383). Reference data: Lafuente B, Downs RT, Yang H, Stone N (2015). *The power of databases: the RRUFF project*. In: Highlights in Mineralogical Crystallography, T Armbruster & RM Danisi, eds., De Gruyter, Berlin, 1-30 |
| Weights source | None (classical algorithm). Reference library is RRUFF `excellent_unoriented` (~229 MB raw archive), preprocessed at build time and baked as a single ~30-50 MB numpy index at `/opt/rruff_index.npz` |
| Weights license | N/A. RRUFF reference spectra are redistributed in preprocessed numerical form; downstream users must cite Lafuente et al. 2015 |
| Container stack | python:3.11-slim + numpy>=1.26,<2.3 + scipy>=1.11 + ramanspy>=0.2 (BSD-3) |
| H100 status | N/A (CPU-only; single-spectrum match is sub-second after index load) |
| Lab status | **utility** — Path A of the long-deferred raman-classifier slot. Path B (Liu-2017-style 1D-CNN trained on RRUFF, weights deposited at Zenodo + HF Hub under Apache-2) remains queued |
| Architecture | **Multi-arch** — `linux/amd64` + `linux/arm64`. ramanspy and its scientific-Python dependencies all publish aarch64 wheels |
| First-run / current behavior | Build smoke test passes at build time (ramanspy public API reachable + RRUFF index built and shape-validated); real-data validation pending — see PR review |
| Tags | `:v1` (= `:latest`, `:rruff-excellent-cpu`) |
| Notes | Index covers the 100-1500 cm⁻¹ fingerprint region at 1 cm⁻¹ resolution. OH/H₂O stretch peaks (3000-3700 cm⁻¹) and other RRUFF archives (`fair_unoriented`, `excellent_oriented`, `poor_unoriented`, `unrated_*`, `LR-Raman`) are excluded by default to keep the image lean; rebuild with extra `--dataset` and/or `--wavenumber-max` flags to widen coverage |

## geoclip

| | |
|--|--|
| Task | Worldwide image geolocalization — given an RGB photo, return top-k predicted (lat, lon) locations and probabilities |
| Sensor | Image:rgb (anything Pillow reads at ~224×224 after CLIP preprocessing) |
| Upstream repo | [VicenteVivan/geo-clip](https://github.com/VicenteVivan/geo-clip) |
| Upstream license | MIT |
| Paper | Vivanco Cepeda, Nayak, Shah (2023), *NeurIPS* — *GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization*, [arXiv:2309.16020](https://arxiv.org/abs/2309.16020) |
| Weights source | Hugging Face Hub (fetched on `GeoCLIP(from_pretrained=True)` instantiation). Backbone is CLIP ViT-L/14 image encoder + small MLP location encoder + 100K-point GPS gallery (`coordinates_100K.csv` bundled with the pip package). Baked into image at `$HF_HOME=/opt/hf-cache` during build, so runtime is offline |
| Weights license | Per upstream repo (MIT-aligned); verify before redistribution |
| Container stack | python:3.11-slim + PyTorch 2.5.1 + torchvision 0.20.1 (CPU wheels) + `geoclip>=1.2` + `huggingface_hub>=0.25` + Pillow |
| H100 status | N/A (CPU runtime by design; model is small enough that GPU adds no value for typical one-shot use) |
| Lab status | **utility** — geo-tagged photo QA, locating photos with stripped EXIF, provenance / deduplication |
| Architecture | **Multi-arch** — `linux/amd64` + `linux/arm64`. All deps publish aarch64 wheels |
| First-run / current behavior | Build smoke test passes at build time (instantiates GeoCLIP, validates `gps_gallery.shape == (100000, 2)`); weights are baked at build time so runtime is offline-capable |
| Tags | `:v1` (= `:latest`, `:torch2.5-cpu`) |
| Notes | Differs from `remoteclip:v2` in that GeoCLIP weights are baked into the image at build time (~900 MB); `remoteclip` keeps weights at runtime via mounted HF cache. Pragmatic choice for one-shot photo QA vs. batch embedding workflows. Predictions over the fixed 100K-point gallery — for sub-kilometre accuracy, use `top_k_radius` mode (not exposed in v1) |

## dofa

| | |
|--|--|
| Task | Multispectral / SAR / optical / hyperspectral foundation model — embedding extraction (no shipped task head) |
| Sensor | Image:multispectral (Sentinel-2, Landsat, Gaofen), SAR (Sentinel-1), RGB (NAIP), hyperspectral. Wavelengths supplied at runtime |
| Upstream repo | [zhu-xlab/DOFA](https://github.com/zhu-xlab/DOFA); torchgeo loader at [microsoft/torchgeo](https://github.com/microsoft/torchgeo) |
| Upstream license | CC-BY-4.0 (torchgeo `dofa_*` weights) |
| Paper | Xiong, Z. et al. (2024) *Neural Plasticity-Inspired Foundation Model for Observing the Earth Crossing Modalities* — [arXiv:2403.15356](https://arxiv.org/abs/2403.15356) |
| Weights source | Hugging Face Hub: [`torchgeo/dofa`](https://huggingface.co/torchgeo/dofa). Base (445 MB, 768-D embeddings, ~111M params) baked at build time; Large (1.35 GB, 1024-D embeddings, ~336M params) fetched lazily on `--variant large`. Cache at `$TORCH_HOME=/opt/torch-cache` |
| Weights license | CC-BY-4.0 (per HF model card) |
| Container stack | python:3.11-slim + PyTorch 2.5.1 + torchvision 0.20.1 (CPU wheels) + `torchgeo>=0.6` + `timm>=1.0` + `huggingface_hub>=0.25` |
| H100 status | N/A in v1 (CPU runtime; GPU variant deferred until a batch-embedding workload lands) |
| Lab status | **utility** — embeddings only, no task head; useful for downstream classification / change-detection / retrieval workflows |
| Architecture | **Multi-arch** — `linux/amd64` + `linux/arm64`. All deps publish aarch64 wheels |
| First-run / current behavior | Build smoke test passes at build time (Base instantiates with DOFA_MAE weights, synthetic 12-band S2-shaped input → embedding shape (1, 768) verified) |
| Tags | `:v1` (= `:latest`, `:torch2.5-cpu`) |
| Notes | Wavelength-conditioning hypernetwork: user must pass per-band wavelengths in micrometers at inference. Convenience flags `--sentinel2-{12,10}band`, `--sentinel1`, `--naip-rgb` cover common configurations; `--wavelengths` for arbitrary lists. Embedding-only — task heads are downstream user responsibility. **For the text-aligned variant** see `bradleylab/dofa-clip` (separate container, CC-BY-NC-4.0) |

## dofa-clip

| | |
|--|--|
| Task | Multispectral / RGB image-text retrieval and zero-shot scene scoring (CLIP architecture; SigLIP text encoder; DOFA wavelength-conditioned image trunk) |
| Sensor | Image:multispectral (Sentinel-2 / Sentinel-1 / Gaofen / hyperspectral via wavelength conditioning), or RGB. Wavelengths supplied at runtime |
| Upstream repo | [xiong-zhitong/DOFA-CLIP](https://github.com/xiong-zhitong/DOFA-CLIP) (vendored open_clip fork) |
| Upstream license | Apache-2.0 (code, this repo + xiong-zhitong/DOFA-CLIP) |
| Paper | Xiong et al. (2025) *DOFA-CLIP: Vision-Language Foundation Model for Earth Observation* — [arXiv:2503.06312](https://arxiv.org/abs/2503.06312) |
| Weights source | Hugging Face Hub: [`earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO`](https://huggingface.co/earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO) (~1.7 GB safetensors) baked at build time via the vendored open_clip's `create_model_from_pretrained("hf-hub:...")` |
| Weights license | **CC-BY-NC-4.0** per HF model card (non-commercial only). Commercial use requires explicit upstream permission (`xiongzhitong@gmail.com`). The only NC-licensed image in the catalog as of v1 |
| Container stack | python:3.11-slim + PyTorch 2.5.1 + torchvision 0.20.1 (CPU wheels) + vendored open_clip from `xiong-zhitong/DOFA-CLIP` (Apache-2.0) + `timm` + `einops` + `transformers>=4.40,<5` + `huggingface_hub<1.0` |
| H100 status | N/A in v1 (CPU runtime; GPU variant deferred until a batch-screening workload lands) |
| Lab status | **utility** — multispectral CLIP, sister to `remoteclip` (RGB-only Apache-2.0) and `dofa` (multispectral embedding-only CC-BY-4.0) |
| Architecture | **Multi-arch** — `linux/amd64` + `linux/arm64`. All deps publish aarch64 wheels |
| First-run / current behavior | Build smoke test passes at build time: model loads, text embeddings differentiate (asserts pairwise cosine < 0.95), image-text scoring on the upstream airplane.png correctly puts "a busy airport" above "a forest" / "a stadium". Catches the Path A failure mode that the BiliSakura HF mirrors collapse the text encoder |
| Tags | `:v1` (= `:latest`, `:torch2.5-cpu`) |
| Notes | **Path B build** — uses the upstream xiong-zhitong/DOFA-CLIP repo's vendored open_clip fork. **Path A** (HF transformers via `BiliSakura/DOFA-CLIP-{ViT-B-16,VIT-L-14}` mirrors) was evaluated and is broken: text encoder self-attention stored as `in_proj.{weight,bias}` is silently dropped by HF's `CLIPModel`, leaving every text attention layer randomly initialized; text embeddings collapse to ~identical vectors across prompts. See README for details. Output dim 1152, image res 384×384, text context length 64. SigLIP-style scoring (sigmoid not softmax) — per-prompt independent |

## terramind

| | |
|--|--|
| Task | Any-to-any generative geospatial foundation model — embeddings, segmentation, cross-modality translation (e.g. S1 → NDVI when S2 cloud-blocked); supports Thinking-in-Modalities fine-tuning |
| Sensor | image:multi (S1 GRD, S1 RTC, S2 L2A, DEM, NDVI, LULC). Six tokenizers under same HF org |
| Upstream repo | [IBM/terramind](https://github.com/IBM/terramind) (config + notebooks); model code in [terrastackai/terratorch](https://github.com/terrastackai/terratorch) `BACKBONE_REGISTRY` |
| Upstream license | Apache-2.0 (terramind config + terratorch toolkit) |
| Paper | Jakubik et al. (2025), *ICCV 2025* — *TerraMind: Large-Scale Generative Multimodality for Earth Observation*, [arXiv:2504.11171](https://arxiv.org/abs/2504.11171) |
| Weights source | HF Hub: [`ibm-esa-geospatial/TerraMind-1.0-tiny`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny), [`-small`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small), [`-base`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base), [`-large`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large). Cache at `$HF_HOME=/opt/hf-cache` for bind-mount persistence |
| Weights license | Apache-2.0 (per HF model cards) |
| Container stack | nvidia/cuda:12.1.0-cudnn8 + python 3.11 + PyTorch 2.5.1 + torchvision 0.20.1 (cu121) + `terratorch>=1.2.5` + `diffusers==0.30.0` (TerraMind any-to-any pin) + `setuptools<81` |
| H100 status | Native sm_90 |
| Lab status | **utility** — pretrained backbone; downstream task heads + fine-tuning required for any actual prediction. Recommended tier: Compute2 H100 for the base/large variants |
| First-run / current behavior | Build smoke test passes (terratorch + diffusers import; ≥1 `terramind_*` backbone registered in `BACKBONE_REGISTRY`); no production embedding output yet |
| Tags | `:v1` (= `:latest`, `:torch2.5-cu121`) |
| Notes | Sister container to `prithvi-eo` (both TerraTorch-fronted). TerraMind covers the multimodal S1+S2+DEM+NDVI+LULC pretraining; Prithvi-EO is HLS-only. The `_tim` backbone variants enable Thinking-in-Modalities fine-tuning (the model first generates a missing modality before predicting the downstream task). For any-to-any modality generation, the `diffusers==0.30.0` pin is load-bearing — newer diffusers break the upstream generation pipeline |

## timesfm

| | |
|--|--|
| Task | Univariate time-series forecasting — zero-shot point + continuous-quantile predictions; LoRA fine-tuning via HF Transformers + PEFT |
| Sensor | time_series:univariate (any 1D regularly-sampled signal — streamflow, soil moisture, climate-reanalysis pixel-time-series, eddy-covariance fluxes, met-station observations) |
| Upstream repo | [google-research/timesfm](https://github.com/google-research/timesfm) |
| Upstream license | Apache-2.0 |
| Paper | Das, Kong, Sen, Zhou (2024), *ICML 2024* — *A decoder-only foundation model for time-series forecasting*, [arXiv:2310.10688](https://arxiv.org/abs/2310.10688) |
| Weights source | HF Hub: [`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (canonical, via `timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)`) and [`google/timesfm-2.5-200m-transformers`](https://huggingface.co/google/timesfm-2.5-200m-transformers) (HF Transformers integration for LoRA fine-tuning). Cache at `$HF_HOME=/opt/hf-cache` for bind-mount persistence |
| Weights license | Apache-2.0 (per HF model cards) |
| Container stack | python:3.11-slim + PyTorch 2.5.1 (CPU wheels) + `timesfm` installed from `google-research/timesfm` at pinned commit `d720daa67865` (2026-04-15). PyPI's `timesfm 1.3.0` is the v1/v2 archive — TimesFM 2.5 is GitHub-only |
| H100 status | N/A in v1 (CPU runtime by design; 200M-param model + Apache-2.0 weights mean short-horizon inference runs comfortably on a laptop). GPU variant deferred until a panel-of-thousands-of-series workload lands |
| Lab status | **utility** — different modality from the rest of the catalog (time-series, not imagery); fits hydrology, soil moisture, climate reanalysis, eddy-covariance gap-filling. Sister to `neuralhydrology`: TimesFM is the zero-shot fallback when there isn't enough history to fine-tune a CAMELS-style LSTM |
| Architecture | **Multi-arch** — `linux/amd64` + `linux/arm64`. All deps publish aarch64 wheels |
| First-run / current behavior | Build smoke test passes (`timesfm` import, `TimesFM_2p5_200M_torch` + `ForecastConfig` resolve); no production forecast output yet |
| Tags | `:v1` (= `:latest`, `:torch2.5-cpu`) |
| Notes | Decoder-only architecture, 200M parameters, supports up to 16k context length, optional 30M continuous-quantile head for probabilistic forecasts (sigmoid-style not softmax). 2.5 release (Sept 2025) drops the v2.0 frequency indicator and bumps context from 2048 to 16k. Closes the highest-priority Tier 1 wishlist candidate from STATUS.md per the 2026-05-07 prior-art triage |

---

## Deprecated images

For history of `bradleylab/multispec-species` and `bradleylab/tree-analysis`, see [`DEPRECATIONS.md`](DEPRECATIONS.md).
