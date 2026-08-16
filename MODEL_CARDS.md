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

## dinov3-sat

| | |
|--|--|
| Task | Dense feature extraction — frozen backbone for downstream segmentation / classification heads |
| Sensor | RGB aerial or satellite orthoimagery (pretrained at 0.6 m GSD; runs at any GSD) |
| Upstream repo | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) |
| Upstream license | **DINOv3 License — not open source.** Redistribution permitted only with a copy of the Agreement (shipped at `/opt/licenses/LICENSE.dinov3.md`); publications must acknowledge DINO Materials; trade-control + no-military/ITAR terms |
| Paper | Simeoni, Vo, Oquab et al. (2025), *DINOv3* |
| Weights source | `timm/vit_large_patch16_dinov3.sat493m`, **baked at build**. The `facebook/` repo of the same weights is manually gated; the timm mirror is not |
| Weights license | DINOv3 License (as above) |
| Container stack | PyTorch 2.5.1 + CUDA 12.1; `timm==1.0.28`; ViT-L/16, 1024-D |
| H100 status | Native sm_90; amd64 only (CUDA base has no arm64 build) |
| Lab status | **utility** — general RGB-ortho encoder; first consumer is the nc-helene gravel-bar head |
| First-run / current behavior | Loading, token geometry and feature grid verified against real 0.152 m aerial chips (2026-08-11): 512 px -> (32, 32, 1024). Note 5 prefix tokens (CLS + 4 registers) and SAT-specific normalization — both handled by `extract_features.py`, both asserted in the build smoke test |
| Tags | `:latest`, `:v1`, `:torch2.5-cu121` |

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
| First-run / current behavior | **Real-input validated on Compute2** (job 712196, A100 80GB PCIe, 1m33s): 12-band S2L2A-shaped cube from real RGB → 87.3M-param `terramind_v1_base` from HF Hub → 12-layer ViT output (1, 196, 768), real vs synth distance 704.27. Backbone resolves under unprefixed `terramind_v1_base` (no prefix needed). See `geospatial-containers/terramind_test/` |
| Tags | `:v1` (= `:latest`, `:torch2.5-cu121`) |
| Notes | Sister container to `prithvi-eo` (both TerraTorch-fronted). TerraMind covers the multimodal S1+S2+DEM+NDVI+LULC pretraining; Prithvi-EO is HLS-only. The `_tim` backbone variants enable Thinking-in-Modalities fine-tuning (the model first generates a missing modality before predicting the downstream task). For any-to-any modality generation, the `diffusers==0.30.0` pin is load-bearing — newer diffusers break the upstream generation pipeline |

## momo

| | |
|--|--|
| Task | Multi-task Mars surface analysis — ViT pretrained on HiRISE / CTX / THEMIS. Evaluated on 9 downstream tasks from the separate Mars-Bench benchmark (arXiv 2510.24010, NeurIPS 2025; 20 datasets spanning classification, segmentation, and object detection; task data at `Mirali33/mars-bench-*` on HF) — e.g. crater segmentation, boulder detection, dust-devil tracking, S5Mars rover surface, DoMars16k landmark classification. This container ships the MOMO backbone + fine-tuning engine, not the benchmark datasets |
| Sensor | image:multi-resolution (Mars orbital — HiRISE 0.25 m/px, CTX 5 m/px, THEMIS 100 m/px) |
| Domain | `planetary-mars` (label `bradleylab.model.domain`) — the catalog's Mars group; any future Mars container should carry the same label |
| Upstream repo | [kerner-lab/MOMO](https://github.com/kerner-lab/MOMO) |
| Upstream license | MIT (per upstream LICENSE) |
| Paper | Model: *MOMO: Mars Orbital Model Foundation Model for Mars Orbital Applications* ([arXiv:2604.02719](https://arxiv.org/abs/2604.02719)). Benchmark: *Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks* ([arXiv:2510.24010](https://arxiv.org/abs/2510.24010), NeurIPS 2025) |
| Weights source | HF Hub: [`Mirali33/MOMO`](https://huggingface.co/Mirali33/MOMO) — single multi-sensor checkpoint + three sensor-specific (HiRISE / CTX / THEMIS), in ViT-Small / ViT-Base / ViT-Large variants. Cache at `$HF_HOME=/opt/hf-cache` for bind-mount persistence |
| Weights license | CC-BY-4.0 (per HF model card [`Mirali33/MOMO`](https://huggingface.co/Mirali33/MOMO)) |
| Container stack | nvidia/cuda:12.1.0-cudnn8 + python 3.11 + PyTorch 2.5.1 + torchvision 0.20.1 (cu121) + `kerner-lab/MOMO` at pinned commit `a837ab5` (full upstream `requirement.txt`: pytorch-lightning, hydra-core, segmentation-models-pytorch, albumentations, rasterio, shapely, scikit-image, scikit-learn, imbalanced-learn, timm 0.6.12, einops, lpips, lxml). MOMO installed with `--no-deps` to work around upstream's unsatisfiable `timm==0.6.12 + smp>=0.5.0` pin (smp 0.5.0+ requires `timm>=0.9`); verified safe by reading MOMO's only timm imports (`PatchEmbed`, `Block`, stable across timm 0.6.x → 1.x) |
| H100 status | Native sm_90 |
| Lab status | **utility / experimental** — pretrained backbone for Mars surface tasks; downstream task heads + fine-tuning required for production prediction. Recommended tier: Compute2 H100 |
| First-run / current behavior | Compute2 A100 validated 2026-05-03 (PR #31): real HiRISE Mars surface (DoMars16k sample), ViT-B 86.6M loads from multi-sensor checkpoint, `forward_features` produces (1, 768) CLS embedding. 2 missing keys (`head.weight`, `head.bias` — encoder-only build) and 104 unexpected keys (`mask_token`, `decoder_*` — MAE-specific decoder) are both expected for a downstream-feature-extraction setup. ‖real-synth‖=44 |
| Tags | `:v1` (= `:latest`, `:torch2.5-cu121`) |
| Notes | Different domain from rest of catalog (planetary, not Earth-observation). Useful as a teaching example of how foundation models from Earth-orbital pretraining transfer (or don't) to off-Earth surface imagery. Future variants could build pre-fine-tuned heads against the external Mars-Bench tasks |

## dbloops

| | |
|--|--|
| Task | Density-based 3D point-cloud clustering for grain-size distribution + boulder mapping. Two-pass DBSCAN over local 3D neighbourhood with epsilon scaling (`np`, `esfa`, `esfb`). Detector-only — does NOT bundle the upstream Random Forest clast/matrix classifier (Terpunkto) |
| Sensor | lidar:tls,mls (high-density gravel-bed point clouds — point spacing well under 1 cm) |
| Upstream repo | [haydenjgeo/DBloops](https://github.com/haydenjgeo/DBloops) at v1.0.0 (commit `c3acb15`) |
| Upstream license | MIT (DBloops MATLAB code) |
| Paper | Jacobson et al. (2025) — "DBloops: density-based loop scaling for grain-size distribution from 3D point clouds" (DOI TBD) |
| Weights source | None (classical algorithm; no learned components). Bundled source: `DBloops/`, `Terpunkto/`, `G3point/` from upstream tag, with backslash-path patches applied (eight one-line edits; see `bradleylab/rock_glaciers/scripts/matlab/dbloops_patches.diff`) |
| Weights license | N/A |
| Container stack | MATLAB Compiler R2024b standalone binary against the free MATLAB Runtime. Wrapper: env-driven `run_dbloops_patch.m` reading `PATCH_XYZ`, `PATCH_OUT`, `NP_VAL`, `ESFA`, `ESFB` |
| H100 status | N/A (CPU-only; MATLAB Compiler binary) |
| Lab status | **experimental** — DBloops is the clustering step only; without the Random Forest classifier the detections at typical TLS densities (11–100 pts/m²) do not correspond to real boulders. The right deployment is high-density gravel-bed lidar matching Jacobson's training data. See `bradleylab/rock_glaciers/EXPERIMENTS.md` v3a for the visual-audit failure on rock-glacier surfaces |
| First-run / current behavior | Build smoke test passes (clusterable synthetic data, PR #35). Real-input audit on rock-glacier TLS (`bradleylab/rock_glaciers`) shows the density-mismatch failure mode. Also includes 2026-05-08 doc warning against `ENROOT_TEMP_PATH` under `/storage1/.../Active/` (PR #43) |
| Tags | `:v1` (= `:latest`) |
| Notes | Different domain from rest of catalog (geomorphology, not forest or imagery). MATLAB Runtime is licensed under the Mathworks runtime EULA; the binary inside this image was compiled from upstream MIT-licensed MATLAB code under WashU's TAH license. The resulting standalone binary is redistributable to anyone with a MATLAB Runtime install (free) |

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

## crossearth

| | |
|--|--|
| Task | Cross-domain semantic segmentation — domain-generalizable RSDG; trains on source domains and runs zero-shot on unseen target domains differing in region, resolution, spectral bands, or climate. Frozen DINOv2 backbone + Mask2Former head |
| Sensor | image:multi (optical RS, region/resolution/spectral domain transfer) |
| Upstream repo | [VisionXLab/CrossEarth](https://github.com/VisionXLab/CrossEarth) (vendored at SHA `644a5a1b3c01b2e5531820b5291d4397597f75de`, HEAD as of 2026-04-02) |
| Upstream license | MIT |
| Paper | Gong et al. (2025), *TPAMI 2025* — *CrossEarth: Geospatial Vision Foundation Model for Domain Generalizable Remote Sensing Semantic Segmentation*, [arXiv:2410.22629](https://arxiv.org/abs/2410.22629) |
| Weights source | HF Hub: [`Cusyoung/CrossEarth`](https://huggingface.co/Cusyoung/CrossEarth) (`dinov2_converted.pth`, `dinov2_converted_1024x1024.pth`). Note typo: HF org is `Cusyoung`, not author handle `Cuzyoung`. Cache at `$HF_HOME=/opt/hf-cache` |
| Weights license | Per HF model card — verify before redistribution |
| Container stack | `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` (amd64-only) + Python 3.10 + `mmengine` + `mmcv>=2.0.0,<2.2` + `mmsegmentation>=1.0.0,<1.3` + `mmdet>=3.0.0,<3.4` + `xformers==0.0.20` (exact pin per upstream — newer breaks DINOv2 attention) |
| H100 status | Native sm_90 (TORCH_CUDA_ARCH_LIST covers 7.0–9.0); CUDA 11.7 stack |
| Lab status | **utility** — pretrained backbone + segmentation head; out-of-the-box inference on supported RSDG benchmarks. Recommended tier: any GPU with ≥8 GB VRAM for inference |
| First-run / current behavior | **Real-input validated on Compute2** (job 729263, A100 80GB PCIe, 1m10s): real RGB → 512×512 → 307.2M-param `ReinsDinoVisionTransformer` (DINOv2-Large + LoRAReins) with `dinov2_converted.pth` (1.22 GB) loaded from HF Hub `Cusyoung/CrossEarth` → 4 multi-scale features (1, 1024, 32, 32) + 100 query tokens (100, 256), real vs synth distance 93.62. See `geospatial-containers/crossearth_test/` |
| Tags | `:v1` (= `:latest`, `:torch2.0-cu117`) |
| Notes | Sister to `dofa`/`dofa-clip`/`terramind` in the RS foundation-model cluster, but addresses the domain-generalization problem differently: instead of spectral conditioning (DOFA) or multimodal pretraining (TerraMind), CrossEarth uses Earth-Style Injection (data-level augmentation) + multi-task training over a 32-scenario RSDG benchmark. Only RS FM in the catalog built on a generalist self-supervised vision backbone (DINOv2). Closes the second remaining Tier 2 wishlist candidate per 2026-05-07 prior-art triage. Upstream is a research codebase (not pip-installable) — vendored at HEAD. mmcv 2.x + mmseg 1.x + xformers 0.0.20 + torch 2.0 stack mirrors upstream conda recipe; newer torch likely works but isn't validated |

## croma

| | |
|--|--|
| Task | Sentinel-1/Sentinel-2 radar-optical foundation model — embedding extraction (no shipped task head) |
| Sensor | Image:SAR (Sentinel-1, 2-ch VV/VH) + multispectral (Sentinel-2, 12-band, cirrus removed). Fixed channel layout |
| Upstream repo | [antofuller/CROMA](https://github.com/antofuller/CROMA) (`use_croma.py` pinned at commit `59505a6`) |
| Upstream license | MIT |
| Paper | Fuller, A., Millard, K. & Green, J. R. (2023) *CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders*, NeurIPS 2023 — [arXiv:2311.00566](https://arxiv.org/abs/2311.00566) |
| Weights source | Hugging Face Hub: [`antofuller/CROMA`](https://huggingface.co/antofuller/CROMA). Base (ViT-B, 768-D) baked at build time; Large (ViT-L, 1024-D) fetched lazily on `--variant large`. Cache at `$HF_HOME=/opt/hf-cache` |
| Weights license | MIT (per HF repo) |
| Container stack | python:3.11-slim + PyTorch 2.5.1 (CPU wheels) + `einops>=0.7` + `huggingface_hub>=0.25` (no torchvision) |
| H100 status | N/A in v1 (CPU runtime; GPU variant deferred until a batch-embedding workload lands) |
| Lab status | **utility** — embeddings only, no task head; intended for `nc-helene-sar` Stage-3 fusion (CROMA frozen embeddings → disturbance head vs LiDAR DoD) |
| Architecture | **Multi-arch** — `linux/amd64` + `linux/arm64`. CROMA needs only torch + einops; both publish aarch64 wheels |
| First-run / current behavior | Build smoke test passes (Base instantiates from baked weights; synthetic S1 2-ch + S2 12-ch 120×120 → SAR_GAP / optical_GAP / joint_GAP (1, 768), joint_encodings (1, 225, 768) verified) |
| Tags | `:v1` (= `:latest`, `:torch2.5-cpu`) |
| Notes | Sentinel-1/Sentinel-2-native with a fixed channel layout (no wavelength conditioning — cf. `bradleylab/dofa`). `--modality both\|SAR\|optical`; optical must drop B10/cirrus (12 bands). Embedding-only — task heads are downstream user responsibility |

## evo2

| | |
|--|--|
| Task | Genomic foundation model — DNA sequence scoring (log-likelihoods), variant-effect prediction, and genome generation across all domains of life |
| Sensor | `dna-sequence` (label `bradleylab.model.sensor`) — nucleotide sequence, not a sensor product |
| Upstream repo | [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2) |
| Upstream license | Apache-2.0 (code) |
| Paper | "Genome modeling and design across all domains of life with Evo 2" (Arc Institute, 2025). No DOI or arXiv ID is recorded in this repo — `evo2/README.md` defers to the upstream repo for the current citation |
| Weights source | HF Hub, [`arcinstitute`](https://huggingface.co/arcinstitute) org — `evo2_7b`, `evo2_40b`, `evo2_7b_base`, `evo2_1b_base`, `evo2_7b_262k`, … Not baked; fetched on first `Evo2('evo2_7b')` call into `$HF_HOME=/root/.cache/huggingface` (bind-mount for persistence) |
| Weights license | Per the HF model cards — verify before redistribution |
| Container stack | `nvcr.io/nvidia/pytorch:25.04-py3` (torch 2.7.0a0, CUDA 12.9, Transformer Engine 2.2, flash-attn, Python 3.12) + `evo2` + `biopython` pip packages. No separate TE / flash-attn install — the NGC base already ships what evo2 builds against |
| H100 status | Native sm_90 (base `TORCH_CUDA_ARCH_LIST` includes 9.0). `evo2_7b` / `evo2_1b_base` / `evo2_7b_base` run bf16 on a single H100 80 GB; `evo2_20b` / `evo2_40b` need FP8 via Transformer Engine on Hopper, and 40B needs multiple H100s — out of scope for the single-GPU C2 `general-gpu` allocation |
| Lab status | **experimental** — backfilled recipe, not yet benchmarked on lab data. The 7B path is the intended first target |
| First-run / current behavior | No real load test yet. The build-time smoke test is metadata-only (`importlib.metadata`) because `import evo2` needs a GPU at import time (Transformer Engine) and the CPU build runner has none; a real load test must run on an H100 |
| Tags | `:v1` (= `:latest`, `:torch2.7-cu129`); weights NOT baked |
| Notes | Backfilled recipe — reproduces a previously ad-hoc `ghcr.io/bradleylab/evo2` image that had no committed recipe, reconstructed from the published image's build history. Large image (~12 GB); the build workflow frees runner disk first, and disk exhaustion is the first thing to check on a CI failure. Upstream ships no CLI — inference is a Python-API script under scratch. This image carries NGC 25.04's TE 2.2 while the upstream README suggests TE 2.3.0 for the FP8 (20B/40B) path; a TE bump is the likely fix if the large models misbehave. With `esm`, one of the catalog's first molecular-sequence models; neither sets a `bradleylab.model.domain` label |

## esm

| | |
|--|--|
| Task | Protein language model — sequence embeddings and per-residue logits from amino-acid sequence alone, no structure required |
| Sensor | `protein-sequence` (label `bradleylab.model.sensor`) — amino-acid sequence, not a sensor product |
| Upstream repo | [Biohub/esm](https://github.com/Biohub/esm) — EvolutionaryScale became Chan Zuckerberg Biohub; the old `evolutionaryscale/esm` URL 301-redirects here |
| Upstream license | MIT |
| Paper | Preprint — "A world model of protein biology: ESMC, ESMFold2 & ESM Atlas" (Biohub, 2026), [biorxiv 2026.06.03.729735](https://www.biorxiv.org/content/10.64898/2026.06.03.729735) (per the `bradleylab.model.paper` label); `esm/README.md` defers to the upstream repo for the current citation |
| Weights source | HF Hub, [`biohub`](https://huggingface.co/biohub) org — `ESMC-300M` (333M, 1.33 GB), **`ESMC-600M` (575M, 2.30 GB — default)**, `ESMC-6B` (6.35B, 25.41 GB F32 across 6 shards), `esm3-sm-open-v1` (1.4B, 5.50 GB all components). Not baked; fetched on first `from_pretrained` call into `$HF_HOME=/root/.cache/huggingface` (bind-mount for persistence) |
| Weights license | MIT and **ungated** — an anonymous pull works, so Compute2 jobs need no HF token. A separate Acceptable Use Policy applies as a behavioral condition, not a weight restriction |
| Container stack | `nvcr.io/nvidia/pytorch:25.04-py3` (torch 2.7.0a0, CUDA 12.9, **Python 3.12**) + `esm` from `Biohub/esm` at pinned SHA `26b0bc2b` (2026-07-28) + Biohub's `transformers` fork at pinned SHA `ef32577f` (2026-06-08), force-reinstalled `--no-deps` over the resolved tree. Both SHAs are Docker ARGs recorded in the image labels (`bradleylab.build.esm_ref`, `bradleylab.build.transformers_ref`) |
| H100 status | Native sm_90. ESMC-300M / 600M are trivial (any GPU, or CPU); ESMC-6B fits a single H100 80 GB at F32 and bf16 halves it |
| Lab status | **experimental** — not yet benchmarked on lab data. Default target is batch embedding with ESMC-600M; ESMC-6B is worth the extra job time for high-value work |
| First-run / current behavior | No real load test yet. The build-time smoke test is metadata-only (`importlib.metadata`) because `import esm` reaches for GPU-only extensions (flash_attn) the CPU build runner lacks — same situation as evo2. It goes one step further than evo2's: it reads pip's `direct_url.json` for both git packages and asserts the installed commit IDs match the ARGs, so a silently-drifted pin fails the build rather than shipping. A real load test must run on an H100 |
| Tags | `:v1` (= `:latest`, `:torch2.7-cu129`); weights NOT baked |
| Notes | **Python 3.12 is a hard pin** — esm declares `requires-python = ">=3.12,<3.13"`, so pip fails outright on 3.11 or 3.13; that is why the NGC base is used rather than a `pytorch/pytorch:*-runtime` tag (those have historically shipped 3.11). **Do not add a pinned upstream `transformers`** or merge this environment with one carrying stock transformers — the Biohub fork is what provides the ESMC model class. Local inference goes through that fork, not an `esm.*` class; `esm.sdk` targets the hosted Biohub Platform API and needs a key. The large ESM3 checkpoints (7B, 98B) are API-only and have never been downloadable. The pinned commit reports `esm 3.3.0` in package metadata while the newest GitHub tag is `v3.2.2.post2` — upstream's state, not a packaging error. With `evo2`, one of the catalog's first molecular-sequence models; neither sets a `bradleylab.model.domain` label |

## clean

| | |
|--|--|
| Task | Enzyme Commission (EC) number prediction from protein sequence — ESM-1b embedding projected through a contrastively-trained network, assigned by distance to pre-computed EC cluster centres. One sequence can receive several EC numbers |
| Sensor | `protein-sequence` (label `bradleylab.model.sensor`) — amino-acid sequence, not a sensor product |
| Upstream repo | [tttianhao/CLEAN](https://github.com/tttianhao/CLEAN) — last commit 2025-04-06, last release v1.0.1 (2023-03-31). Low activity, not archived |
| Upstream license | **Research use only — NOT MIT.** GitHub repository metadata advertises MIT, but the tree contains **no LICENSE file**; the only licence artifact upstream ships is `NON-EXCLUSIVE RESEARCH USE LICENSE FOR CLEAN SOFTWARE.pdf`. The two cannot both be true, so the image is labelled `LicenseRef-CLEAN-Non-Exclusive-Research-Use` + `bradleylab.model.use_restriction="research-only"`, deliberately **not** MIT. University research is accepted as within terms; a commercial pipeline, a service to third parties, or redistribution under a permissive licence is not established and needs the authors' written agreement |
| Paper | Yu et al. (2023), *Science* 379:1358 — [doi:10.1126/science.adf2465](https://doi.org/10.1126/science.adf2465) |
| Weights source | **NOT baked** — two independent downloads, neither from a package registry. (1) CLEAN pretrained weights + EC cluster centres + GMM, ~141 MB zipped, **Google Drive only**, mounted at `/opt/CLEAN/app/data/pretrained` (must contain at least `split100.pth`, `100.pt`, `gmm_ensumble.pkl` — upstream's spelling — for the default max-separation path). (2) ESM-1b checkpoint + contact-regression head, ~7.3 GB from `dl.fbaipublicfiles.com`, into `$TORCH_HOME=/root/.cache/torch/hub/checkpoints`. **Unresolved:** upstream cites two conflicting Drive file IDs for the same bundle (`1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ` in its README, `1gsxjSf2CtXzgW1XsennTr-TcvSoTSDtk` in its Dockerfile) and publishes no checksum or version string for either — record the file ID and sha256 alongside whatever results it produced. Mirror both artifacts to Storage3 with `sha256` manifests before any analysis depends on them |
| Weights license | **Not stated anywhere.** The research-use PDF addresses the software, not the weights; treat the weights as research-use-only by the same reading |
| Container stack | python:3.10-slim (upstream's manuscript environment was Python 3.10.4) + torch 2.5.1+cpu + `fair-esm 1.0.2` + upstream's `requirements.txt` pins (numpy 1.22.3, pandas 1.4.2, scipy 1.7.3, scikit-learn 1.2.0, matplotlib 3.7.0, tqdm 4.64.0) + a `facebookresearch/esm` clone at tag `v1.0.2` in `/opt/CLEAN/app/esm` (load-bearing, not a duplicate of the pip package — CLEAN shells out to `./esm/scripts/extract.py` by relative path). Installed with upstream's legacy `python build.py install` under a `setuptools<85` ceiling. Pinned SHAs: `tttianhao/CLEAN` `f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15`, `facebookresearch/esm` `839c5b82c6cd9e18baa7a88dcbed3bd4b6d48e47`; both written into the image and asserted by the smoke test, so a drifted pin fails the build |
| H100 status | **N/A — CPU-only by design.** The image ships a CPU-only torch build and asserts it at build time. Documented resource floor is **>12 GB of system RAM**, not VRAM (the "7.3 GB" in upstream's README is the ESM-1b download size). CLEAN calls `torch.cuda.is_available()` and will use a GPU if one is visible, so a GPU allocation would not fail — it would just occupy an H100 for a job that does not need one. Run on Compute2 `general-cpu` |
| Lab status | **experimental** — not yet benchmarked on lab data |
| First-run / current behavior | **Not built or run.** The image has not been built on this machine (Docker unavailable), so CI is the first real build; there is no smoke-test result and no lab inference yet. The old dependency pins all have cp310 manylinux wheels, so no compiler toolchain is installed — if a wheel turns out to be missing, the fix is to add `build-essential`, not to float the pin |
| Tags | `:v1` (= `:latest`, `:torch2.5-cpu`); weights NOT baked |
| Notes | **`fair-esm 1.0.2` is chosen over upstream's own pin, deliberately.** `app/requirements.txt` pins `fair-esm==2.0.0`, but upstream's README states the manuscript results were produced with 1.0.2 and warns twice that predictions depend on the ESM-1b version; manuscript fidelity won. The tradeoff is real — anyone reproducing "CLEAN as shipped" gets 2.0.0, the two have not been compared here, and upstream publishes no comparison. Build the as-shipped combination with `--build-arg FAIR_ESM_VERSION=2.0.0 --build-arg ESM_GIT_SHA=0b59d87ebef95948c735b1f7aad463dc6dfa991b`, and record which one produced any result that leaves the lab. **torch is pinned below 2.6 and the ceiling is load-bearing** — 2.6.0 changed `torch.load` to default `weights_only=True`, which refuses the ESM-1b checkpoint because it pickles an `argparse.Namespace`; upstream's own Dockerfile installs torch unpinned and so fails at first inference rather than at build time. The `esm/` clone shadows the `fair-esm` package as a namespace package whenever the working directory is on `sys.path`, so read versions with `importlib.metadata.version("fair-esm")`. Max-separation is the default and recommended mode (deterministic, no hyperparameters); ESM-1b's 1024-token limit applies to long proteins and is an upstream concern |

## saprot

| | |
|--|--|
| Task | Protein language model — structure-aware sequence embedding and mutational scoring. Tokenizes *residue-states* rather than amino acids: each token pairs the amino acid with that residue's foldseek 3Di structural state (`Aq`, `Md`, `Gp`), and the structural half can be masked (`A#`) for sequence-only input |
| Sensor | `protein-sequence+structure` (label `bradleylab.model.sensor`) — amino-acid sequence plus a PDB / mmCIF structure |
| Upstream repo | [westlake-repl/SaProt](https://github.com/westlake-repl/SaProt), vendored helper pinned at commit `e91e4858` (2026-03-08); the project has never cut a GitHub release. Bundled alongside: [steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) release `10-941cd33` (2025-01-19) |
| Upstream license | **`MIT AND GPL-3.0`** — SaProt code and weights are MIT, but this image bundles the **foldseek** binary, which is **GPL-3.0**. The binary is an unmodified upstream release invoked as a subprocess, which is the arrangement foldseek itself documents, but the licence mix is material for a publicly published image and must be known before redistribution outside the lab |
| Paper | *SaProt: Protein Language Modeling with Structure-aware Vocabulary* — preprint [doi:10.1101/2023.10.01.560349](https://doi.org/10.1101/2023.10.01.560349); journal version in *Nature Biotechnology*, 2025-10-24 |
| Weights source | HF Hub, [`westlake-repl`](https://huggingface.co/westlake-repl) org — **`SaProt_1.3B_AFDB_OMG_NCBI` (1.30B, 5.21 GB — default)**, `SaProt_1.3B_AF2` (1.30B, 5.20 GB), `SaProt_650M_AF2` / `SaProt_650M_PDB` (650M, 2.61 GB), `SaProt_35M_AF2` (35M, 0.13 GB). Not baked; fetched on first `from_pretrained` into `$HF_HOME=/root/.cache/huggingface` (bind-mount for persistence). Every checkpoint repo ships its weights twice (`model.safetensors` alongside `pytorch_model.bin`, or a SaProt-native `.pt`), so pre-stage with `allow_patterns` rather than a bare `snapshot_download` |
| Weights license | MIT and **ungated** — the cleanest licence situation of the protein models in this repo. (The image as a whole is still `MIT AND GPL-3.0` because of foldseek) |
| Container stack | `nvcr.io/nvidia/pytorch:25.04-py3` (torch 2.7.0a0, CUDA 12.9, Python 3.12) + **stock** `transformers` 5.15.0 + `biopython` 1.88 + `accelerate` 1.14.0 + foldseek `10-941cd33` static AVX2 binary at `/opt/foldseek/bin/foldseek` (`$FOLDSEEK_BIN`) + `saprot_utils.foldseek_util` (~200 lines, fetched from `raw.githubusercontent.com` at `e91e4858` with its sha256 checked at build time). None of SaProt's training code is installed |
| H100 status | Native sm_90. Weights are 5.21 GB at F32 and half that in bf16, so any H100 or A100 is oversized for the model itself. **The bottleneck is foldseek on CPU** — `get_struc_seq` invokes it with `--threads 1`, so parallelism comes from processing several structures at once. Size the job by CPU count, not VRAM |
| Lab status | **experimental** — not yet benchmarked on lab data. Default target is embedding structures with `SaProt_1.3B_AFDB_OMG_NCBI`. **The checkpoint choice is forced by the pipeline, not by size:** upstream warns that frozen embeddings from the 35M and 650M models are usable only with structure tokens, while the 1.3B models accept AA-only input too — parts of the pipeline will have no structure, and using one model throughout keeps embeddings comparable |
| Architecture | **amd64.** `foldseek-linux-avx2` requires AVX2, which every current Compute2 node supports; the release also ships `arm64` and `gpu` assets, so an ARM build would need a different one |
| First-run / current behavior | Build-time smoke test is offline and CPU-only: runs the foldseek binary and asserts the reported version matches the pin, tokenizes a probe string through `EsmTokenizer` to confirm the two-character-token trie behaves, and runs a forward pass through a toy `EsmForMaskedLM` with SaProt's architecture settings. It downloads no weights, so a real end-to-end load still has to happen on Compute2. **The 3Di output has not been validated against upstream's `example/8ac8.cif`** — run that check once before trusting embeddings |
| Tags | `:v1` (= `:latest`, `:torch2.7-cu129`); weights NOT baked |
| Notes | **Bundling foldseek is the documented exception to one-model-per-container**, and about as clear-cut as that exception gets: SaProt's input representation does not exist until foldseek has produced it. `get_struc_seq` shells out to `foldseek structureto3didescriptor` *inside a single function call* and reads back the temp file it wrote, so there is no intermediate artifact a separate container could hand over without restructuring the call; foldseek is also not pip-installable. **Upstream's environment is deliberately not used** — `requirements.txt` pins `torch==1.13.1`, which predates Hopper; the checkpoints declare `model_type: "esm"` / `architectures: ["EsmForMaskedLM"]` and every field they set is still supported by transformers 5.15.0, so they load on a modern stack. Consequence: the repo's YAML-driven fine-tuning and evaluation scripts are **not** available here — this image covers embedding and scoring. The helper is installed as `saprot_utils`, not upstream's top-level `utils`. **`plddt_mask=True` must be passed explicitly for any predicted structure** — the `"auto"` default only turns masking on when the file text contains "alphafold", so it catches AFDB downloads and misses locally-run AlphaFold, ESMFold, Boltz, or renamed files; pass `False` for experimental structures, where the B-factor column is a real B-factor. **No published checksum for the foldseek tarball** — the GitHub release carries no asset digest, so the pin is the release tag and nothing stronger |

## boltz

| | |
|--|--|
| Task | Biomolecular co-folding — structure prediction for complexes of proteins, RNA, DNA, and small molecules, plus binding-affinity prediction for a nominated ligand chain requested from a block in the same input file |
| Sensor | `biomolecular-sequence` (label `bradleylab.model.sensor`) — sequence + chemical-component identity, not a sensor product |
| Upstream repo | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) (Barzilay / Jaakkola group, MIT), tag **v2.2.1** (2025-09-08). `main` still receives commits but no newer tag has been cut |
| Upstream license | **MIT — code *and* weights.** Academic and commercial use are both permitted, with no acceptance step and no gated download |
| Paper | Not recorded in this repo — `boltz/README.md` cites the upstream repo at tag v2.2.1 and gives no paper reference; defer to upstream for the current citation |
| Weights source | **NOT baked** — ~6.2 GB fetched on first run into `$BOLTZ_CACHE=/opt/boltz-cache`: `boltz2_conf.ckpt` (2.29 GB, structure + confidence), `boltz2_aff.ckpt` (2.06 GB, affinity head), `mols.tar` (1.86 GB, CCD chemical-component data). The two checkpoints come from `model-gateway.boltz.bio` with a Hugging Face fallback at [`boltz-community/boltz-2`](https://huggingface.co/boltz-community); **`mols.tar` is Hugging Face only, with no gateway mirror**. `$BOLTZ_CACHE` must be an **absolute** path — boltz raises rather than falling back if it is relative |
| Weights license | MIT, ungated |
| Container stack | `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime` (torch 2.8.0, CUDA 12.9, cuDNN 9, Python 3.11) + `boltz[cuda]==2.2.1` from PyPI with `cuequivariance*` pinned to 0.6.1. Not an NGC base: boltz pins `numpy>=1.26,<2.0` and NGC images hold a large NVIDIA-built stack behind a global pip constraint file, where forcing a numpy downgrade is how those images break; boltz's `requires-python >=3.10,<3.13` also rules out 3.13 |
| H100 status | sm_90 (the CUDA 12.9 build covers Hopper). Reported VRAM ~11 GB for structure and ~7–8 GB for affinity (third-party production report) against a **≥48 GB** vendor floor in NVIDIA's NIM support matrix — the gap is headroom on large multimers, not the typical case. Host RAM 64 GB is comfortable; ~10–60 s for a ~500-residue system on the stock PyTorch path (NVIDIA's TensorRT figures are 1.45–6.4× faster and do not apply here) |
| Lab status | **experimental** — not yet benchmarked on lab data |
| First-run / current behavior | Build smoke test imports a **real model class** — `from boltz.model.models.boltz2 import Boltz2` succeeds on the CPU build runner because boltz reaches cuEquivariance only from inside `kernel_triangular_mult()`, not at module scope (unlike `esm` and `evo2`, which stop at the metadata layer). It also asserts the cuEquivariance version, the numpy major, and that torch is still the base image's build. No prediction on lab data yet |
| Tags | `:v1` (= `:latest`, `:torch2.8-cu129`); weights NOT baked |
| Notes | **Silent out-of-memory is the failure mode to design around.** `boltz predict` catches CUDA OOM per batch, prints `\| WARNING: ran out of memory, skipping batch`, and still **exits 0** ([upstream issue #167](https://github.com/jwohlwend/boltz/issues/167)), so a Slurm job that produced no structures at all reports success. Every job must gate on output files existing, not on exit status — the one-line gate is in `boltz/README.md`. **MSAs are required and are not fetched for you** (`--use_msa_server` defaults to False); the lab default is precomputed per-chain `.a3m` files from our own mmseqs2, reusing the AlphaFold3 database investment, so the image makes no MSA network calls. `msa: empty` (single-sequence mode) reduces accuracy and is an ablation, not a shortcut. **Affinity constraints, from upstream:** exactly one ligand chain, ≤128 heavy atoms (authors advise ≤56), **protein targets only** — RNA / DNA / cofactor targets run but are not reliable; outputs are `affinity_pred_value` (log10 IC50) and `affinity_probability_binary`; affinity adds ~1.5–3× to runtime and templates ~4×. **The cuEquivariance pin is load-bearing** — boltz's `[cuda]` extra asks only for `>=0.5.0`, and 0.11.1 promoted `torch>=2.11` from a test extra to a hard runtime dependency, so an unpinned build replaces the base image's torch and CUDA stack with PyPI wheels, quietly and differently on different build days. The image carries a duplicate cuBLAS (a few hundred MB) because `cuequivariance-ops-cu12` requires its own `nvidia-cublas-cu12` wheel |

## chai-1

| | |
|--|--|
| Task | Biomolecular co-folding — structure prediction for complexes of proteins, RNA, DNA, and small molecules, reaching most of its accuracy **without MSAs** by using a traced ESM-2 3B embedder in their place |
| Sensor | `biomolecular-sequence` (label `bradleylab.model.sensor`) — FASTA with typed chain headers (protein / ligand / …), not a sensor product |
| Upstream repo | [chaidiscovery/chai-lab](https://github.com/chaidiscovery/chai-lab), release **v0.6.1** (2025-03-18). Commits continue on `main` but no newer release has been cut, and upstream itself recommends pinning |
| Upstream license | **Apache-2.0 — code *and* weights.** Chai Discovery relicensed in **November 2024**; the original September-2024 research-only terms no longer apply, and academic and commercial use are both permitted. Any note in older lab documents describing Chai-1 as restrictively licensed is obsolete |
| Paper | Chai Discovery (2024), *Chai-1: Decoding the molecular interactions of life*, bioRxiv — [doi:10.1101/2024.10.10.615955](https://doi.org/10.1101/2024.10.10.615955) |
| Weights source | **NOT baked** — ~7.0 GB across 8 assets into `$CHAI_DOWNLOADS_DIR=/opt/chai-downloads`, which is read at **import** time and so must be set before Python starts. **The Hugging Face mirror at [`chaidiscovery/chai-1`](https://huggingface.co/chaidiscovery/chai-1) is partial:** it hosts only the six `models_v2/*.pt` components (~1.18 GB total). The 5.68 GB traced ESM-2 embedder (`esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt`) and the 125 MB `conformers_v1.apkl` come from **`chaiassets.com` only** — so pre-staging from HF alone leaves an air-gapped job to fail at the ESM-embedding step rather than at startup |
| Weights license | Apache-2.0, ungated |
| Container stack | `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime` (torch 2.6.0, CUDA 12.6, cuDNN 9, Python 3.11) + `chai_lab==0.6.1` from PyPI, pinned to the release tag. CLI: `chai-lab fold`, `chai-lab a3m-to-pqt`, `chai-lab citation` |
| H100 status | sm_90 (the CUDA 12.6 build covers Hopper and the bfloat16 support Chai-1 needs). Upstream recommends A100 / H100 80 GB or an L40S 48 GB. Measured PoseBench benchmark (A100 80 GB, 5 diffusion samples, `low_memory=True`): **56.2 GB VRAM peak, 58.5 GB host RAM peak**, ~115 s per small protein–ligand complex |
| Lab status | **experimental** — not yet benchmarked on lab data |
| First-run / current behavior | Build smoke test imports the **real inference entrypoint** — no chai_lab module touches CUDA or the network at import scope, so `from chai_lab.chai1 import run_inference` resolves on the CPU build runner with no metadata-only fallback. It asserts torch is inside `[2.3, 2.7)` and numpy is still 1.x. No prediction on lab data yet |
| Tags | `:v1` (= `:latest`, `:torch2.6-cu126`); weights NOT baked |
| Notes | **Request ≥64 GB host `--mem`** — the host-RAM figure above is the one people miss, and a GPU-sized job with a default host allocation is the common way this model dies; the Slurm OOM message points at the host, not the GPU, so it reads as unrelated. Keep `low_memory=True` (the default). The output directory must be **absent or empty** — `run_inference` asserts on a non-empty output dir and dies at startup. `--no-use-esm-embeddings` avoids the 5.68 GB download but **changes what the model is given as input** — a methodological change, not a deployment convenience; record it if used. The stock configuration makes zero MSA and template network calls (`use_msa_server` and `msa_directory` both default off), so the image is fully air-gappable once assets are staged; local MSAs are supplied as `aligned.pqt` via `--msa-directory`, converted from our own mmseqs2 a3m output with `chai-lab a3m-to-pqt`. **This image and `boltz` sit on different bases** (2.6.0-cu126 vs 2.8.0-cu129) because chai_lab pins `torch>=2.3.1,<2.7` and `numpy~=1.21`; the two ceilings are incompatible, so do not "harmonise" them. **Chai-2 is not obtainable** — bioRxiv report only, no public weights or inference code; the `chai-lab` package ships Chai-1 |

## dnabert-s

| | |
|--|--|
| Task | DNA language model — species-aware sequence embedding (768-D, mean pooling) for metagenomic binning. Contigs from the same species sit close together in the embedding space, so bins can be clustered without alignment or reference genomes |
| Sensor | `dna-sequence` (label `bradleylab.model.sensor`) — nucleotide sequence, not a sensor product |
| Upstream repo | [MAGICS-LAB/DNABERT_S](https://github.com/MAGICS-LAB/DNABERT_S) — released 2024, last pushed 2026-01-01; effectively frozen but functional |
| Upstream license | **The GitHub repo ships no LICENSE file** — formally all rights reserved. Fine to read and run; do **not** redistribute its code. This image sidesteps that entirely by vendoring nothing from the repo: the custom modelling files the checkpoint needs (`bert_layers.py`, `flash_attn_triton.py`, …) live in the Apache-2.0 HF repo and are fetched at load time by `trust_remote_code=True`, so the image is Apache-2.0 clean. The missing GitHub licence matters only if someone later copies training or evaluation scripts out of that repo into a lab artifact |
| Paper | *DNABERT-S: Pioneering Species Differentiation with Species-Aware DNA Embeddings* — no DOI or arXiv ID is recorded in this repo; `dnabert-s/README.md` defers to the upstream repo for the current citation |
| Weights source | HF Hub: [`zhihan1996/DNABERT-S`](https://huggingface.co/zhihan1996/DNABERT-S) — a single checkpoint (~117M params, 468 MB `pytorch_model.bin`, fp32); there is no size ladder to choose from. Not baked; fetched on the first `from_pretrained` call into `$HF_HOME=/root/.cache/huggingface` (bind-mount for persistence). `trust_remote_code=True` is mandatory — the architecture is not in `transformers`, it ships as Python inside the HF repo |
| Weights license | **Apache-2.0** (HF model card) and **ungated** — no token needed. Note this differs from the upstream GitHub repo, which has no licence at all (above) |
| Container stack | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` + Python 3.11 + PyTorch 2.5.1 cu121 + `transformers==4.27.*` (upstream's own pin) + `huggingface_hub>=0.16,<0.26` + `einops>=0.6` + `numpy<2`. **Triton is uninstalled as the last pip step** |
| H100 status | Native sm_90 (`TORCH_CUDA_ARCH_LIST` includes 9.0). The model is small enough that a GPU buys throughput, not capability — CPU inference is feasible, and the reason to run on Compute2 is embedding a whole contig set in one pass. `device_map="auto"` is unavailable (`accelerate` is not installed, because versions compatible with transformers 4.27 conflict with the pins); use `.to("cuda")` |
| Lab status | **experimental** — not yet benchmarked on lab data. Intended first target is batch embedding of assembled contigs |
| First-run / current behavior | Build smoke test is a **real import**, not metadata — nothing in this stack needs a device at import time, so the build resolves `AutoModel` / `AutoTokenizer` / `AutoConfig` for real on the CPU runner, and asserts triton is absent. It stays offline, so what it does not cover is the remote-code load path, which needs network and is first exercised by a job's `from_pretrained` call. No lab embedding output yet |
| Tags | `:v1` (= `:latest`, `:torch2.5-cu121`); weights NOT baked |
| Notes | **This model cannot share a container with `ntv3`** — DNABERT-S requires `transformers==4.27` and NTv3 requires `>=4.55`; no version satisfies both, so the two DNA images are separate by necessity, not only by the one-model-per-container convention. A "consolidate the DNA models" change fails at pip resolution, and forcing it past that point breaks one model silently. **transformers 4.27 dictates the whole stack:** Python 3.11 not 3.12 (4.27 requires `tokenizers<0.14`, which has no cp312 wheel), torch 2.5.1 not 2.6+ (2.6 flipped `torch.load` to `weights_only=True` and this checkpoint is a `.bin`), `huggingface_hub<0.26` (0.26 removed `cached_download`, which transformers 4.27 imports at package-import time — **our pin, not upstream's**, and upstream's requirements file will not warn you), `numpy<2`. That is also why this image is not on the NGC base `esm` and `evo2` use, which is Python 3.12. **Triton is removed on upstream's advice** for non-A100 GPUs, where its flash-attention kernel misbehaves; the uninstall must stay the last pip step because torch declares triton as a Linux dependency and any later install restores it. Consequence: `torch.compile` / TorchInductor are unavailable — not verified on our own H100, this follows upstream's instruction rather than a measurement here. **Mean pooling over the attention mask is the documented embedding method** — not `[CLS]`, not max pooling; changing it invalidates comparison against upstream's reported results. `max_position_embeddings` is 512 but ALiBi lets the encoder extrapolate; upstream trained to roughly 10 kb and publishes **no recipe for longer contigs** — chunk and pool, and record the window and overlap in `METHODS.md`, because it changes the embedding |

## ntv3

| | |
|--|--|
| Task | Genomic language model — sequence embeddings (1536-D) at nucleotide resolution over windows up to **1 Mb**, plus ~16,000 functional genomic tracks across 24 species (the signal set otherwise obtained from BigWig files) and base-resolution annotation suitable for writing out as BED |
| Sensor | `dna-sequence` (label `bradleylab.model.sensor`) — nucleotide sequence, not a sensor product |
| Upstream repo | [instadeepai/nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer) (see `docs/nucleotide_transformer_v3.md`) — released December 2025, actively maintained. HF collection `InstaDeepAI/nucleotide-transformer-v3` |
| Upstream license | **CC BY-NC-SA 4.0** (upstream code) — non-commercial |
| Paper | Not recorded in this repo — `ntv3/README.md` cites the upstream repo and its v3 documentation page, and gives no paper reference; defer to upstream for the current citation |
| Weights source | HF Hub: [`InstaDeepAI/NTv3_650M_post`](https://huggingface.co/InstaDeepAI/NTv3_650M_post) (650M params, 2.72 GB fp32 safetensors, embedding dim 1536 — the recommended checkpoint), plus `NTv3_100M_post`, `NTv3_{100M,650M}_post_131kb`, pre-trained-only `NTv3_{8M,100M,650M}_pre`, and a separate `NTv3_generative` checkpoint (Jan 2026, out of scope here). Sizes for the non-default checkpoints were not recorded in the lab's research pass — check the HF model card rather than assuming. **Gated**: an HF account must accept the terms on the model page before any download works, or it returns 403. Not baked; staged on a login node with a token from the environment, after which jobs run offline with `HF_HUB_OFFLINE=1` |
| Weights license | **InstaDeep NTv3 non-commercial licence** — no commercial use, and no training a competing model on this model's outputs. Image labelled `LicenseRef-InstaDeep-NTv3-NonCommercial` with a `bradleylab.model.use_restriction` label, because the restriction travels with the image and will not be obvious to whoever pulls it next. The **second** non-commercial image in the catalog, after `dofa-clip` |
| Container stack | `nvcr.io/nvidia/pytorch:25.04-py3` (torch 2.7.0a0, CUDA 12.9, flash-attn, Python 3.12 — the same base as `esm`, `evo2`, `saprot`) + `transformers>=4.55,<5`, upstream's floor for the custom `ntv3_posttrained` architecture, reached via `trust_remote_code=True`. Large image (~9 GB base); the build workflow frees runner disk first |
| H100 status | Native sm_90; bf16 recommended. **There is no official VRAM figure and none is invented here** — the weights are the easy part (2.72 GB fp32, roughly half in bf16, against 80 GB on a C2 H100); the unknown is activation memory, which scales with context length and decides whether a 1 Mb window fits. Start at 131 kb windows, measure with `torch.cuda.max_memory_allocated()`, and scale up empirically, recording the window size that worked and the memory it used |
| Lab status | **experimental** — not yet benchmarked on lab data. Intended first target is embeddings and track prediction at 131 kb windows |
| First-run / current behavior | Build smoke test is a **real import, offline**: transformers and torch both import on the CPU build runner, so `AutoModel` / `AutoTokenizer` / `AutoConfig` resolve for real — more than `esm` and `evo2` manage. It cannot reach the model class, which lives in gated remote code. It also **asserts no HF token is present in the build environment** (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`), so a token cannot be published to GHCR by accident. No lab inference yet |
| Tags | `:v1` (= `:latest`, `:torch2.7-cu129`); weights NOT baked |
| Notes | **Three input rules produce wrong output rather than an error.** (1) Input length must be a **multiple of 128 bp** — the tokenization depends on it. (2) Pad with the character **`N`**, appended to the string before tokenization — not the tokenizer's `[PAD]` token, which is the reflex from every other HF model and which the model has no representation for. (3) Post-trained track outputs are cropped to the **middle 62.5%** of the window, so a 131,072 bp window yields 81,920 bp of usable track and tiling a chromosome steps by that 81,920 bp, not by the full window, or gaps appear. **The HF token is never baked and never passed into a job** — stage weights on a login node, then the job needs no credential. **This model cannot share a container with `dnabert-s`** (transformers `>=4.55` vs `==4.27`; see that card). **HF PyTorch route, not the GitHub repo** — `instadeepai/nucleotide-transformer` is a JAX codebase, and its install instructions do not apply here. The transformers pin is a range rather than a SHA, so two builds months apart can differ; pin the exact version if a result must be reproducible to the byte. Remote code fetched at `from_pretrained` time may want packages this image does not ship — run one load on a login node before submitting a batch, and add any missing package to the Dockerfile rather than pip-installing inside a job. No track-writer library (`pyBigWig`) is installed |

---

## Deprecated images

For history of `bradleylab/multispec-species` and `bradleylab/tree-analysis`, see [`DEPRECATIONS.md`](DEPRECATIONS.md).
