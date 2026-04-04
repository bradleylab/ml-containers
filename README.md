# ml-containers

Custom ML Docker images for bradleylab research compute (WashU RIS Compute2, EC2).

## Images

### segment-any-tree-h100

SegmentAnyTree individual tree segmentation rebuilt for H100 GPUs (sm_90).

- Base: PyTorch 2.7.0 + CUDA 12.8
- MinkowskiEngine from [CiSong10/cuda12-installation](https://github.com/CiSong10/MinkowskiEngine/tree/cuda12-installation) fork
- Includes all patches for CUDA 12 compatibility (thrust namespace, NVTX3, std::to_address)

Pull: `ghcr.io/bradleylab/segment-any-tree-h100:latest`

Built automatically via GitHub Actions on push to `segment-any-tree-h100/`.
