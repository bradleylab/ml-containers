# FSCT

Forest Structural Complexity Tool (Krisanski et al. 2021, *Remote
Sensing* 13:4677) — TLS / MLS individual tree segmentation.

Upstream: <https://github.com/SKrisanski/FSCT>

## Image variants

Pull: `ghcr.io/bradleylab/fsct:v1` (also tagged `:latest`, `:cpu-torch1.9`).

**CPU-only.** FSCT hard-pins `torch==1.9.0+cu111` and matching PyG
extension wheels. Those wheels don't run on H100 (SM 9.0); porting to
torch 2.x + CUDA 12 is the same multi-day exercise we went through for
SegmentAnyTree (see `../segment-any-tree-h100`). FSCT inference is
CPU-friendly and finishes a ~60 M-point TLS file in tens of minutes on
a modern CPU, so we stay on the pinned config and run CPU-only.

Use on the Compute2 `general-cpu` partition (no GPU allocation needed).

## Invocation

Headless CLI wrapper `run_cli.py` replaces the upstream tkinter GUI:

```bash
enroot start --mount /scratch2/fs1/.../tls_bp7:/data \
    bradleylab+fsct+v1.sqsh \
    --input /data/TRC_BP7_voxel10cm.laz \
    --output-dir /data/fsct/
```

Key flags:

- `--input PATH` (required) — `.las` or `.laz`
- `--output-dir DIR` (required) — receives `<stem>_FSCT_output/` with
  segmented point cloud, per-tree CSV, DTM, and plot report
- `--plot-centre X Y`, `--plot-radius R` — optional circular crop
- `--batch-size N`, `--num-cpu-cores N` — tune for hardware
- `--gpu` — override CPU-only mode (only useful if you rebuild the
  image with CUDA-matched PyG wheels)
- `--skip-report` — skip `make_report` for faster batch runs

## Outputs

FSCT creates `<output-dir>/<stem>_FSCT_output/` containing:

- `segmented.las` — per-point tree instance label (`treeID`) + semantic
  class (ground / vegetation / stem / CWD)
- `tree_data.csv` — per-tree DBH (cm), height (m), stem position
  (easting / northing / elevation), crown dimensions
- `DTM.tif` — 1-m digital terrain model
- `plot_report.html` — summary report with figures

## Build

GitHub Actions workflow:
`.github/workflows/build-fsct.yml`. Triggers on any change under
`fsct/**`. Pushes to GHCR.

## Known limitations

- Runs CPU-only; not particularly fast on very large files. For the
  2025-09-02 BP7 survey (58 M points raw, thinned to ~10 M at 10 cm
  voxel) expect 30-60 min per file.
- FSCT's semantic classifier was trained on mid-latitude coniferous +
  some mixed forest. Oak-hickory performance is reasonable but the
  understory / CWD classes may need post-hoc filtering.
- Hardwired tkinter import in `run_tools.py` — we install `python3-tk`
  in the image so the module import succeeds without actually using
  the GUI.
