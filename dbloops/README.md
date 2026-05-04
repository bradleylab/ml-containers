# dbloops

DBloops density-based 3D point-cloud clustering for grain-size
distribution and boulder mapping (Jacobson et al. 2025), compiled to a
standalone binary with MATLAB Compiler R2024b and packaged against the
free MATLAB Runtime.

## What it is

- **Detector:** DBloops itself — a two-pass DBSCAN over the local
  3D neighbourhood with epsilon scaling (`np`, `esfa`, `esfb`).
- **Bundled source:** `DBloops/`, `Terpunkto/`, `G3point/` from
  upstream tag v1.0.0 (commit `c3acb15`), with backslash-path patches
  applied (eight one-line edits; see
  bradleylab/rock_glaciers/`scripts/matlab/dbloops_patches.diff`).
- **Wrapper:** the env-driven `run_dbloops_patch.m` from the same repo,
  reading `PATCH_XYZ`, `PATCH_OUT`, `NP_VAL`, `ESFA`, `ESFB`.

## What it is NOT

- **Not a full Jacobson pipeline.** Upstream's `MasterScript` couples
  DBloops with a Random Forest clast/matrix classifier (Terpunkto) that
  separates true rock surfaces from matrix points before clustering.
  This container ships the clustering step only — the classifier needs
  per-deployment training and is not bundled. At typical TLS densities
  (11–100 pts/m²), running this container's binary on raw points
  produces detections that do not correspond to real boulders. See
  bradleylab/rock_glaciers/`EXPERIMENTS.md` v3a for the visual-audit
  failure on rock-glacier surfaces.

  The right deployment is high-density gravel-bed lidar matching
  Jacobson's training data (point spacing well under 1 cm, sharp
  clast-matrix contrast).

## Run

The image is the entrypoint. Mount your data at `/work` and pass the
input/output paths via environment variables:

```bash
docker run --rm \
    -e PATCH_XYZ=/work/imogene_patch1.xyz \
    -e PATCH_OUT=/work/imogene_patch1_np18 \
    -e NP_VAL=18 \
    -v "$(pwd)":/work \
    ghcr.io/bradleylab/dbloops:latest
```

Outputs land at `${PATCH_OUT}.mat`, `${PATCH_OUT}_clusters.csv`,
`${PATCH_OUT}_gsd.csv`. Optional env vars:

| Variable | Default | Meaning |
|----------|---------|---------|
| `NP_VAL` | 18 | DBloops `np` (target points per cluster) |
| `ESFA`   | 2.6 | small-epsilon scale factor |
| `ESFB`   | 0.6 | large-epsilon scale factor |

## On Compute2 (pyxis/enroot)

```bash
enroot import \
    -o /storage1/fs1/<user>/Active/dbloops/bradleylab+dbloops+latest.sqsh \
    'docker://ghcr.io#bradleylab/dbloops:latest'

srun --account=compute2-<user> \
     --partition=general-preempt-cpu --time=01:00:00 --mem=16G \
     --container-image=/storage1/fs1/<user>/Active/dbloops/bradleylab+dbloops+latest.sqsh \
     --container-mounts=/scratch2/fs1/<user>/dbloops:/work \
     bash -lc 'PATCH_XYZ=/work/imogene_patch1.xyz \
               PATCH_OUT=/work/imogene_patch1_np18 \
               NP_VAL=18 \
               /opt/dbloops/entrypoint.sh'
```

## Files

| File | What |
|------|------|
| `Dockerfile` | MCR install + binary copy + smoke test |
| `entrypoint.sh` | discovers MCR root, forwards to mcc-generated wrapper |
| `bin/run_dbloops_patch` | compiled MATLAB binary (1.3 MB) |
| `bin/run_run_dbloops_patch.sh` | mcc-generated `LD_LIBRARY_PATH` setup |
| `bin/requiredMCRProducts.txt` | toolbox dependency manifest |
| `bin/readme.txt` | upstream mcc deployment notes |
| `compile.sh` | re-compile recipe (run on Compute2 with `module load matlab/R2024b`) |

## Updating

To rebuild the binary from updated DBloops source:

1. Re-apply the patches on a fresh DBloops v1.0.0 clone via
   bradleylab/rock_glaciers/`scripts/matlab/apply_patches.sh`, with the
   updated `.m` wrappers in place.
2. Run `compile.sh <repo-root> <out-dir>` on a host with MATLAB
   Compiler R2024b licensed (Compute2 is the default).
3. Replace `bin/` in this directory with the contents of `<out-dir>`,
   commit, push. The GH Action (`build-dbloops.yml`) rebuilds the
   image on push.

To bump the MATLAB Runtime version, update `MCR_UPDATE` (and
`MCR_VERSION` if a major release change) in `Dockerfile` or pass at
build time. R2024b major-version compatibility is guaranteed across
Updates by Mathworks; cross-major-version requires re-compiling the
binary against the matching Compiler.

## Caveats

- The smoke test runs DBloops on 2,000 synthetic uniform points. It
  verifies the binary launches, the runtime loads, all toolboxes are
  resolvable, and a `.mat` file lands. It does NOT verify scientific
  correctness — that lives upstream in the F9 reproduction (which needs
  the 3.4 GB Hanging Lake test data and is not part of this container).
- MATLAB Runtime is licensed under the Mathworks runtime EULA. The
  binary inside this image was compiled from upstream MIT-licensed
  MATLAB code under WashU's TAH licence; the resulting standalone
  application is redistributable per Mathworks's compiler runtime
  redistribution terms.
