# Project Learnings

## Validation and tooling

- 2026-08-10: Compute2 enroot 3.4.1 cannot import buildx OCI indexes containing provenance/SBOM attestation manifests. Publish a plain single-platform `linux/amd64` image and keep provenance in explicit run manifests.
- 2026-08-10: Docker is unavailable on the trusted Mac, so AlphaFold image builds must use the committed per-image GitHub Actions workflow.

## Product and domain invariants

- 2026-08-10: The GHCR image is software-only. Model parameters and reference databases remain restricted Storage3 mounts and are never baked into an image.
- 2026-08-10: AlphaFold production acceptance targets the `general-gpu` H100 pool; the heterogeneous free/preempt pool is a separate experimental route.
- 2026-08-10: Installation, publication, and a successful model-load smoke are distinct from end-to-end `READY`; readiness requires a real output-producing prediction.

## Known traps

- 2026-08-10: Containerized Python on Compute2 must set `PYTHONNOUSERSITE=1` or host user-site packages can shadow the image runtime.
- 2026-08-10: `databases/current` must change only after complete required-file validation and deterministic manifest generation.
