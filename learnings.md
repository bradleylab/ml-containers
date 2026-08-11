# Project Learnings

## Validation and tooling

- 2026-08-10: Compute2's live enroot is 4.1.2. Publish AlphaFold as a plain single-platform `linux/amd64` image for reliable Pyxis/Enroot import, and keep scientific provenance in explicit run manifests instead of a multi-manifest attestation index.
- 2026-08-10: Docker is unavailable on the trusted Mac, so AlphaFold image builds must use the committed per-image GitHub Actions workflow.

## Product and domain invariants

- 2026-08-10: The GHCR image is software-only. Model parameters and reference databases remain restricted Storage3 mounts and are never baked into an image.
- 2026-08-10: AlphaFold production acceptance targets the `general-gpu` H100 pool; the heterogeneous free/preempt pool is a separate experimental route.
- 2026-08-10: Installation, publication, and a successful model-load smoke are distinct from end-to-end `READY`; readiness requires a real output-producing prediction.
- 2026-08-10: A technical Slurm bootstrap does not prove the agent transport. Keep Compute2 `VERIFYING` until the installed CLI completes submit, status, sanitized diagnostics, and fetch for a real prediction.

## Known traps

- 2026-08-10: Containerized Python on Compute2 must set `PYTHONNOUSERSITE=1` or host user-site packages can shadow the image runtime.
- 2026-08-10: Compute2's Enroot NVIDIA hook invokes `nvidia-container-cli` on CPU jobs unless `NVIDIA_VISIBLE_DEVICES=void` is exported before `srun`; forward that variable with Pyxis `--container-env` for AlphaFold's CPU data pipeline.
- 2026-08-10: `databases/current` must change only after complete required-file validation and deterministic manifest generation.
- 2026-08-10: Nested Mac-to-pliny-to-Compute2 SSH must pass the entire inner `ssh c2 '<command>'` invocation as one pliny-shell argument; separate argv elements allow shell metacharacters to execute on the wrong host.
- 2026-08-10: Agent-facing AlphaFold diagnostics must never return raw job stdout or stderr. Return allowlisted scheduler fields, fixed provenance-manifest fields, and log file sizes only.
- 2026-08-11: An SSH timeout after `sbatch` is an ambiguous acknowledgement, not proof that no job exists. Persist the parsable native ID atomically on the remote host, recover by receipt or unique Slurm name, and cancel every known allocation on local persistence failure.
- 2026-08-11: A fetch is not complete merely because an archive exists. Require normalized scheduler success, reject unsafe or unexpected archive members, verify the exact output set and every provenance/output hash, then atomically promote a new private destination.
