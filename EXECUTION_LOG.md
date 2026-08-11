# Execution Log

## Run Digest

- **Last updated:** 2026-08-10 22:01 CDT
- **Current phase:** B1 verification and B2 publication in progress
- **Active batch:** B1 — Close the restricted Storage3 installation
- **Last completed batch:** none
- **Next exact batch:** B1
- **Active PR:** not created
- **Docs promoted this run:** canonical vault container note already contains staged AF3 access
- **Latest Elves Report:** not generated

## Session Setup: 2026-08-10 21:35 CDT

**Plan:** `docs/plans/alphafold3-compute2-ready.md`
**Survival guide:** `SURVIVAL_GUIDE.md`
**Learnings:** `learnings.md`
**Execution log:** `EXECUTION_LOG.md`
**Branch:** `feat/alphafold3-container` at start tip
`9dc06133025b4c712b6864f016b8e0a6dd49d4f3`
**Run mode:** finite, blocker-only; stop is not allowed while acceptance remains.

**Batch breakdown:**

1. B1 — close Storage3 model/database installation.
2. B2 — publish a Compute2-compatible GHCR candidate.
3. B3 — pass scheduled H100/JAX/model-load smoke.
4. B4 — execute minimal CPU-pipeline plus H100 prediction.
5. B5 — register verified agent access and close documentation/review.

**Preflight evidence:**

- Live CLI state: Compute2 `VERIFYING`; GHCR digest `UNSET`.
- Authenticated GHCR API: package not found before publication.
- Compute2 job `2715132`: running database installation.
- Compute2 job `2715106`: pending free-pool identity follow-up, not a production gate.
- Git branch/worktree: dedicated `feat/alphafold3-container` worktree based on `origin/main`.
- Escalated GitHub preflight: keyring authentication and push dry-run passed.
- Elves acceptance staging validation passed with exact plan/session ID and text mappings.
- B1 rollback authority: `refs/elves/rollback/alphafold3-compute2-read/019fed64-1c4d-74/b1-83cd75ce9637` at the start tip, pushed to `origin`.
- B2 rollback authority: `refs/elves/rollback/alphafold3-compute2-read/019fed64-1c4d-74/b2-ca620a221981` at the start tip, pushed to `origin`.
- Docker daemon is unavailable locally; GitHub Actions is the required build route.

**Decisions made:**

- Beads is not added because Codex Goal plus Elves plan/session/survival/log already provide durable tracking.
- Host-native work is used because the critical path crosses terms-restricted storage, university Slurm, and external CI; no worker needs independent write ownership.
- The branch workflow will build SHA candidates before merge so runtime acceptance does not require changing `main`.
- Buildx provenance/SBOM attestations are disabled for this image because Compute2 enroot 3.4.1 cannot import the resulting OCI image index; scientific provenance remains in explicit run manifests.
- The user previously received the proposed commit `feat: add AlphaFold 3 software container`; the current explicit instruction to get it working authorizes the necessary feature-branch commit and push, but not merge.
- Selected the pinned official v3.0.4 ubiquitin monomer example as the technical smoke input. The local and upstream bytes both hash to `1880c15e12df1a331e0ee464705639ee80c6b036a3273b5cf5dcd8875e8c7749`; local structural validation passed without mutation.
- Independent B2 review found mutable release aliases, missing native license notices, deleted Git version provenance, and mutable package-write actions. The candidate now publishes only `sha-<commit>`, retains HMMER/Easel/libdivsufsort notices, preserves Git metadata through installation and asserts package version 3.0.4, and pins every package-write action to a full commit SHA.
- Candidate visibility is intentionally unresolved until the package exists. Compute2 import will use the observed package visibility or a non-persistent authenticated pull; no secret will be copied into source, Storage3, or logs.

**Continuation guard:** stop_allowed=false; remaining_batches=5;
next action is B1 evidence plus B2 candidate preparation.
