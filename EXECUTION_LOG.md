# Execution Log

## Run Digest

- **Last updated:** 2026-08-11
- **Current phase:** all batches complete; final landing validation
- **Active batch:** none
- **Last completed batch:** B5 — Expose verified agent access
- **Next exact batch:** none
- **Active PR:** not created
- **Docs promoted this run:** canonical vault container note contains verified AF3 access and immutable evidence
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
- Buildx provenance/SBOM attestations are disabled for this image because the acceptance-tested Compute2 enroot 4.1.2 route uses a plain single-architecture manifest rather than the resulting OCI image index; scientific provenance remains in explicit run manifests.
- The user previously received the proposed commit `feat: add AlphaFold 3 software container`; the current explicit instruction to get it working authorizes the necessary feature-branch commit and push, but not merge.
- Selected the pinned official v3.0.4 ubiquitin monomer example as the technical smoke input. The local and upstream bytes both hash to `1880c15e12df1a331e0ee464705639ee80c6b036a3273b5cf5dcd8875e8c7749`; local structural validation passed without mutation.
- Independent B2 review found mutable release aliases, missing native license notices, deleted Git version provenance, and mutable package-write actions. The candidate now publishes only `sha-<commit>`, retains HMMER/Easel/libdivsufsort notices, preserves Git metadata through installation and asserts package version 3.0.4, and pins every package-write action to a full commit SHA.
- Candidate visibility is intentionally unresolved until the package exists. Compute2 import will use the observed package visibility or a non-persistent authenticated pull; no secret will be copied into source, Storage3, or logs.
- CI run `31454207671` built and installed `alphafold3==3.0.4`, then failed at the local version assertion because double quotes were escaped inside a single-quoted Python expression. The correction invokes the installed virtual-environment Python directly and removes every post-install `uv run` resync so deleting `.git` cannot trigger the upstream 3.0.2 fallback version.
- Corrective CI run `31454889316` passed. GHCR digest: `sha256:6c6d8a36f5a9bd204a446b5e62cbd46780fcbf8ee4767763f7fa16d28a02f881`.
- Database finalizer `2719624` and scheduled verification `2719625` passed; content-manifest SHA-256 is `50da8ff704c606f9dd2dd5600e3bdd3de10b130213073a41cd5a5d0f6277c77c`.
- Compute2 enroot 4.1.2 imported the exact digest. Scheduled hash job `2720238` recorded `.sqsh` SHA-256 `619644486715f8a6389ecb8ed14fad2664f55cd39b209f1211c93d37d3301d24`.
- Active billable acceptance job: H100 smoke `2720279`, submitted to `general-gpu`.
- H100 smoke `2720279` completed `0:0` on `c2-gpu-015` in 22 seconds. It identified an NVIDIA H100 80GB HBM3, one JAX GPU, and a successfully loaded 405-leaf parameter tree; stderr was empty.
- Controlled bootstrap run `af3-20260811T033910Z-7fa36cfc` staged the exact official ubiquitin input (SHA-256 `1880c15e12df1a331e0ee464705639ee80c6b036a3273b5cf5dcd8875e8c7749`). CPU pipeline `2720501` failed before AlphaFold started because Compute2's Enroot NVIDIA hook invoked `nvidia-container-cli` on a CPU node where the NVIDIA driver was not loaded.
- The CPU launcher now exports and forwards `NVIDIA_VISIBLE_DEVICES=void`, the documented bypass condition observed in Compute2's installed hook. The fixed files were promoted as immutable runner `runner-36e7d02c78fb2694`, whose `content.sha256` hash is `36e7d02c78fb26945481e3bcc522ddeefa456dc8dd2f71689e2838d6bbb2bc42`.
- Corrected controlled-bootstrap retry `af3-20260811T034719Z-b6e64107` records `retry_of=af3-20260811T033910Z-7fa36cfc` and uses the same verified input and immutable assets. CPU pipeline `2720553` completed `0:0` on `c2-node-100` in 10m56s. Processed-input SHA-256 is `e6622da13749c03b7b9d1a8af829c1d65a989f120960f78d2f7ef41e9b392f16`; data-pipeline-manifest SHA-256 is `eecf9745ccdcb2b3426178a047b590ba8972fb0200d03b1430f6d7de64391543`.
- Gated one-H100 inference `2720682` was cancelled before allocation after the scheduler estimated no `general-gpu` backfill until 2026-08-11 16:44. The same request was submitted once as `2720745` to the available `general-interactive` H100 nodes; production configuration remains `general-gpu`.
- H100 inference `2720745` completed `0:0` on `c2-gpu-001` in 1m14s. Its hardware record identifies an NVIDIA H100 80GB HBM3. The run produced 21 non-empty output files, including six model CIFs and confidence/ranking artifacts. Every entry in `output.sha256` verified; output-manifest SHA-256 is `92dd68e6748f1e4cdd54aedae6d5fc39f6cb6e77c8c28a4b9b92b14c78cee9c9`, and provenance-manifest SHA-256 is `fe5d09064146e4a4399dc07cb7c5748c6e32c6f0f8cbe780f548d9e77c9f36c4`.
- The owner-only local configuration now reports Compute2 `READY`, with Modal and EC2 `UNCONFIGURED`. Global Codex MCP `af3` is enabled; a real stdio session listed seven bounded tools and returned `READY` from `alphafold_capabilities`.
- Independent terminal review then found that the first end-to-end acceptance bypassed the installed agent CLI, the nested SSH argv could execute shell metacharacters on pliny, configured Slurm account/partitions were not structurally bound to `sbatch`, raw job logs could enter an agent prompt, and CPU-stage scheduler provenance was incomplete. The active config was immediately rolled back to `VERIFYING` while these blockers were fixed.
- The installed runner now passes the entire quoted inner SSH command as one pliny argument, always requires explicit billable approval, passes the configured account and partitions directly to both `sbatch` calls, preserves discoverable failure/partial-submission records, records scheduler identity in both stage manifests, and returns only sanitized scheduler/manifests/log-size diagnostics. Focused tests increased to 17 and all pass; Ruff, Bash, and ShellCheck pass.
- The corrected Slurm files were promoted without changing the earlier release as immutable runner `runner-1bf9acfffb09817c`; `content.sha256` is `1bf9acfffb09817cab4d0457d72b5328b404094743890e8b25052dc619a7ef05`, and all release directories/files are mode `0550`/`0440`.
- The runner was installed from a built wheel rather than an editable source tree. Installed CLI acceptance job `af3-20260811T043457Z-feab9ac0` submitted CPU `2721167` and dependent H100 inference `2721168` through the exact agent transport. CPU completed `0:0` on `c2-node-100` in 10m43s; inference completed `0:0` on `c2-gpu-001` in 43s and recorded an NVIDIA H100 80GB HBM3.
- The installed CLI's privacy-preserving diagnostics returned complete CPU/GPU scheduler and manifest identities without raw logs. `af3 fetch` retrieved 21 outputs into a private temporary directory; every output hash verified against manifest SHA-256 `5204e688c4a8c7dc0bcb7983a2230417f7e62b326f3420c0c6cecabd5fc71eca`, and every provenance component hash matched. The owner-only production config was then promoted to `READY` on `general-gpu`, and the installed stdio MCP again listed seven tools and returned `READY`.
- The bootstrap-only non-READY acceptance switch was removed from the final production CLI after the test, so neither CLI nor MCP can bypass a future `VERIFYING` state. Partial provider records remain `FAILED` even after a CPU allocation reports `CANCELLED`.
- Terminal hardening added remote atomic native-ID receipts and unique-name recovery for lost SSH acknowledgements, best-effort cancellation after post-submit persistence failures, BatchMode on both SSH hops, sealed stdin when no private bytes are staged, generic MCP-facing remote errors, private-field-free validation, and fail-closed fetched-output/provenance verification before atomic destination promotion.
- The retained and installed final wheel SHA-256 is `5d154f90bca23346b65b9525ea3f716a0800d7aeff1e0ff65ef8ee5ce94976d5`; it contains the Apache license and every installed module byte-matches source. All 30 focused tests and Ruff pass; Bash syntax, ShellCheck, YAML parse, package parity, and diff checks pass.
- Independent cumulative review found no P0/P1 blocker. Non-blocking residuals are mutable upstream/base inputs and intentionally disabled SBOM/attestation for the current single-manifest Enroot route, site material included in the unpublished local sdist, and policy rather than code enforcement of the private fetch destination. These are recorded as future hardening, not current operational failures.
- The final owner-only config reports Compute2 `READY` on production partition `general-gpu`; the installed seven-tool MCP reports `READY`, the acceptance job `SUCCEEDED`, and the sanitized-diagnostics marker. Exact installed fetch automatically verified the 21-file output and provenance chain. Direct Slurm reconciliation on 2026-08-11 shows CPU `2721167` and H100 `2721168` both `COMPLETED 0:0` with no active allocation.
- Canonical vault documentation was promoted from `VERIFYING` to `READY`; `qmd update`, `qmd embed`, and exact search returned `qmd://vault/computing/containers.md` with the READY callout.

**Continuation guard:** stop_allowed=true after the completion commit/push and
strict landing check; remaining_batches=0; merging remains unauthorized.
