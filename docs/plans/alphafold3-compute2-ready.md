# Plan: AlphaFold 3 Compute2 readiness

## Mission

Publish a software-only AlphaFold 3 image to Bradley Lab GHCR, bind it to the
restricted model and reference-database releases on Storage3, and execute one
small successful prediction on a scheduled Compute2 H100. The runner is marked
`READY` only after the output and immutable provenance are verified.

## Scope

### In scope

- Complete and verify the current Storage3 model/database releases.
- Publish and inspect `ghcr.io/bradleylab/alphafold3` without restricted assets.
- Import the immutable image into Compute2's enroot cache.
- Run a scheduled H100 model-load smoke and one minimal prediction.
- Bind the verified identities into the CLI/MCP configuration and documentation.

### Out of scope

- Modal or EC2 provisioning.
- Clinical, commercial, or real research interpretation of the smoke output.
- Redistribution of model parameters, databases, inputs, or outputs.
- Merging the feature branch without separate user authorization.

## Batches

### Batch 1 [B1]: Close the restricted Storage3 installation

**Tasks:**

- Monitor the reference-database installer to completion.
- Verify release symlinks, expected files, permissions, and immutable manifests.
- Reconcile the model/database state into the project status record.

**Acceptance criteria:**

- [ ] [B1-A1] Database job `2715132` completes with exit code 0 and `databases/current` resolves to release `af3-db-v3.0-2026-08-10`.
- [ ] [B1-A2] A deterministic database content manifest and its SHA-256 are recorded, and every installer-required database artifact is non-empty.
- [ ] [B1-A3] The model release resolves through `models/current`, matches the recorded SHA-256, and retains the restricted directory and file modes.

**Risk:** standard — the long download/extraction can fail late or expose an incomplete release.
**Caution:** never advance `databases/current` before all checks pass.
**Affected surfaces:** Storage3 AlphaFold directory and local status records.
**Constitution impacts:** restricted model/data boundary.
**Review focus:** atomic release promotion, manifests, and permissions.
**Focused tests:** scheduled installer exit state, link resolution, manifest verification.
**Depends on:** none.

### Batch 2 [B2]: Publish a Compute2-compatible GHCR candidate

**Tasks:**

- Validate the Dockerfile, offline smoke, workflow, documentation, and secret scan.
- Publish a branch SHA candidate through GitHub Actions.
- Record and inspect the immutable digest and manifest type.

**Acceptance criteria:**

- [ ] [B2-A1] Static validation passes for the Dockerfile, Python smoke test, YAML workflow, ShellCheck-relevant scripts, and repository diff.
- [ ] [B2-A2] GitHub Actions publishes a `linux/amd64` SHA-tagged candidate that Compute2 enroot can import as a plain single-architecture image manifest.
- [ ] [B2-A3] The GHCR digest is recorded and inspection confirms the pinned AlphaFold 3 source identity and absence of model parameters, databases, inputs, and outputs.

**Risk:** high — GitHub build resources and enroot manifest compatibility are external gates.
**Caution:** buildx provenance/SBOM attestations create an OCI index that enroot 3.4.1 cannot import.
**Affected surfaces:** `alphafold3/`, its workflow, catalog README, and model cards.
**Constitution impacts:** software-only public artifact boundary.
**Review focus:** deterministic pins, licenses, image manifest, and restricted-asset exclusion.
**Focused tests:** Ruff/Python compile, YAML parse, Dockerfile checks, Actions build, GHCR inspection.
**Depends on:** none.

### Batch 3 [B3]: Pass the scheduled Compute2 H100 smoke

**Tasks:**

- Import the immutable GHCR image into the Storage3 enroot cache.
- Run the scheduled `general-gpu` H100 smoke.
- Record GPU, driver, CUDA/JAX device, image, and model-load evidence.

**Acceptance criteria:**

- [ ] [B3-A1] The immutable GHCR digest imports successfully into a named `.sqsh` cache on Compute2.
- [ ] [B3-A2] A `general-gpu` job identifies an H100, initializes exactly one JAX GPU, and loads a non-empty AlphaFold model parameter tree.
- [ ] [B3-A3] The smoke record binds the Slurm job ID, node, GPU identity, image digest, model release/hash, and successful exit code.

**Risk:** high — this is the first billable accelerator and runtime compatibility gate.
**Caution:** production acceptance must run on `general-gpu`, not the heterogeneous free pool.
**Affected surfaces:** Compute2 enroot cache, Scratch2 logs, and deployment manifests.
**Constitution impacts:** billable compute approval and reproducibility boundary.
**Review focus:** actual hardware identity and immutable image/model binding.
**Focused tests:** `smoke_gpu.sbatch` plus log/manifests inspection.
**Depends on:** B2.

### Batch 4 [B4]: Execute a minimal end-to-end prediction

**Tasks:**

- Select a small public or upstream-derived schema-valid smoke input.
- Run the CPU data pipeline followed by dependent H100 inference.
- Verify non-empty structure/confidence outputs and provenance manifests.

**Acceptance criteria:**

- [ ] [B4-A1] The exact smoke input validates without scientific-field mutation and its SHA-256 is recorded.
- [ ] [B4-A2] The CPU data-pipeline job and dependent H100 inference job submitted through the controlled bootstrap acceptance path both complete with exit code 0.
- [ ] [B4-A3] The run produces non-empty AlphaFold structure and confidence artifacts with a deterministic output manifest.
- [ ] [B4-A4] The run manifest binds input, image, model, database, GPU, and Slurm identities without exposing restricted bytes.

**Risk:** high — end-to-end database search, compilation, and inference can expose integration failures.
**Caution:** this is a technical smoke only; do not interpret it scientifically.
**Affected surfaces:** trusted Scratch2 run directory and local deployment evidence.
**Constitution impacts:** scientific-input immutability and output terms.
**Review focus:** dependency chaining, failure propagation, and artifact verification.
**Focused tests:** `af3 validate`, controlled Slurm submission, Slurm accounting, output manifest checks.
**Depends on:** B1 and B3.

### Batch 5 [B5]: Expose verified agent access

**Tasks:**

- Write a resolved local configuration with immutable identities and `READY` state.
- Register and exercise the trusted-Mac MCP server.
- Update project/vault documentation and perform terminal review.

**Acceptance criteria:**

- [ ] [B5-A1] `af3 capabilities` reports Compute2 `READY` with immutable image, model, and database identities; Modal and EC2 remain `UNCONFIGURED`.
- [ ] [B5-A2] The registered local MCP exposes the seven bounded AlphaFold tools and a read-only capability call succeeds.
- [ ] [B5-A3] Project status, agent contract, methods/provenance, and canonical vault documentation describe the verified route and residual restrictions accurately.
- [ ] [B5-A4] Focused tests, terminal cumulative review, and the Elves acceptance/landing checks pass at the final evidence tip.

**Risk:** standard — configuration drift could make agents report readiness unsupported by live evidence.
**Caution:** never encode credentials or restricted assets in MCP configuration or logs.
**Affected surfaces:** AlphaFold runner configuration, MCP registration, project docs, and vault note.
**Constitution impacts:** trusted-agent boundary and accurate readiness claims.
**Review focus:** capability truthfulness, tool bounds, and documentation consistency.
**Focused tests:** runner tests, CLI/MCP parity, QMD retrieval, cumulative diff review.
**Depends on:** B4.

## Master Acceptance

- [ ] [M-A1] One small AlphaFold 3 prediction completes successfully on a Compute2 H100 and produces verified non-empty structure and confidence outputs.
- [ ] [M-A2] The run records immutable input, software image, model, database, GPU, scheduler, and output identities without redistributing restricted assets.
- [ ] [M-A3] Trusted agents can discover and call the bounded `af3` interface, while unverified or unconfigured backends remain unavailable.
- [ ] [M-A4] All local gates and terminal review pass, documentation is current, and no billable or long-running compute is left idle.

## Non-negotiables

- Model parameters, databases, private inputs, and outputs never enter Git, GHCR, prompts, or public logs.
- Compute2 jobs use `-A compute2-alexander.s.bradley`; analysis runs only under Slurm.
- `READY` requires the verified end-to-end H100 result, not installation or publication alone.
- The smoke input is never silently modified and the output is not used clinically or interpreted as research.
- Elves may commit and push only the feature branch; it may not merge without separate user authorization.

## Test strategy

- **Static gate:** Python compile/Ruff, YAML parse, ShellCheck, Dockerfile and restricted-asset inspection.
- **Container gate:** GitHub Actions build plus immutable GHCR manifest inspection.
- **Runtime gate:** scheduled H100/JAX/model-load smoke on `general-gpu`.
- **E2E gate:** chained CPU pipeline and H100 inference with artifact/manifests verification.
- **Agent gate:** existing focused runner tests plus CLI/MCP parity and capability-state checks.

## Notes

- Source recipe target is `bradleylab/ml-containers`; GHCR is canonical and Compute2 `.sqsh` files are cache.
- The AlphaFold project root on the Mac contains the runner and live deployment records.
- This is a finite, blocker-only run with a persistent Codex Goal and no deadline checkpoint.
