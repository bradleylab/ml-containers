# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

## Mission

Make AlphaFold 3 usable through the trusted `af3` interface by publishing the
software-only image, verifying Storage3, passing a Compute2 H100 smoke, and
completing one minimal prediction with immutable provenance.

## Run Control

- **Run mode:** finite
- **Stop policy:** blocker-only
- **User intent:** "make a goal to get it working and run a small working test. Use beads if necessary to follow a plan. Use elves to keep moving"
- **Checkpoint:** none; continue until acceptance or a genuine blocker
- **Workspace ownership:** dedicated worktree `/Users/abradley/Documents/AI_projects/AlphaFold/.worktrees/ml-containers-alphafold3`, branch `feat/alphafold3-container`
- **Branch tip at start:** `9dc06133025b4c712b6864f016b8e0a6dd49d4f3`
- **Merge policy:** user merges; Elves may not merge
- **Final-response policy:** disallowed while work remains and the Stop Gate says no
- **Coordination mode:** Cobbler-first with host-native implementation
- **E2E mode:** chat-to-work
- **Work driver:** host-native
- **Implementation lane:** fast
- **Delegation scope:** none during implementation; independent terminal review allowed
- **Git mode:** host_only
- **Driver monitor mode:** interactive
- **Driver review policy:** final independent review only
- **Risk posture:** high
- **Trust mode:** trusted
- **Landing outcome:** landable branch; PR posting requires its own approval
- **Driver merge authorized:** no
- **Worker merge authority:** false
- **Worker packet:** n/a — host-native
- **Staging acceptance command:** `python3 /Users/abradley/.agents/skills/elves/scripts/acceptance_contract.py validate --repo-root . --session .elves-session.json`
- **Staging acceptance validation:** PASS
- **High-risk checkpoints:** GHCR manifest compatibility; first billable H100 job; end-to-end output verification
- **GitHub push auth route:** host SSH/keyring
- **Continuation harness:** Codex Goal plus host-native Elves loop
- **Continuation rule:** if work remains and no true blocker exists, continue without waiting for acknowledgment

## Cobbler Session State

- **Cobbler default:** on
- **Activated by:** explicit Elves invocation
- **Scope:** current Elves run
- **Behavior:** fit planning, risk, debugging, and review decisions before acting
- **Persistence:** this guide and `.elves-session.json`

## Stop Gate

- **Planned batches remaining:** 5
- **Stop allowed right now:** no
- **Why:** no end-to-end H100 prediction has succeeded yet
- **Next required action:** close B1 Storage3 installation evidence while preparing the branch-build container candidate

## Current Phase

- **Status:** staging and B1 in progress
- **Active batch:** B1 — Close the restricted Storage3 installation
- **What was just finished:** goal created, Elves policy loaded, and live readiness gaps reconciled
- **Single next action:** validate and record the database install, then publish the branch candidate

## Active Compute

| Resource | Purpose | Current status | Last verified | Stop / repurpose trigger |
| --- | --- | --- | --- | --- |
| Compute2 `2715132` | Install and hash AF3 reference databases | Running on `c2-node-102` | 2026-08-10 21:27 CDT | Ends after manifest and atomic current-link promotion |
| Compute2 `2715106` | Free-pool PCI identity follow-up | Pending, non-production | 2026-08-10 21:27 CDT | Cancel during B1 resource hygiene; not needed for H100 acceptance |

## Next Exact Batch

**Batch:** B1 — Close the restricted Storage3 installation

**Acceptance criteria:**

- [ ] [B1-A1] Database job completes cleanly and current link resolves.
- [ ] [B1-A2] Database content manifest and required artifacts verify.
- [ ] [B1-A3] Restricted model release/hash/modes revalidate.

**Risk:** the installer may fail late during extraction or full hashing.

## Non-negotiables

- Never place restricted model/data/input/output bytes in Git, GHCR, prompts, or public logs.
- Never claim `READY` without the successful end-to-end H100 prediction.
- Never run analysis on a Compute2 login node; every job carries the exact account.
- Never merge; never use destructive Git operations; never weaken tests to get green.
- Reconcile active paid/long-running compute after every topology change.

## Tool Configuration

- Static image checks: Python compile, Ruff, YAML parse, Dockerfile inspection.
- Runner checks: `uv run pytest`, `uv run ruff check .`, `uv build` from the AlphaFold project.
- Runtime checks: scheduled Slurm smoke and end-to-end jobs only.
- Evidence root: `/private/tmp/alphafold3-elves-evidence`.

## Plan and Log Paths

- **Plan:** `docs/plans/alphafold3-compute2-ready.md`
- **Learnings:** `learnings.md`
- **Execution log:** `EXECUTION_LOG.md`
- **Session:** `.elves-session.json`
- **Branch:** `feat/alphafold3-container`
- **PR:** not created; external PR text requires separate approval
- **Plan hash at session start:** `9a604bca02ee1194a96d8c7b2381ed7e`

## After Any Compaction

Read this guide, `.elves-session.json`, `learnings.md`, the plan, and
`EXECUTION_LOG.md`; reconcile Active Compute; then execute the single next
action. Do not redo completed evidence.
