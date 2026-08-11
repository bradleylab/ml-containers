# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

## Mission

Make AlphaFold 3 usable through the trusted `af3` interface by publishing the
software-only image, verifying Storage3, passing a Compute2 H100 smoke, and
completing one minimal prediction with immutable provenance.

## Run Control

- **Run mode:** finite
- **Stop policy:** blocker-only
- **User intent:** "make a goal to get it working and run a small working test. Use beads if necessary to follow a plan. Use elves to keep moving"
- **Checkpoint due by:** none
- **Checkpoint semantics:** none
- **May continue after checkpoint:** yes
- **Actual stop conditions:** all master acceptance criteria pass, no AlphaFold compute remains active, or a genuine blocker or explicit user stop occurs
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
- **Driver update policy:** interactive
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
- **Re-drive budget:** n/a — host-native
- **Continuation harness:** Codex Goal plus host-native Elves loop
- **Continuation rule:** if work remains and no true blocker exists, continue without waiting for acknowledgment
- **Batch completion rule:** Every completed batch must end with a commit and push.
- **Re-read rule:** re-read this survival guide before doing anything else after every host-owned commit and push
- **Checkpoint rule:** a checkpoint is progress evidence, not permission to stop

## Cobbler Session State

- **Cobbler default:** on
- **Activated by:** explicit Elves invocation
- **Scope:** current Elves run
- **Behavior:** fit planning, risk, debugging, and review decisions before acting
- **Persistence:** this guide and `.elves-session.json`
- **Exit phrases:** "Cobbler Mode: off", "leave Cobbler Mode", or "stop using Cobbler by default"

## Stop Gate

- **Planned batches remaining:** 0
- **Stop allowed right now:** yes, after the required completion commit/push and strict landing check
- **Why:** all batch and master acceptance criteria have evidence; no AlphaFold compute remains active
- **Next required action:** commit and push the completed B5 evidence, re-read this guide, and run the strict landing check; do not merge

## Effort Standard

- Work as hard as you can for the full run and maintain the same verification depth through the final landing batch.
- Do not settle for the minimum acceptable change or a green individual gate while later acceptance work remains.
- Continue with the next highest-value action until the Stop Gate permits stopping.

## Forbidden Stop Reasons

- A checkpoint, commit, push, green CI run, or useful summary completed.
- The user is silent or the current batch reached a clean boundary.
- The remaining review, documentation, or reconciliation work feels large.

## Post-Checkpoint Control Loop

Every completed batch must end with a commit and push. After every host commit
and push, re-read this survival guide before doing anything else and answer:
what remains, what paid or long-running resources are active, whether any
active resource is idle or ambiguous, and whether user scope changed. Does the
Stop Gate still say `Stop allowed right now: no`, or does `.elves-session.json`
still forbid stopping? If so, continue immediately.

## Current Phase

- **Status:** B5 complete; all master acceptance criteria met
- **Active batch:** none
- **What was just finished:** final wheel/MCP/CLI readiness smoke, 30-test and static gates, independent cumulative review, QMD vault retrieval, and Slurm reconciliation all passed
- **Single next action:** commit and push the completed B5 evidence, re-read this guide, and run the strict Elves landing check

## Active Compute

| Resource | Purpose | Current status | Last verified | Stop / repurpose trigger |
| --- | --- | --- | --- | --- |
| None | No AlphaFold compute remains active | Acceptance jobs `2721167`/`2721168` completed `0:0` | 2026-08-11 | Recheck only if new submission occurs |

## Next Exact Batch

**Batch:** B5 — Expose verified agent access

**Scope:** finish durable project/vault/container documentation, final local
checks, independent cumulative review, Elves landing validation, and active
compute reconciliation without opening or merging a PR.

**Acceptance criteria:**

- [x] [B5-A1] `af3 capabilities` reports Compute2 `READY` with immutable identities.
- [x] [B5-A2] Registered stdio MCP exposes seven tools and capability call succeeds.
- [x] [B5-A3] Project and canonical vault documentation are current.
- [x] [B5-A4] Final tests, independent review, and Elves landing checks pass.

**Risk:** configuration or documentation drift could overstate the verified route.

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

Read this guide including the Run Control section and Stop Gate,
`.elves-session.json` including `continuation_guard`, `learnings.md`, the plan,
and `EXECUTION_LOG.md`; reconcile Active Compute; then execute the single next
action. Do not redo completed evidence.

## Launch Readiness

- [x] Stop Gate initialized with `Stop allowed right now: no` and consistent with `.elves-session.json`.
