# AlphaFold 3

Version-pinned software-only container for Google DeepMind AlphaFold 3
`v3.0.4`, pinned to commit
`85c4d20505fd5cef05eac22b534d4e793971ae69`.

Image: `ghcr.io/bradleylab/alphafold3`

## Private-asset boundary

The image contains the Apache-2.0 AlphaFold source, pinned Python/JAX/CUDA
runtime, patched HMMER 3.4, and generated chemical-component data. It does
not contain model parameters, genetic reference databases, inputs, outputs,
credentials, or site configuration.

The image retains the upstream HMMER, Easel, and libdivsufsort license notices
under `/usr/share/licenses/hmmer`. AlphaFold code is Apache-2.0; those bundled
native components use their documented BSD- and MIT-style terms.

Model parameters must be obtained directly from Google and used under the
[AlphaFold 3 Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/v3.0.4/WEIGHTS_TERMS_OF_USE.md).
Generated outputs are subject to the
[AlphaFold 3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/v3.0.4/OUTPUT_TERMS_OF_USE.md).

## Tags

- `sha-<ml-containers-commit>`: candidate image for an exact recipe commit.

The build workflow intentionally publishes no mutable or stable aliases.
Stable aliases require a separate reviewed release action after runtime
acceptance and must never overwrite an existing `vN` tag.

Scientific jobs must record and run the immutable image digest, not only a
mutable tag.

## Compute2

The lab runner imports the image into an enroot `.sqsh` cache and mounts:

- `/storage3/fs1/alexander.s.bradley/Active/alphafold3/models/current`
  at `/models` (read-only);
- `/storage3/fs1/alexander.s.bradley/Active/alphafold3/databases/current`
  at `/databases` (read-only for inference; validated permissions for the data
  pipeline);
- a generated Scratch2 job directory at `/work`.

The image itself is backend-neutral and can also be selected by immutable
digest from Modal or EC2 after those backends are explicitly configured.

## Build checks

The offline build smoke test verifies installed package metadata, patched HMMER
executables, generated CCD pickles, and `run_alphafold.py --help`. GPU/JAX
device initialization and model loading require a scheduled GPU job and are
separate deployment gates.

## Runtime acceptance

The candidate built by GitHub Actions run `31454889316` has immutable digest
`sha256:6c6d8a36f5a9bd204a446b5e62cbd46780fcbf8ee4767763f7fa16d28a02f881`.
Compute2 `general-gpu` job `2720279` verified an H100 80GB HBM3, one JAX GPU,
and successful model-parameter loading. End-to-end ubiquitin acceptance used
the installed agent CLI: job `af3-20260811T043457Z-feab9ac0`, CPU pipeline
`2721167`, and H100 inference `2721168`. Every file in the 21-file fetched
output manifest verified against SHA-256
`5204e688c4a8c7dc0bcb7983a2230417f7e62b326f3420c0c6cecabd5fc71eca`.
Private model, database, input, and output bytes remain outside this repository
and GHCR.

## Upstream support boundary

AlphaFold 3 officially supports and tests numerical accuracy on one NVIDIA
A100 80 GB or one NVIDIA H100 80 GB. Other accelerators must be recorded as
experimental until independently validated.
