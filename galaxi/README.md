# galaxi

Multiphase identification from powder X-ray diffraction.

Tong, Jin, Xu, Rao, Jiang and Szymanski (2026),
[arXiv:2609.06908](https://arxiv.org/abs/2609.06908).
Upstream: [Szymanski-Group/galaxi](https://github.com/Szymanski-Group/galaxi).

## What this is

GALAXI trains one independent binary classifier per phase rather than a single
multi-class model. The classifiers narrow a pattern to a few plausible phases,
and Rietveld refinement settles which combination explains the full pattern.
Reported micro-F1 0.935 on experimental patterns, holding up under low impurity
fractions, small crystallite size, peak shifts, sample displacement and texture.

Because the per-phase models are independent, **adding a phase means training
one more classifier and leaving the rest alone**. That is what this image is
for here: building a catalog for the minerals this lab actually encounters.

`xrd-classifier` (autoXRD) does the same job by the older single-model route,
and GALAXI's paper benchmarks against that lineage. Both images stay. Run them
on the same patterns before retiring anything.

## The image is code only — three assets stage beside it

| Asset | Size | Needed for | Source |
|---|---|---|---|
| COD reference structures | ~5.5 GB | generating training data | `galaxi-setup-cod` |
| background / negative profiles | ~670 MB | generating training data | `galaxi-setup-bg-profiles` |
| 365-phase pretrained catalog | 618 MB | trying the evaluation path | [figshare](https://doi.org/10.6084/m9.figshare.33360183) |

Neither of the first two is needed to *evaluate* a pattern against models you
already have. The third is upstream's worked example, and its chemistry is
batteries and oxides rather than rock-forming minerals — useful for checking
the install end to end, less so for geology.

**Everything the 64,594-structure library gives you stays on their website.**
`galaxi-xrd.com` serves it; it is not distributed. Using it means sending
patterns to a third-party service.

## Where the data goes, and the trap it avoids

GALAXI resolves its data paths from the environment before falling back to
`~/.local/share/galaxi`. That fallback is a trap under enroot, which
bind-mounts `$HOME`: a host directory would silently shadow anything staged.
So the image sets

```
GALAXI_COD_DIR=/data/cod
GALAXI_BG_PROFILES=/data/bg_profiles/public_phase_profile_50000.h5
```

and you mount the staged directory onto `/data`.

**Note the asymmetry — it costs a staging run to learn.** `GALAXI_COD_DIR`
names a *directory*; `GALAXI_BG_PROFILES` names the HDF5 *file* itself,
702,781,444 bytes. Point the second at a directory and the setup script decides
the library is already installed, then fails to open it as HDF5. Stage once to Storage3
from a **login node** — the COD download runs through `gdown` against Google
Drive, which throttles non-interactive clients, the same failure `kpconv`
documents for its weights.

```bash
GALAXI_COD_DIR=/storage3/.../galaxi/cod \
GALAXI_BG_PROFILES=/storage3/.../galaxi/bg_profiles/public_phase_profile_50000.h5 \
  galaxi-setup-cod --temp-dir /storage3/.../galaxi/tmp && galaxi-setup-bg-profiles
```

`galaxi-setup-cod` needs `--temp-dir` on a path with room for 5.5 GB; a
container's default temp has nowhere near that. `galaxi-setup-bg-profiles`
takes no such flag — it writes beside its destination, which is already on
Storage3.

Verify either later with `--verify-only`.

## Training your own phases

Put the CIFs for your target phases in a `References/` directory — your own
files, or query them with `galaxi.CODQuery` — then:

```python
from galaxi.workflows.streamlined_workflow import StreamlinedWorkflow, create_default_config

config = create_default_config()        # writes workflow_config.json
workflow = StreamlinedWorkflow(config=config)

phases = ["Quartz_1011097"]             # matching CIF filenames in References/

workflow.step_1_generate_training_data(phases=phases)
workflow.step_2_train_models(phases=phases)
workflow.step_3_evaluate_experimental_patterns()
```

Or in one shot:

```bash
galaxi-workflow --create-config
galaxi-workflow --config workflow_config.json
```

`workflow_config.json` carries the simulation ranges (angle, shift, strain,
texture, noise), the CNN architecture and training hyperparameters, and the
evaluation threshold. Every key is commented in the file it writes.

Steps 4 and 5 build a synthetic test set spanning individual artifact types and
score the trained models against it — worth running before trusting a new
classifier on real samples.

The repository's `tutorials/` notebooks are kept at `/opt/galaxi/tutorials/`:
`01_basic_pattern_generation` walks through the physical simulation,
`02_model_training` trains one detection model end to end.

## Stack

- Base `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (Python 3.11)
- `pymatgen`, `pyxtal`, `scikit-image`, `zarr`, `adabelief-pytorch`
- DARA, the refinement engine, is pinned to a **fork** rather than upstream
  (`cuzno200161/dara` at an exact commit) because GALAXI's config uses a
  phase-grouping metric only that fork provides. Pinned by SHA, but worth
  knowing it is an individual's fork rather than the upstream project.
- GPU optional: the detection models are small 1D CNNs and run on CPU, but
  training a catalog of many phases is where a GPU earns its place.

Upstream declares `requires-python >= 3.8` and tests on 3.10; this image runs
3.11.

Pull: `ghcr.io/bradleylab/galaxi:v1`

## Status

Nothing here has been run on lab data. The build smoke test simulates a
diffraction pattern from a CIF the repository ships, which exercises the
crystallography stack but says nothing about identification accuracy on real
samples.
