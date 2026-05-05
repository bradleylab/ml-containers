# raman-classifier

Raman mineral identification via nearest-neighbour matching against
the RRUFF reference library. Spectrum in, top-k mineral candidates
out, with cosine similarity scores. No learned weights — preprocessing
and matching are deterministic and defensible.

This is **Path A** of the long-deferred raman-classifier slot. Path B
(an in-house Liu-2017-style 1D-CNN trained on RRUFF, weights deposited
at Zenodo + HF Hub under Apache-2) remains queued as a follow-up; this
container's matching code becomes the inference harness for Path B's
weights when they exist.

## Image tag

`ghcr.io/bradleylab/raman-classifier:v1` (also `:latest`,
`:rruff-excellent-cpu`).

Multi-arch: `linux/amd64` + `linux/arm64`. Apple-Silicon Macs pull
native arm64; Compute2 / EC2 pull amd64. ramanspy and its dependencies
(numpy, scipy, pybaselines, scikit-learn) all publish aarch64 wheels.

## Stack

- Base: `python:3.11-slim`
- `numpy>=1.26,<2.3`, `scipy>=1.11`
- `ramanspy>=0.2` ([Georgiev et al. 2024, *Anal. Chem.*](https://doi.org/10.1021/acs.analchem.4c00383), BSD-3)

## Reference library

Baked at build time:

- **RRUFF `excellent_unoriented`** archive (~229 MB), pulled from
  https://www.rruff.net/zipped_data_files/raman/ via ramanspy's
  `rp.datasets.rruff('excellent_unoriented')` loader.
- Each reference is preprocessed (Whitaker-Hayes despike → SavGol
  denoise → ASLS baseline → vector L2-normalisation) and resampled
  onto a 100-1500 cm⁻¹ grid at 1 cm⁻¹ resolution.
- Result is a compressed numpy index at `/opt/rruff_index.npz`
  (typically ~30-50 MB) with arrays:

  | key | shape | dtype |
  |---|---|---|
  | `wavenumbers` | (W,) | float32 |
  | `intensities` | (N, W) L2-normalised | float32 |
  | `names` | (N,) | object (str) |
  | `rruff_ids` | (N,) | object (str) |
  | `lasers` | (N,) | object (str) |

The other RRUFF archives (`fair_unoriented`, `excellent_oriented`,
`poor_unoriented`, `unrated_*`, `LR-Raman`) are not pulled by default
to keep the image lean. Rebuild with extra `--dataset` flags inside
the container if needed.

### Citation for the reference data

Lafuente B, Downs RT, Yang H, Stone N (2015). The power of databases:
the RRUFF project. In: *Highlights in Mineralogical Crystallography*,
T Armbruster & RM Danisi, eds., De Gruyter, Berlin, 1-30.

The RRUFF project does not post an explicit Creative Commons license
on the download page; the documented expectation is to cite Lafuente
et al. when redistributing or publishing analyses derived from the
data. This image redistributes the preprocessed spectra in numerical
form alongside the citation.

## Inference

CSV / whitespace-delimited 2-column input (wavenumber cm⁻¹,
intensity arbitrary units). Comments (`#` lines) are skipped.

```bash
docker run --rm \
  -v "$PWD:/work" \
  ghcr.io/bradleylab/raman-classifier:v1 \
  python /opt/scripts/raman_match.py \
    --spectrum /work/unknown.txt \
    --top-k 10 \
    --out /work/matches.csv
```

Output columns: `rank`, `mineral`, `rruff_id`, `laser`, `cosine`, `sad_rad`.
`cosine` is dot product of L2-normalised vectors (range −1..1, but
positive in practice on Raman). `sad_rad` is the spectral angle
distance (`arccos(cosine)`), reported for compatibility with the
ramanspy / EO-spectral conventions.

## Inputs

- 2-column text file: wavenumber (cm⁻¹), intensity. Comma- or
  whitespace-delimited. Comment lines starting with `#` skipped.
- Wavenumber range can be anything; the matcher resamples onto the
  index's grid (default 100-1500 cm⁻¹). Spectra entirely outside that
  window will return uninformative scores.
- Intensity scale is irrelevant — preprocessing L2-normalises.

## Limitations

- **Fingerprint region only.** OH/H₂O stretch peaks (3000-3700 cm⁻¹)
  are excluded from the index by default. Re-build with
  `--wavenumber-max 3700` to include them; no expected accuracy gain
  for typical rock-forming silicates / carbonates / oxides.
- **Single-spectrum input.** Mixed mineralogy (multi-phase samples)
  is not handled — top-k will tend to surface the dominant phase plus
  near-neighbours.
- **`excellent_unoriented` only by default.** ~1500 spectra covering
  a few hundred minerals. For rarer phases or specific orientation
  effects, broaden the index at rebuild time.
- **No uncertainty calibration.** Cosine ≥ 0.9 is typically a confident
  match; 0.7-0.9 is plausible; below 0.7 is suspicious. These
  thresholds are heuristic — use top-k and human judgement.

## Run on Compute2

CPU-trivial — single-spectrum match is sub-second after the index
loads (~0.5 s startup). For batch identification across hundreds of
spectra, submit a CPU job array on `general-cpu` and bind-mount the
spectra directory:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=2 \
       --mem=4G \
       --time=01:00:00 \
       --array=0-99 \
       --wrap='srun --container-image=$IMG \
         --container-mounts=/scratch2/fs1/$USER:/scratch2/fs1/$USER \
         --container-workdir=/work \
         bash -lc "export PYTHONNOUSERSITE=1; \
                   python /opt/scripts/raman_match.py \
                     --spectrum /scratch2/fs1/$USER/raman/${SLURM_ARRAY_TASK_ID}.txt \
                     --out /scratch2/fs1/$USER/raman/results/${SLURM_ARRAY_TASK_ID}.csv"'
```

## Caveats / future work

- **Path B (1D-CNN) follow-up.** Liu et al. (2017, *Analyst*) report
  >88% top-1 on a held-out RRUFF split with a small 1D-CNN. Training
  is ~1 day on CPU. Once trained, weights would be deposited at
  Zenodo + HF Hub under Apache-2 and a `:v2-cnn` tag added that uses
  the CNN as the matcher (this image's preprocessing + index format
  remain the basis).
- **Mixed-phase support.** Could be added via greedy peak-residual
  matching: find best single match, subtract its scaled contribution,
  re-match, etc. Out of scope for v1.
- **Calibration.** A held-out RRUFF cross-validation would let us
  publish recommended cosine thresholds with false-positive rates.
  Out of scope for v1; the heuristic thresholds in "Limitations"
  above are sufficient for triage.
