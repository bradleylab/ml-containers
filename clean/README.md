# clean — CLEAN enzyme EC-number prediction

Assigns Enzyme Commission (EC) numbers to amino-acid sequences. CLEAN
(Contrastive Learning–Enabled Enzyme Annotation) embeds a sequence with
ESM-1b, projects the embedding through a contrastively-trained network, and
assigns EC numbers by distance to pre-computed EC cluster centres. One
sequence can receive several EC numbers.

- Upstream: https://github.com/tttianhao/CLEAN — last commit 2025-04-06, last
  release v1.0.1 (2023-03-31). Low activity, not archived.
- Paper: Yu et al. (2023), *Science* 379:1358 —
  [doi:10.1126/science.adf2465](https://doi.org/10.1126/science.adf2465)

> **Status: experimental.** Not yet benchmarked on lab data.

## Licence — research use only, and not MIT

Read this before using the image for anything.

GitHub's repository metadata advertises MIT. **The tree contains no LICENSE
file.** The only licence artifact upstream ships is
`NON-EXCLUSIVE RESEARCH USE LICENSE FOR CLEAN SOFTWARE.pdf`. Research-use-only
is not MIT, and the two cannot both be true.

This image is therefore built and labelled under the research-use reading:

- `org.opencontainers.image.licenses="LicenseRef-CLEAN-Non-Exclusive-Research-Use"`
- `bradleylab.model.use_restriction="research-only"`

The image is **not** labelled MIT, deliberately. University research use is
accepted as within terms. Anything outside it — a commercial pipeline, a
service offered to third parties, redistribution under a permissive licence —
is not established and needs the authors' written agreement first.

Terms for the *pretrained weights* are stated nowhere at all. They are not
covered by the PDF, which addresses the software. Treat the weights as
research-use-only by the same reading.

## Image tag

`ghcr.io/bradleylab/clean:latest` (also `:v1`, `:torch2.5-cpu`)

## CPU only — do not ask for a GPU

Upstream documents CPU inference as supported and its own Dockerfile installs
CPU-only torch. The documented resource floor is **>12 GB of system RAM**, not
VRAM; the "7.3 GB" figure in upstream's README is the size of the ESM-1b
download, not a memory requirement.

CLEAN's code calls `torch.cuda.is_available()` and will use a GPU if one is
visible, so a GPU allocation will not fail — it will just occupy an H100 for a
job that does not need one. This image ships a CPU-only torch build (asserted
at build time), and the Compute2 recipe below targets `general-cpu`.

## Contents

- `python:3.10-slim` (upstream's manuscript environment was Python 3.10.4).
- CLEAN cloned to `/opt/CLEAN` at a pinned commit and installed with
  upstream's own `python build.py install`. Working directory is
  `/opt/CLEAN/app`, which is where CLEAN's relative paths resolve from.
- `torch 2.5.1+cpu`, `fair-esm 1.0.2`, and upstream's `requirements.txt` pins
  (numpy 1.22.3, pandas 1.4.2, scipy 1.7.3, scikit-learn 1.2.0, matplotlib
  3.7.0, tqdm 4.64.0).
- `facebookresearch/esm` cloned to `/opt/CLEAN/app/esm` at the `v1.0.2` tag.
  This clone is load-bearing, not a duplicate of the pip package: CLEAN shells
  out to `./esm/scripts/extract.py` by a path relative to the working
  directory.
- `TORCH_HOME=/root/.cache/torch` — where ESM-1b lands.
- No wrapper CLI beyond upstream's `CLEAN_infer_fasta.py`.

## Weights — nothing is baked, and getting them is the awkward part

Two independent downloads, from two different places, neither of which is a
package registry.

| What | Size | Source | Mount point in container |
|---|---|---|---|
| CLEAN pretrained weights + EC cluster centres + GMM | ~141 MB zipped | Google Drive **only** | `/opt/CLEAN/app/data/pretrained` |
| ESM-1b checkpoint + contact-regression head | ~7.3 GB | `dl.fbaipublicfiles.com` | `/root/.cache/torch/hub/checkpoints` |

### CLEAN weights — two conflicting Drive file IDs (unresolved)

Upstream cites **two different Google Drive file IDs** for the pretrained
bundle, and does not acknowledge the discrepancy:

| Cited in | File ID |
|---|---|
| `README.md` | `1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ` |
| `Dockerfile` | `1gsxjSf2CtXzgW1XsennTr-TcvSoTSDtk` |

They are probably the same artifact — the Dockerfile's unzip step renames a
directory called `CLEAN_pretrained (2)`, which reads like a re-upload — but
that is a guess, not a fact, and nothing checkable (a checksum, a version
string) is published for either. **This is unresolved.** Whichever bundle you
use, record its file ID and sha256 alongside the results it produced.

After unzipping, the directory mounted at `data/pretrained` must contain at
least `split100.pth`, `100.pt`, and `gmm_ensumble.pkl` (upstream's spelling)
for the default max-separation path. The 70%-identity split adds
`split70.pth` and `70.pt`.

### ESM-1b

```bash
curl -o esm1b_t33_650M_UR50S.pt \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
curl -o esm1b_t33_650M_UR50S-contact-regression.pt \
  https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt
```

Both files go in the directory mounted at
`/root/.cache/torch/hub/checkpoints`. Fetched once, they are reused by every
subsequent job; without them the first inference will try to download 7.3 GB
from inside the job.

### Mirror both to Storage3 before depending on them

A consumer Google Drive link is a single point of failure for a
Science-published pipeline, and `dl.fbaipublicfiles.com` is a bare CDN path
with no versioning. Mirror both to Storage3 with `sha256` manifests before any
analysis depends on them, and point the mounts at the mirror rather than at a
freshly-downloaded copy.

## Running

Input FASTA goes in the directory mounted at `data/inputs`, named
`<name>.fasta`. Output lands in `results/inputs/<name>_maxsep.csv`, one row
per sequence: `SeqID,EC:x.x.x.x/<distance>,...`.

```bash
docker run --rm \
  -v "$PWD/pretrained:/opt/CLEAN/app/data/pretrained" \
  -v "$PWD/inputs:/opt/CLEAN/app/data/inputs" \
  -v "$PWD/esm_data:/opt/CLEAN/app/data/esm_data" \
  -v "$PWD/results:/opt/CLEAN/app/results/inputs" \
  -v "$PWD/torch-cache:/root/.cache/torch" \
  ghcr.io/bradleylab/clean:v1 \
  python CLEAN_infer_fasta.py --fasta_data myseqs
```

`data/esm_data` holds the per-sequence ESM-1b embeddings written during
inference. Mounting it is optional but worthwhile — re-running the same
sequences then skips the expensive embedding step.

Max-separation is the default and the recommended mode: deterministic, no
hyperparameters, better precision/recall than the p-value mode per the
authors, and faster.

### Python API

```python
from CLEAN.infer import infer_maxsep, infer_pvalue

infer_maxsep(
    "split100",              # training split the weights come from
    "inputs/myseqs",         # test data, relative to data/
    report_metrics=False,
    pretrained=True,
    gmm="./data/pretrained/gmm_ensumble.pkl",
)
```

Both functions resolve paths relative to the working directory, so run them
from `/opt/CLEAN/app`. `infer_pvalue` is the alternative assignment mode; use
it only when you specifically want p-value–based calls.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+clean+v1.sqsh 'docker://ghcr.io#bradleylab/clean:v1'
```

Submit to `general-cpu`. There is no `--gpus` line — this is a CPU job.

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

SCRATCH=/scratch2/fs1/alexander.s.bradley
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+clean+v1.sqsh \
     --container-mounts=${SCRATCH}/clean/pretrained:/opt/CLEAN/app/data/pretrained,${SCRATCH}/clean/inputs:/opt/CLEAN/app/data/inputs,${SCRATCH}/clean/esm_data:/opt/CLEAN/app/data/esm_data,${SCRATCH}/clean/results:/opt/CLEAN/app/results/inputs,${SCRATCH}/clean/torch-cache:/root/.cache/torch \
     bash -lc 'export PYTHONNOUSERSITE=1; cd /opt/CLEAN/app && python CLEAN_infer_fasta.py --fasta_data myseqs'
```

`--mem=32G` clears the documented >12 GB floor with headroom for ESM-1b on a
batch of sequences; drop it only after measuring. `PYTHONNOUSERSITE=1` is
required on Compute2 — enroot bind-mounts `$HOME` into the container, so a
stray `pip install --user` on the login node would otherwise shadow the
container's site-packages.

## Build notes / caveats

- **`fair-esm 1.0.2` is chosen over upstream's own pin, deliberately.**
  `app/requirements.txt` pins `fair-esm==2.0.0`; the README prose states the
  manuscript results were produced with `fair-esm 1.0.2`, and warns twice that
  predictions depend on the ESM-1b version. Manuscript fidelity won: this
  image installs 1.0.2 and pins the `facebookresearch/esm` clone to the
  matching `v1.0.2` tag, so `extract.py` and the installed package are the
  same release.

  The tradeoff is real. Anyone reproducing "CLEAN as shipped" from a clean
  checkout gets 2.0.0, and this image will not match them; predictions from
  the two have not been compared here, and upstream publishes no comparison.
  For the as-shipped combination:

  ```bash
  docker build \
    --build-arg FAIR_ESM_VERSION=2.0.0 \
    --build-arg ESM_GIT_SHA=0b59d87ebef95948c735b1f7aad463dc6dfa991b \
    clean/
  ```

  Record which one produced any result that leaves the lab.

- **torch is pinned below 2.6, and that ceiling is load-bearing.** torch 2.6.0
  changed the `torch.load` default to `weights_only=True`, which refuses the
  ESM-1b checkpoint because it pickles an `argparse.Namespace`. Upstream's
  Dockerfile installs torch unpinned, so a build of upstream's recipe today
  fails at first inference rather than at build time. The build-time smoke
  test asserts both the version ceiling and that the wheel is a CPU build.

- **Pinned SHAs.**

  | ARG | Repo | Pinned SHA | Note |
  |---|---|---|---|
  | `CLEAN_GIT_SHA` | `tttianhao/CLEAN` | `f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15` | 2025-04-06, current `main` |
  | `ESM_GIT_SHA` | `facebookresearch/esm` | `839c5b82c6cd9e18baa7a88dcbed3bd4b6d48e47` | tag `v1.0.2` |

  Both are written to files in the image (`/opt/CLEAN/GIT_SHA`,
  `/opt/CLEAN/ESM_GIT_SHA`) before `.git` is dropped, and the smoke test
  asserts they match the ARGs — a drifted pin fails the build rather than
  shipping. Move a pin by editing the ARG defaults and committing; the CI
  workflow passes no build-args, so a re-run reproduces the committed pin.

- **The `esm/` clone shadows the `fair-esm` package.** `/opt/CLEAN/app/esm` is
  a directory with no `__init__.py`, so `import esm` from that working
  directory resolves to it as a namespace package instead of to the installed
  `fair-esm`. This does not affect CLEAN — CLEAN never imports `esm`, it
  shells out to `esm/scripts/extract.py`, which runs with its own directory on
  `sys.path` and picks up the real package. But an interactive
  `python -c "import esm"` from `/opt/CLEAN/app` will look broken. Read
  versions with `importlib.metadata.version("fair-esm")`, which is what the
  smoke test does.

- **`python build.py install` is upstream's install command, and it is
  deprecated.** `build.py` is a plain setuptools script rather than a
  `setup.py`, so pip's modern path does not apply to it unaltered. The legacy
  command still works (verified on setuptools 84), and the Dockerfile keeps
  it rather than diverging from upstream — with `setuptools<85` pinned, so
  that the day setuptools finally removes the `install` command, the pin is
  the thing that has to be revisited rather than the build silently breaking.
  The install registers as `CLEAN 0.1` in package metadata.

- **Sequence length.** ESM-1b has a 1024-token limit. Longer sequences are an
  upstream concern, not something this image changes; check `extract.py`
  behaviour before feeding it long proteins.

- **Not build-verified locally.** The image has not been built on this
  machine — Docker was unavailable — so CI is the first real build. The old
  dependency pins (numpy 1.22.3, scipy 1.7.3) all have cp310 manylinux wheels,
  so no compiler toolchain is installed; if a wheel turns out to be missing,
  the fix is to add `build-essential`, not to float the pin.
