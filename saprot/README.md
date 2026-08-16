# saprot — SaProt structure-aware protein language model

Protein language model from Westlake University. Where ESM tokenizes a
protein one amino acid at a time, SaProt tokenizes it one *residue-state* at
a time: each token pairs the amino acid with that residue's foldseek 3Di
structural state, so a 100-residue protein is 100 tokens of the form `Aq`,
`Md`, `Gp`. The structural half can be masked (`A#`), which is how the 1.3B
checkpoint takes sequence-only input.

- Upstream: https://github.com/westlake-repl/SaProt (code MIT). Last commit
  2026-03-08 (`e91e4858`); the project has never cut a GitHub release.
- Weights: https://huggingface.co/westlake-repl — MIT and **ungated**. The
  cleanest license situation of the protein models in this repo.
- Paper: "SaProt: Protein Language Modeling with Structure-aware Vocabulary",
  preprint https://doi.org/10.1101/2023.10.01.560349; journal version in
  Nature Biotechnology, 2025-10-24.
- foldseek: https://github.com/steineggerlab/foldseek (GPL-3.0), release
  `10-941cd33` (2025-01-19), bundled — see below.

> **Status: experimental.** Not yet benchmarked on lab data. The default
> target is embedding structures with `SaProt_1.3B_AFDB_OMG_NCBI`.

## Image tag

`ghcr.io/bradleylab/saprot:latest` (also `:v1`, `:torch2.7-cu129`)

## Contents

- NGC PyTorch 25.04: torch 2.7.0a0, CUDA 12.9, Python 3.12
  (`TORCH_CUDA_ARCH_LIST` includes 9.0 → H100).
- Stock `transformers` 4.57.6 — the checkpoints are ordinary ESM-architecture
  HF repos, so `EsmTokenizer` / `EsmForMaskedLM` load them directly. Plus
  `biopython` 1.88 (pLDDT parsing) and `accelerate` 1.14.0 (`device_map=`).
- `foldseek` `10-941cd33` at `/opt/foldseek/bin/foldseek`, symlinked to
  `/usr/local/bin/foldseek`. `$FOLDSEEK_BIN` holds that path.
- `saprot_utils.foldseek_util` — SaProt's structure→3Di helper, installed
  into site-packages from the pinned upstream commit.
- `HF_HOME=/root/.cache/huggingface` — override at runtime to a mounted
  scratch dir so weights persist.
- No wrapper CLI. Upstream ships YAML-driven training/eval scripts and no
  embedding command, so inference runs from a script (see below).

## Why foldseek is in this image

The one-model-per-container rule has an exception for a tool whose output is
the next tool's input, where a container boundary between them is genuinely
costly. This is that case, and about as clear-cut as it gets: SaProt's input
representation does not exist until foldseek has produced it. `get_struc_seq`
shells out to `foldseek structureto3didescriptor` *inside a single function
call* and reads back the temp file it wrote; there is no intermediate
artifact a separate container could hand over without restructuring the call.
foldseek is also not pip-installable, so it cannot arrive as a dependency.

Upstream ships neither the binary nor a reliable way to get it — `bin/README.md`
says only "place the binary here" and the README links an unversioned,
unchecksummed Google Drive file. This image uses the foldseek project's own
release tarball instead.

## Checkpoints and GPU requirements

Nothing is baked into the image. Weights download from the `westlake-repl` HF
org on first `from_pretrained` call.

| Checkpoint | HF ID | Params | Weights | Input |
|---|---|---|---|---|
| **1.3B AFDB+OMG+NCBI** (default) | `westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI` | 1.30B | 5.21 GB | SA tokens **or** AA-only |
| 1.3B AF2 | `westlake-repl/SaProt_1.3B_AF2` | 1.30B | 5.20 GB | SA tokens or AA-only |
| 650M AF2 | `westlake-repl/SaProt_650M_AF2` | 650M | 2.61 GB | SA tokens only |
| 650M PDB | `westlake-repl/SaProt_650M_PDB` | 650M | 2.61 GB | SA tokens only |
| 35M AF2 | `westlake-repl/SaProt_35M_AF2` | 35M | 0.13 GB | SA tokens only |

**The checkpoint choice is forced by our pipeline, not by size.** Upstream
warns that frozen embeddings from the 35M and 650M models are only usable
with structure tokens; the 1.3B models work with both structure tokens and
AA-only sequences. Parts of the pipeline will have no structure, so
`SaProt_1.3B_AFDB_OMG_NCBI` is the one to use — the extra 2.6 GB buys the
ability to embed structureless proteins with the same model as everything
else, which keeps embeddings comparable.

GPU demand is light: 5.21 GB at F32, half that in bf16, so any H100 or A100
is oversized for the model itself. **The bottleneck is foldseek on CPU.**
Size the job by CPU count, not VRAM.

**Fetch one copy of the weights, not two.** Every checkpoint repo ships its
weights twice — `model.safetensors` alongside `pytorch_model.bin` for the
1.3B AFDB model, a SaProt-native `.pt` alongside the `.bin` for the 650M/35M
ones. A bare `snapshot_download` pulls both and doubles transfer and disk:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI",
    allow_patterns=["*.json", "*.txt", "model.safetensors"],
)
```

`from_pretrained` prefers the safetensors file and downloads only what it
needs, so this matters most when pre-staging weights into a scratch cache.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node (point enroot's scratch dirs off the 50 GB home):

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+saprot+v1.sqsh 'docker://ghcr.io#bradleylab/saprot:v1'
```

Submit a single-H100 job with real CPUs behind it. Mount a scratch dir for
the HF weight cache so the 5.21 GB checkpoint is fetched once:

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+saprot+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/root/.cache/huggingface,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; cd /scratch2/fs1/alexander.s.bradley/saprot_work; python /scratch2/fs1/alexander.s.bradley/scripts/saprot_embed.py'
```

Three lines there are load-bearing:

- `PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
  into the container, so a stray `pip install --user` on the login node would
  otherwise shadow the container's site-packages.
- `--cpus-per-task=8` is where the wall time actually goes. `get_struc_seq`
  invokes foldseek with `--threads 1`, so one structure is single-threaded
  and the parallelism has to come from processing several structures at once
  (each with a distinct `process_id`). One GPU keeps up with many foldseek
  workers.
- The `cd` matters: `get_struc_seq` writes its temp `.tsv` to the *current
  working directory* and deletes it afterwards, so the job must start in a
  writable directory.

## Minimal inference (Python API — no CLI)

Structure in, embedding out. `$FOLDSEEK_BIN` is set in the image, and
`get_struc_seq` asserts on the path, so pass it rather than the bare name:

```python
import os
import torch
from transformers import EsmForMaskedLM, EsmTokenizer
from saprot_utils.foldseek_util import get_struc_seq

CHECKPOINT = "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI"

# 1. Structure (PDB or mmCIF) -> combined AA+3Di sequence, per chain.
#    Returns {chain: (aa_seq, struc_seq, combined_seq)}.
seqs = get_struc_seq(
    os.environ["FOLDSEEK_BIN"],
    "AF-P00520-F1-model_v4.pdb",
    chains=["A"],
    plddt_mask=True,          # see below — matters for predicted structures
)
aa_seq, struc_seq, combined_seq = seqs["A"]

# 2. Combined sequence -> per-residue hidden states.
tokenizer = EsmTokenizer.from_pretrained(CHECKPOINT)
model = EsmForMaskedLM.from_pretrained(CHECKPOINT, device_map="auto").eval()

inputs = tokenizer([combined_seq], return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    out = model(**inputs, output_hidden_states=True)

residue_emb = out.hidden_states[-1][0, 1:-1]   # drop <cls> and <eos>
protein_emb = residue_emb.mean(0)
```

No tokenizer preprocessing is needed: SaProt's `vocab.txt` lists all 446
two-character tokens, and `EsmTokenizer` splits the string against them, so
`"MdApGq"` becomes three tokens.

**Sequence-only input** (1.3B checkpoints only) masks the structural half of
every token with `#`:

```python
combined_seq = "".join(aa + "#" for aa in aa_seq)
```

**pLDDT masking.** `plddt_mask` decides whether residues the structure
predictor was unsure about keep their 3Di state or get masked to `#`. It
matters for anything AlphaFold-like, and upstream stresses it repeatedly:
predicted low-confidence regions carry structural states that are noise, and
feeding them in degrades accuracy. Defaults and behaviour in the vendored
helper:

- `plddt_mask="auto"` (the default) reads the file and turns masking on only
  if the text contains "alphafold". That catches AFDB downloads and misses
  everything else — locally-run AlphaFold output, ESMFold or Boltz
  predictions, renamed files.
- **Pass `plddt_mask=True` explicitly for any predicted structure**, which
  masks every residue whose mean B-factor is below `plddt_threshold=70.`
  (B-factor is where AF2 stores pLDDT).
- Pass `plddt_mask=False` for experimental structures. On a crystal
  structure the B-factor column is a real B-factor, not a confidence, and
  masking on it is meaningless.

`saprot_utils.foldseek_util` also has `transform_pdb_dir`, which runs
foldseek once over a whole directory and writes a FASTA. It is faster than
looping — foldseek uses all cores there — but it does **no pLDDT masking**,
so it suits experimental structures only.

## Build notes / caveats

- **Upstream's environment is deliberately not used, and this is the main
  deviation from the recipe SaProt publishes.** `requirements.txt` pins
  `torch==1.13.1`, which predates the Hopper architecture and will not run
  on an H100. Rather than resurrect a 2022 stack, this image treats the
  checkpoints as what they are: `config.json` declares `model_type: "esm"`
  and `architectures: ["EsmForMaskedLM"]`, and every field it sets
  (`position_embedding_type: "rotary"`, `token_dropout`, `vocab_size: 446`,
  `emb_layer_norm_before`) is still supported by `transformers` 4.57.6. So
  the model loads with stock transformers on modern torch, and none of
  SaProt's training code is installed. Consequence: the repo's YAML-driven
  fine-tuning and evaluation scripts are **not** available here. This image
  covers embedding and scoring. Fine-tuning against upstream's Lightning
  harness would need a different image, and would have to solve the torch
  pin first.

- **Only ~200 lines of upstream Python are installed.** `foldseek_util.py`
  is fetched from `raw.githubusercontent.com` at commit `e91e4858`, which is
  content-addressed by git, and its sha256
  (`5905f0c2…d4b45449`, taken from the GitHub contents API at that commit) is
  checked at build time. It imports only numpy and Bio.PDB — nothing from
  SaProt — so it stands alone.

- **The import path differs from upstream's examples by one line.** Upstream
  writes `from utils.foldseek_util import get_struc_seq`; here the package is
  `saprot_utils`. Installing a top-level `utils` into site-packages would
  silently answer any `import utils` in a user's own code, which is not worth
  the copy-paste convenience.

- **foldseek is GPL-3.0 while SaProt is MIT.** The image label records
  `MIT AND GPL-3.0`. The binary is used as an unmodified upstream release
  invoked as a subprocess, which is the arrangement foldseek itself
  documents, but the license mix is worth knowing before this image is
  redistributed outside the lab.

- **No published checksum for the foldseek tarball.** The GitHub release for
  `10-941cd33` carries no asset digest and the project ships no checksum
  file, so the pin here is the release tag and nothing stronger. Release
  assets can in principle be re-uploaded under an unchanged tag. If that
  matters for a given build, download the tarball once, record its sha256,
  and add a `sha256sum -c` to the foldseek layer.

- **`foldseek-linux-avx2` requires AVX2.** The other Linux assets in that
  release are `arm64` and a `gpu` build; this image takes the static AVX2
  x86-64 one. Every current Compute2 node supports AVX2, but a build for an
  ARM host would need the other asset.

- **The 3Di output has not been validated against upstream's example.**
  SaProt ships `example/8ac8.cif` for exactly this check, and the build-time
  smoke test cannot run it — the check needs the file and a few seconds of
  foldseek, not just a version string. Run it once on Compute2 before
  trusting embeddings, and compare against the 3Di string upstream publishes.

- **Smoke test is offline and CPU-only.** It runs the foldseek binary and
  asserts the reported version matches the pin; tokenizes a probe string
  through `EsmTokenizer` to confirm the two-character-token trie still
  behaves; and builds a toy `EsmForMaskedLM` with SaProt's architecture
  settings and runs a forward pass. It does not download weights, so a real
  end-to-end load still has to happen on Compute2.

### Why transformers is held at 4.x

The first CI build of this image pinned `transformers==5.15.0` and failed to
resolve: the NGC base ships a global pip constraint file pinning
`regex==2024.11.6`, and transformers 5.x requires `regex>=2025.10.22`. The
pin moved to the last 4.x release (4.57.6), which asks only for
`regex!=2019.12.17` and `numpy>=1.17`.

That is also the better choice on its own merits. These are 2023-era
ESM-architecture checkpoints; pinning them to the newest major version of the
library that loads them adds risk without adding capability.
