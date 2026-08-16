# saprot — SaProt structure-aware protein language model (Westlake)
#
# Protein LM over a structure-aware vocabulary: every residue is a single
# token combining its amino acid with its foldseek 3Di structural state
# ("Aq", "Md", ...). Used for embeddings, mutational-effect scoring, and
# fine-tuning. The 1.3B checkpoint also accepts sequence-only input, where
# the structural half of each token is masked ("A#").
#
# Upstream:  https://github.com/westlake-repl/SaProt   (code MIT)
# Weights:   https://huggingface.co/westlake-repl      (MIT, ungated)
# foldseek:  https://github.com/steineggerlab/foldseek (GPL-3.0)
# Stack:     NGC PyTorch 25.04 (torch 2.7.0a0, CUDA 12.9, Python 3.12) +
#            stock transformers + an official foldseek release binary.
#
# Two tools in one image, deliberately. SaProt cannot construct its own
# input without foldseek: `get_struc_seq` shells out to
# `foldseek structureto3didescriptor` to turn a PDB/mmCIF file into the 3Di
# string that half of every SaProt token is made of. Splitting them would
# put a container boundary in the middle of one function call. This is the
# ml-containers exception clause; the reasoning is in README.md.
#
# Upstream's pinned environment is NOT used. SaProt's requirements.txt pins
# torch==1.13.1, which predates Hopper and will not run on an H100. The
# published checkpoints are plain ESM-architecture HF repos
# (config.json: model_type "esm", architectures ["EsmForMaskedLM"]) and load
# with stock transformers on modern torch. The only upstream Python needed
# at inference time is utils/foldseek_util.py, installed below at a pinned
# commit. This deviation is deliberate — see README.md "Build notes".

FROM nvcr.io/nvidia/pytorch:25.04-py3

# Reproducibility pins. foldseek is a released tarball rather than a pip
# package, and foldseek_util.py is fetched from a commit-addressed raw URL,
# so both need explicit version pins to make two builds a month apart
# identical. Change a pin by editing the ARG default and committing, so the
# pin lives in git history; the CI workflow passes no build-args.
ARG FOLDSEEK_VERSION=10-941cd33
ARG SAPROT_REF=e91e4858b55944523f1f8d385f7b96a0d3d34c1d
ARG FOLDSEEK_UTIL_SHA256=5905f0c29db41a0c89677412635c8e3a1886906a3ea7e963124d7218d4b45449
ENV FOLDSEEK_VERSION=${FOLDSEEK_VERSION} \
    SAPROT_REF=${SAPROT_REF} \
    FOLDSEEK_BIN=/usr/local/bin/foldseek

# curl + ca-certificates for the two pinned downloads below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# foldseek, from the project's own release rather than SaProt's Google Drive
# link (which is unversioned and unchecksummed). `linux-avx2` is the static
# x86-64 build; it needs an AVX2-capable CPU, which every Compute2 node has.
# The upstream release publishes no checksum, so the pin is the release tag.
RUN curl -fsSL \
        "https://github.com/steineggerlab/foldseek/releases/download/${FOLDSEEK_VERSION}/foldseek-linux-avx2.tar.gz" \
        -o /tmp/foldseek.tar.gz \
    && tar -xzf /tmp/foldseek.tar.gz -C /opt \
    && ln -s /opt/foldseek/bin/foldseek /usr/local/bin/foldseek \
    && rm /tmp/foldseek.tar.gz

# Inference-time Python. transformers ships the Esm model and tokenizer
# classes the checkpoints declare; biopython backs foldseek_util's pLDDT
# extraction (Bio.PDB); accelerate backs `device_map=`.
#
# transformers is held at the last 4.x rather than 5.x for two reasons. The
# hard one: this base image carries a global pip constraint file pinning
# regex==2024.11.6, and transformers 5.x requires regex>=2025.10.22, so 5.x
# cannot resolve here at all. The soft one: these are 2023-era
# ESM-architecture checkpoints, and pinning them to the newest major version
# of the library that loads them is a risk with no upside. 4.57.x asks only
# for regex!=2019.12.17 and numpy>=1.17, both satisfied by the base.
RUN pip install --no-cache-dir \
        "transformers==4.57.6" \
        "biopython==1.88" \
        "accelerate==1.14.0"

# SaProt's structure->3Di helper, installed as `saprot_utils` rather than
# upstream's `utils` so it does not squat a generic name in site-packages —
# the only difference from upstream's examples is the import line. The raw
# URL is commit-addressed, and the sha256 was taken from the GitHub contents
# API at that commit, so a silently changed file fails the build.
# The install directory is read off the just-installed transformers package
# rather than site.getsitepackages(), whose first entry is not reliably the
# one pip writes to on Debian-derived images.
RUN SITE=$(python -c "import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent.parent)") \
    && mkdir -p "${SITE}/saprot_utils" \
    && : > "${SITE}/saprot_utils/__init__.py" \
    && curl -fsSL \
        "https://raw.githubusercontent.com/westlake-repl/SaProt/${SAPROT_REF}/utils/foldseek_util.py" \
        -o "${SITE}/saprot_utils/foldseek_util.py" \
    && echo "${FOLDSEEK_UTIL_SHA256}  ${SITE}/saprot_utils/foldseek_util.py" | sha256sum -c -

# Weights are NOT baked — fetched from the westlake-repl HF org on first use
# (SaProt_1.3B_AFDB_OMG_NCBI is 5.21 GB). Point HF_HOME at a bind-mounted
# scratch dir at runtime so checkpoints persist across jobs.
ENV HF_HOME=/root/.cache/huggingface

# Build-time smoke test, offline and CPU-only. Versions come from package
# metadata per the ml-containers rule. Beyond the version printout it checks
# the three things that would otherwise fail at first inference:
#   1. the foldseek binary runs and reports the pinned build. `foldseek
#      version` prints the bare commit hash (941cd33ff077...), not the
#      release tag (10-941cd33), so the check compares against the hash
#      half of the tag — matching the whole tag never succeeds;
#   2. EsmTokenizer's trie splits SaProt's two-character tokens (the probe
#      vocab holds real tokens from the 1.3B vocab.txt — "Ap"/"Md" carry a
#      3Di state, "A#" is the sequence-only form), so "MdApA#" must come
#      back as three tokens between <cls> and <eos>, none of them <unk>;
#   3. EsmForMaskedLM builds and runs against SaProt's architecture settings
#      (rotary positions, token dropout, 446-token vocab) at toy width.
RUN printf '%s\n' '<cls>' '<pad>' '<eos>' '<unk>' '<mask>' 'Ap' 'Md' 'A#' \
        > /tmp/saprot_vocab_probe.txt \
    && python -c "import importlib.metadata as im, os, subprocess, torch; \
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer; \
from saprot_utils.foldseek_util import get_struc_seq; \
fs = subprocess.run([os.environ['FOLDSEEK_BIN'], 'version'], capture_output=True, text=True); \
ver = (fs.stdout + fs.stderr).strip(); \
assert os.environ['FOLDSEEK_VERSION'].split('-')[-1] in ver, ver; \
tok = EsmTokenizer(vocab_file='/tmp/saprot_vocab_probe.txt'); \
ids = tok('MdApA#')['input_ids']; \
assert len(ids) == 5 and tok.unk_token_id not in ids, ids; \
cfg = EsmConfig(vocab_size=446, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, max_position_embeddings=1538, position_embedding_type='rotary', token_dropout=True, emb_layer_norm_before=False, pad_token_id=1, mask_token_id=4); \
x = torch.tensor([[0, 5, 6, 7, 2]]); \
out = EsmForMaskedLM(cfg)(input_ids=x, attention_mask=torch.ones_like(x)); \
assert tuple(out.logits.shape) == (1, 5, 446), out.logits.shape; \
print('transformers', im.version('transformers'), '| biopython', im.version('biopython'), '| accelerate', im.version('accelerate'), '| torch', torch.__version__, '| foldseek', ver, '| saprot_ref', os.environ['SAPROT_REF'][:8])" \
    && rm /tmp/saprot_vocab_probe.txt

WORKDIR /workspace

LABEL org.opencontainers.image.title="saprot" \
      org.opencontainers.image.description="SaProt (Westlake) structure-aware protein language model — AA+3Di embeddings and mutational scoring; NGC pytorch 25.04 + stock transformers + pinned foldseek release; H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/saprot/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT AND GPL-3.0" \
      bradleylab.model.upstream="https://github.com/westlake-repl/SaProt" \
      bradleylab.model.paper="https://doi.org/10.1101/2023.10.01.560349" \
      bradleylab.model.task="protein language model, structure-aware sequence embedding + mutational scoring" \
      bradleylab.model.sensor="protein-sequence+structure" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:westlake-repl (runtime fetch; MIT, ungated)" \
      bradleylab.build.foldseek_version="${FOLDSEEK_VERSION}" \
      bradleylab.build.saprot_ref="${SAPROT_REF}"
