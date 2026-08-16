# esm — ESMC protein language model (Chan Zuckerberg Biohub)
#
# Protein LM for sequence embeddings, per-residue logits, and masked-token
# scoring. ESMC-600M is the lab's default workhorse for batch embedding;
# ESMC-6B (25.4 GB F32) fits a single H100 80 GB.
#
# Upstream:  https://github.com/Biohub/esm  (code MIT — EvolutionaryScale
#            became Chan Zuckerberg Biohub; the old evolutionaryscale/esm
#            URL 301-redirects here)
# Weights:   https://huggingface.co/biohub  (ESMC-300M / ESMC-600M / ESMC-6B
#            / esm3-sm-open-v1 — all MIT and ungated)
# Stack:     NGC PyTorch 25.04 (torch 2.7.0a0, CUDA 12.9, Python 3.12) +
#            the esm git package + Biohub's transformers fork.
#
# Base-image constraint: esm declares `requires-python = ">=3.12,<3.13"`,
# so pip refuses to install on anything but Python 3.12. NGC PyTorch 25.04
# ships Python 3.12 (same base as the evo2 image), which is why it is used
# here rather than a `pytorch/pytorch:*-runtime` tag.

FROM nvcr.io/nvidia/pytorch:25.04-py3

# Reproducibility pin. esm's pyproject declares two dependencies that float
# at `@main` — esm itself and Biohub's transformers fork — so an unpinned
# build is not reproducible. Both are resolved to commit SHAs here, recorded
# in the labels below, and asserted by the smoke test. Override locally with
# `docker build --build-arg ESM_REF=<sha> ...`; to change the pin in CI,
# edit these defaults and commit, so the pin lives in git history.
ARG ESM_REF=26b0bc2b771e3e419ea74f445a5f35cc094a1509
ARG ESM_TRANSFORMERS_REF=ef32577f55da19a4989cd7b22e004dc43a4998cb
ENV ESM_REF=${ESM_REF} \
    ESM_TRANSFORMERS_REF=${ESM_TRANSFORMERS_REF}

# Install order matters. The first install resolves esm's whole dependency
# tree but drags in transformers at whatever `@main` points to today; the
# second pins the fork to ESM_TRANSFORMERS_REF. `--no-deps` on the pin keeps
# the already-resolved tree intact — the fork's own requirements were
# satisfied by the first pass. esm's third git dependency (nrontsis/DockQ)
# is already SHA-pinned upstream, so it needs no treatment here.
RUN pip install --no-cache-dir "esm @ git+https://github.com/Biohub/esm.git@${ESM_REF}" \
    && pip install --no-cache-dir --force-reinstall --no-deps \
        "transformers @ git+https://github.com/Biohub/transformers.git@${ESM_TRANSFORMERS_REF}"

# Weights are NOT baked — fetched from the biohub HF org on first use
# (e.g. AutoModelForMaskedLM.from_pretrained("biohub/ESMC-600M")). Point
# HF_HOME at a bind-mounted scratch dir at runtime so checkpoints persist
# across jobs; ESMC-6B is 25.41 GB across 6 shards.
ENV HF_HOME=/root/.cache/huggingface

# Build-time smoke test. Versions come from package metadata per the
# ml-containers rule, never `pkg.__version__`. Like evo2, this does NOT
# `import esm`: the package reaches for GPU-only extensions (flash_attn)
# that the CPU build runner lacks, so a real load test has to run on an
# H100. What it does verify offline is the thing that actually rots — that
# both git dependencies are installed at the SHAs this image claims, read
# back from pip's own direct_url.json records.
RUN python -c "import importlib.metadata as im, json, os, torch; \
sha = lambda n: json.loads(im.distribution(n).read_text('direct_url.json'))['vcs_info']['commit_id']; \
esm_sha, tf_sha = sha('esm'), sha('transformers'); \
print('esm', im.version('esm'), esm_sha, '| transformers', im.version('transformers'), tf_sha, '| torch', torch.__version__); \
assert esm_sha == os.environ['ESM_REF'], (esm_sha, os.environ['ESM_REF']); \
assert tf_sha == os.environ['ESM_TRANSFORMERS_REF'], (tf_sha, os.environ['ESM_TRANSFORMERS_REF'])"

WORKDIR /workspace

LABEL org.opencontainers.image.title="esm" \
      org.opencontainers.image.description="ESMC (Chan Zuckerberg Biohub) protein language model — sequence embeddings and per-residue logits; NGC pytorch 25.04 + esm git package pinned to a commit SHA; H100" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/esm/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/Biohub/esm" \
      bradleylab.model.paper="https://www.biorxiv.org/content/10.64898/2026.06.03.729735" \
      bradleylab.model.task="protein language model, sequence embedding + per-residue logits" \
      bradleylab.model.sensor="protein-sequence" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="huggingface:biohub (runtime fetch; MIT, ungated)" \
      bradleylab.build.esm_ref="${ESM_REF}" \
      bradleylab.build.transformers_ref="${ESM_TRANSFORMERS_REF}"
