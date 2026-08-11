# AlphaFold 3 software runtime only. Model parameters and genetic databases
# are deliberately excluded and must be mounted at runtime.

FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ARG AF3_COMMIT=85c4d20505fd5cef05eac22b534d4e793971ae69

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update --quiet \
    && apt-get install --yes --quiet \
      ca-certificates \
      g++ \
      gcc \
      git \
      make \
      patch \
      python3.12 \
      python3.12-dev \
      wget \
      zlib1g-dev \
      zstd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/alphafold3_venv \
    PATH="/hmmer/bin:/alphafold3_venv/bin:${PATH}"

RUN uv venv "${UV_PROJECT_ENVIRONMENT}"

RUN mkdir /hmmer_build /hmmer \
    && wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz \
      --directory-prefix /hmmer_build \
    && cd /hmmer_build \
    && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  hmmer-3.4.tar.gz" \
      | sha256sum --check \
    && tar zxf hmmer-3.4.tar.gz \
    && rm hmmer-3.4.tar.gz

RUN wget --https-only \
      --output-document=/hmmer_build/jackhmmer_seq_limit.patch \
      "https://raw.githubusercontent.com/google-deepmind/alphafold3/${AF3_COMMIT}/docker/jackhmmer_seq_limit.patch" \
    && echo "df9e3ae35ad1659921d96ebfca67a9616a7a467ddde2be18a56f9bd3edb38c41  /hmmer_build/jackhmmer_seq_limit.patch" \
      | sha256sum --check \
    && cd /hmmer_build \
    && patch -p0 < jackhmmer_seq_limit.patch

RUN cd /hmmer_build/hmmer-3.4 \
    && ./configure --prefix /hmmer \
    && make -j"$(nproc)" \
    && make install \
    && cd easel \
    && make install \
    && mkdir -p /usr/share/licenses/hmmer \
    && cp /hmmer_build/hmmer-3.4/LICENSE /usr/share/licenses/hmmer/HMMER-LICENSE \
    && cp /hmmer_build/hmmer-3.4/easel/LICENSE /usr/share/licenses/hmmer/EASEL-LICENSE \
    && cp /hmmer_build/hmmer-3.4/libdivsufsort/COPYING \
      /usr/share/licenses/hmmer/LIBDIVSUFSORT-COPYING \
    && rm -rf /hmmer_build

RUN git clone --filter=blob:none https://github.com/google-deepmind/alphafold3.git /app/alphafold \
    && cd /app/alphafold \
    && git checkout --detach "${AF3_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${AF3_COMMIT}"

WORKDIR /app/alphafold

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --all-groups --no-editable \
    && test "$(uv run python -c 'import importlib.metadata; print(importlib.metadata.version(\"alphafold3\"))')" = "3.0.4" \
    && rm -rf .git

RUN uv run build_data

COPY scripts/smoke_test.py /usr/local/bin/alphafold3_smoke_test.py
RUN uv run python /usr/local/bin/alphafold3_smoke_test.py

ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_CLIENT_MEM_FRACTION=0.95 \
    PYTHONNOUSERSITE=1

LABEL org.opencontainers.image.title="AlphaFold 3" \
      org.opencontainers.image.description="AlphaFold 3 v3.0.4 software runtime without model parameters or databases" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/alphafold3/README.md" \
      org.opencontainers.image.version="3.0.4" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause AND BSD-2-Clause" \
      bradleylab.model.upstream="https://github.com/google-deepmind/alphafold3" \
      bradleylab.model.commit="85c4d20505fd5cef05eac22b534d4e793971ae69" \
      bradleylab.model.task="biomolecular structure prediction" \
      bradleylab.model.status="experimental" \
      bradleylab.model.private_assets="model parameters, reference databases"

CMD ["uv", "run", "python3", "run_alphafold.py"]
