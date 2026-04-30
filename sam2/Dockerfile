# SAM 2 / SAM 2.1 inference container for bradleylab.
#
# Goal: a portable model image you can `docker run` (or `apptainer run`) on
# any GPU host — Compute2, EC2, local — to segment any RGB image.
# Weights are downloaded on first run from the Hugging Face Hub and cached
# under $HF_HOME (override-bindable).
#
# Pull:
#   docker pull ghcr.io/bradleylab/sam2:latest
#   apptainer pull docker://ghcr.io/bradleylab/sam2:latest
#
# Run (Docker on a CUDA host):
#   docker run --gpus all --rm -v $PWD:/work \
#     -v $PWD/.hf-cache:/home/runner/.cache/huggingface \
#     ghcr.io/bradleylab/sam2:latest \
#     --image /work/input.jpg --output /work/masks.json --mode amg
#
# Run (Apptainer on Compute2 / TGI RAILS):
#   apptainer run --nv --bind $PWD:/work,$PWD/.hf-cache:/home/runner/.cache/huggingface \
#     sam2_latest.sif \
#     --image /work/input.jpg --output /work/masks.json --mode amg

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/runner/.cache/huggingface

# System deps. Keep this minimal — SAM 2 inference does not need build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. Apptainer rootless on Compute2 runs as the invoking
# user, but Docker on EC2 / locally benefits from a stable non-root id.
ARG USERNAME=runner
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

WORKDIR /app

# Python deps. SAM 2 isn't on PyPI; install from upstream.
# SAM2_BUILD_CUDA=0 skips an optional CUDA C++ extension that we don't need
# for inference and that requires nvcc (only present in -devel base images).
# Pure-Python fallback is automatic and only marginally slower for our paths.
ENV SAM2_BUILD_CUDA=0
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir \
        "git+https://github.com/facebookresearch/sam2.git@main"

COPY sam2_infer.py /app/sam2_infer.py

# Cache dir for HF weights — created with the right ownership so first-run
# weight download works under the runner user.
RUN mkdir -p ${HF_HOME} && chown -R ${USERNAME}:${USERNAME} /app ${HF_HOME}

USER ${USERNAME}

ENTRYPOINT ["python", "/app/sam2_infer.py"]
CMD ["--help"]
