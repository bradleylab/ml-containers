# DBloops boulder-clustering for boulder mapping (Jacobson et al. 2025).
#
# Stack: MATLAB Runtime R2024b + DBloops/Terpunkto/G3point compiled to
# a standalone binary via MATLAB Compiler (`mcc`).
#
# Upstream: https://github.com/haydenjgeo/DBloops (v1.0.0, c3acb15)
# Paper:    Jacobson, H. et al. (2025) — DBloops: density-based 3D point
#           cloud clustering for boulder grain-size distributions.
#
# What runs: per-patch DBloops invocation. Reads an XYZ point cloud
# (3-column text), produces per-point cluster labels and per-cluster
# b-axis sizes (.mat + .csv). The compiled binary bundles the patched
# upstream MATLAB source from
# bradleylab/rock_glaciers/scripts/matlab/.
#
# What this DOES NOT include: Jacobson's upstream Terpunkto random-forest
# clast/matrix classifier. Without that classifier, DBloops is a generic
# spatial cluster finder — at typical TLS densities (11–100 pts/m²) it
# produces detections that do not correspond to real boulders for rock
# glacier surfaces. See bradleylab/rock_glaciers/EXPERIMENTS.md v3a.
# This container is the right detector for high-density gravel-bed cases
# matching Jacobson's training data; for rock-glacier boulder mapping,
# use a different model (Track A / Mask R-CNN).
#
# Compile recipe: dbloops/compile.sh (run on a machine with MATLAB
# Compiler R2024b licensed; we use Compute2).

FROM mathworks/matlab-runtime-deps:r2024b

ARG MCR_VERSION=R2024b
ARG MCR_UPDATE=9
ARG MCR_INSTALL_DIR=/opt/mcr

ENV DEBIAN_FRONTEND=noninteractive

# System deps:
#   curl/unzip — fetch the MCR installer
#   python3 + numpy — synthesize the smoke-test XYZ
#   xvfb — MATLAB Runtime expects an X server context for some OpenGL
#         init even in headless deployments; Xvfb-as-fallback prevents
#         silent abort on minimal images
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl unzip xvfb \
        python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Install MATLAB Runtime R2024b silently.
# The installer extracts ~5 GB of zip into ~3 GB on disk under
# ${MCR_INSTALL_DIR}/${MCR_VERSION}.
RUN curl -fsSL \
        "https://ssd.mathworks.com/supportfiles/downloads/${MCR_VERSION}/Release/${MCR_UPDATE}/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_${MCR_VERSION}_Update_${MCR_UPDATE}_glnxa64.zip" \
        -o /tmp/mcr.zip \
    && mkdir -p /tmp/mcr \
    && unzip -q /tmp/mcr.zip -d /tmp/mcr \
    && /tmp/mcr/install -mode silent -agreeToLicense yes -destinationFolder "$MCR_INSTALL_DIR" \
    && rm -rf /tmp/mcr /tmp/mcr.zip

# Install the compiled DBloops binary + its mcc-generated launch wrapper.
WORKDIR /opt/dbloops
COPY bin/ ./bin/
COPY scripts/ ./scripts/
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh ./bin/run_run_dbloops_patch.sh ./bin/run_dbloops_patch

# Smoke test: synthesize a clusterable XYZ patch (30 boulders + matrix
# points -- uniform random caused all points to fall out as DBSCAN
# noise, so DBloops returned with no clusters and the wrapper's
# 6-output destructure tripped MATLAB:unassignedOutputs), run the
# binary end-to-end, verify the .mat lands. Catches missing toolbox
# runtime libs and entrypoint wiring at build time.
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
RUN mkdir -p /tmp/smoke \
    && python3 /opt/dbloops/scripts/make_smoke_xyz.py /tmp/smoke/in.xyz \
    && PATCH_XYZ=/tmp/smoke/in.xyz PATCH_OUT=/tmp/smoke/out NP_VAL=18 \
       /opt/dbloops/entrypoint.sh \
    && test -s /tmp/smoke/out.mat \
    && echo "SMOKE TEST OK" \
    && rm -rf /tmp/smoke
SHELL ["/bin/sh", "-c"]

LABEL org.opencontainers.image.title="dbloops" \
      org.opencontainers.image.description="DBloops density-based 3D point-cloud clustering (Jacobson et al. 2025) compiled with MATLAB Compiler R2024b; runs against MATLAB Runtime; no learned components in the bundled detector (RF clast classifier is upstream of this container)" \
      org.opencontainers.image.source="https://github.com/bradleylab/ml-containers" \
      org.opencontainers.image.documentation="https://github.com/bradleylab/ml-containers/blob/main/dbloops/README.md" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.vendor="bradleylab" \
      org.opencontainers.image.licenses="MIT" \
      bradleylab.model.upstream="https://github.com/haydenjgeo/DBloops" \
      bradleylab.model.upstream.commit="c3acb15b2828e3e422c9e1ed2f6171ab6b1e8f1d" \
      bradleylab.model.upstream.tag="v1.0.0" \
      bradleylab.model.paper="Jacobson et al. 2025 (DOI TBD)" \
      bradleylab.model.task="point-cloud clustering, grain-size distribution" \
      bradleylab.model.sensor="lidar:tls,mls" \
      bradleylab.model.status="experimental" \
      bradleylab.model.weights.source="none (classical algorithm; binary compiled from MATLAB source)"

WORKDIR /work
ENTRYPOINT ["/opt/dbloops/entrypoint.sh"]
