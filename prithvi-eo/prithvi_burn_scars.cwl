#!/usr/bin/env cwl-runner
# prithvi_burn_scars, described as a CWL v1.2 CommandLineTool for the image
# ghcr.io/bradleylab/prithvi-eo:v3, whose ENTRYPOINT takes --input-dir,
# --output-dir and --params-json (the geospatial executor's entrypoint contract
# v1). The staging layout and params.json are built from the inputs below, so
# an engine that honors the image ENTRYPOINT runs it as the executor does.
# Fields under gx: are the executor's extension for what CWL cannot say
# (fossettlab/geospatial-executor, docs/contracts/cwl-extensions-v1.md).
cwlVersion: v1.2
class: CommandLineTool
id: prithvi_burn_scars
label: Burn-scar map of one HLS scene with Prithvi-EO 2.0
doc: |
  Maps burn scars in one 6-band HLS scene (int16 reflectance x 10000, fill
  -9999; blue, green, red, narrow NIR, SWIR 1, SWIR 2) with the
  Prithvi-EO-2.0-300M-BurnScars fine-tune, run as its own inference script
  runs it. Writes a GeoTIFF of not burned (0), burn scar (1) and nodata (255),
  a PNG preview of it at most 1024 pixels on a side, and a report of the
  burned area. Clouds are not masked.

$namespaces:
  s: https://schema.org/
  gx: https://github.com/fossettlab/geospatial-executor/blob/main/docs/contracts/cwl-extensions-v1.md#

s:codeRepository: https://github.com/bradleylab/ml-containers
s:license: Apache-2.0
s:citation: https://arxiv.org/abs/2412.02732
s:version: "3"
gx:model: prithvi-eo
gx:capability: map_burn_scars
gx:tags: [segmentation]
gx:output_kind: raster
gx:contract: 1

requirements:
  DockerRequirement:
    dockerPull: ghcr.io/bradleylab/prithvi-eo:v3
  InitialWorkDirRequirement:
    listing:
      # writable: the file is copied into place rather than bind-mounted
      # inside the output directory, a nested mount Docker Desktop refuses.
      - entryname: input/primary/$(inputs.scene.basename)
        entry: $(inputs.scene)
        writable: true
      - entryname: params.json
        entry: "{}"
  ResourceRequirement:
    coresMin: 8
    ramMin: 16384
  ToolTimeLimit:
    timelimit: 3600

inputs:
  scene:
    type: File
    label: 6-band HLS scene, int16 reflectance x 10000, in a projected CRS in meters
    gx:kind: raster
    gx:primary: true
    gx:formats: [".tif", ".tiff"]

arguments:
  - --input-dir
  - $(runtime.outdir)/input
  - --output-dir
  - $(runtime.outdir)/output
  - --params-json
  - $(runtime.outdir)/params.json

outputs:
  burn_scars:
    type: File
    outputBinding: {glob: output/burn_scars.tif}
    gx:role: burn_scars
  preview:
    type: File
    outputBinding: {glob: output/burn_scars_preview.png}
    gx:role: preview
  report:
    type: File
    outputBinding: {glob: output/burn_scars_report.json}
    gx:role: report
  manifest:
    type: File
    outputBinding: {glob: output/run.json}
    gx:manifest: true
