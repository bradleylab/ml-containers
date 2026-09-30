#!/usr/bin/env cwl-runner
# clay_embed, described as a CWL v1.2 CommandLineTool for the image
# ghcr.io/bradleylab/clay:v3, whose ENTRYPOINT takes --input-dir, --output-dir
# and --params-json (the geospatial executor's entrypoint contract v1). The
# staging layout and params.json are built from the typed inputs below, so an
# engine that honors the image ENTRYPOINT runs it as the executor does.
# Fields under gx: are the executor's extension for what CWL cannot say
# (fossettlab/geospatial-executor, docs/contracts/cwl-extensions-v1.md).
cwlVersion: v1.2
class: CommandLineTool
id: clay_embed
label: Clay embeddings of a PlanetScope scene, one per chip
doc: |
  Cuts a 4-band PlanetScope surface-reflectance scene (blue, green, red, NIR,
  reflectance scaled by 10,000) into square chips and embeds each with the Clay
  v1.5 encoder. Writes a GeoTIFF with one cell per chip and one band per
  embedding dimension (1,024), and a report of the chips embedded and skipped.
  A chip holding any masked pixel is left empty.

$namespaces:
  s: https://schema.org/
  gx: https://github.com/fossettlab/geospatial-executor/blob/main/docs/contracts/cwl-extensions-v1.md#

s:codeRepository: https://github.com/bradleylab/ml-containers
s:license: Apache-2.0
s:citation: https://clay-foundation.github.io/model
s:version: "3"
gx:model: clay
gx:capability: embed_scene
gx:tags: [embedding]
gx:output_kind: raster
gx:contract: 1

requirements:
  DockerRequirement:
    dockerPull: ghcr.io/bradleylab/clay:v3
  InlineJavascriptRequirement: {}
  InitialWorkDirRequirement:
    listing:
      # writable: the file is copied into place rather than bind-mounted
      # inside the output directory, a nested mount Docker Desktop refuses.
      - entryname: input/primary/$(inputs.scene.basename)
        entry: $(inputs.scene)
        writable: true
      # Contract v1 hands the image every value as a string.
      - entryname: params.json
        entry: |-
          ${
            return JSON.stringify({chip_size: inputs.chip_size});
          }
  ResourceRequirement:
    coresMin: 8
    ramMin: 16384
  ToolTimeLimit:
    timelimit: 3600

inputs:
  scene:
    type: File
    label: 4-band PlanetScope surface-reflectance GeoTIFF in a projected CRS in meters
    gx:kind: raster
    gx:primary: true
    gx:formats: [".tif", ".tiff"]
  chip_size:
    type:
      type: enum
      symbols: ["64", "128", "256"]
    default: "256"
    label: Chip side in pixels; Clay v1.5 was trained on 256-pixel chips

arguments:
  - --input-dir
  - $(runtime.outdir)/input
  - --output-dir
  - $(runtime.outdir)/output
  - --params-json
  - $(runtime.outdir)/params.json

outputs:
  embeddings:
    type: File
    outputBinding: {glob: output/embeddings.tif}
    gx:role: embeddings
  report:
    type: File
    outputBinding: {glob: output/embeddings_report.json}
    gx:role: report
  manifest:
    type: File
    outputBinding: {glob: output/run.json}
    gx:manifest: true
