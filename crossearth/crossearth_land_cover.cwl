#!/usr/bin/env cwl-runner
# crossearth_land_cover, described as a CWL v1.2 CommandLineTool for the image
# ghcr.io/bradleylab/crossearth:v3, whose ENTRYPOINT takes --input-dir,
# --output-dir and --params-json (the geospatial executor's entrypoint contract
# v1). The staging layout and params.json are built from the typed inputs below,
# so an engine that honors the image ENTRYPOINT runs it as the executor does.
# Fields under gx: are the executor's extension for what CWL cannot say
# (fossettlab/geospatial-executor, docs/contracts/cwl-extensions-v1.md).
cwlVersion: v1.2
class: CommandLineTool
id: crossearth_land_cover
label: Land-cover map of an RGB orthophoto with CrossEarth
doc: |
  Classifies every pixel of an 8-bit RGB orthophoto, in a projected coordinate
  system in meters, into six land-cover classes: impervious surface, building,
  low vegetation, tree, car and clutter. The model is CrossEarth's Potsdam RGB
  source model, trained on the ISPRS Potsdam RGB true orthophotos, whose cells
  are 5 cm. Writes the class map as a GeoTIFF with a color table and a report
  with each class's area.

$namespaces:
  s: https://schema.org/
  cwltool: http://commonwl.org/cwltool#
  gx: https://github.com/fossettlab/geospatial-executor/blob/main/docs/contracts/cwl-extensions-v1.md#

s:codeRepository: https://github.com/bradleylab/ml-containers
s:license: MIT
s:citation: https://arxiv.org/abs/2410.22629
s:version: "3"
gx:model: crossearth
gx:capability: map_land_cover
gx:tags: [segmentation]
gx:output_kind: raster
gx:contract: 1

requirements:
  DockerRequirement:
    dockerPull: ghcr.io/bradleylab/crossearth:v3
  InlineJavascriptRequirement: {}
  InitialWorkDirRequirement:
    listing:
      # writable: the file is copied into place rather than bind-mounted
      # inside the output directory, a nested mount Docker Desktop refuses.
      - entryname: input/primary/$(inputs.orthophoto.basename)
        entry: $(inputs.orthophoto)
        writable: true
      # Contract v1 hands the image every value as a string.
      - entryname: params.json
        entry: |-
          ${
            return JSON.stringify({resolution: String(inputs.resolution)});
          }
  ResourceRequirement:
    coresMin: 4
    ramMin: 32768
  ToolTimeLimit:
    timelimit: 3600

hints:
  # Run on H100s only so far; the image's torch build is CUDA 12.1.
  cwltool:CUDARequirement:
    cudaVersionMin: "12.1"
    cudaComputeCapability: "9.0"
    cudaDeviceCountMin: 1
    cudaDeviceCountMax: 1

inputs:
  orthophoto:
    type: File
    label: RGB or RGBA orthophoto, an 8-bit GeoTIFF in a projected CRS in meters
    gx:kind: raster
    gx:primary: true
    gx:formats: [".tif", ".tiff"]
  resolution:
    type: float
    default: 0
    label: Cell size in meters to resample onto first; 0 keeps the orthophoto's own cells
    gx:min: 0
    gx:max: 1

arguments:
  - --input-dir
  - $(runtime.outdir)/input
  - --output-dir
  - $(runtime.outdir)/output
  - --params-json
  - $(runtime.outdir)/params.json

outputs:
  land_cover:
    type: File
    outputBinding: {glob: output/land_cover.tif}
    gx:role: land_cover
  report:
    type: File
    outputBinding: {glob: output/land_cover_report.json}
    gx:role: report
  manifest:
    type: File
    outputBinding: {glob: output/run.json}
    gx:manifest: true
