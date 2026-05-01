# 3dfin

Deterministic TLS stem detection and DBH estimation via 3DFin
(Laino et al. 2024).

Classical (non-deep-learning) counterpart to `treelearn` and
`segment-any-tree-h100` on TLS data. CPU-only.

## Image

`ghcr.io/bradleylab/3dfin:v1`

## Usage

```
3DFin cli --help                           # discover current CLI flags
scripts/run_3dfin.sh input.laz /out/dir    # wrapper invocation
```

## Reference

Laino, D., Cabo, C., Prendes, C., Janvier, R., Ordoñez, C., Nicolas-Cuevas,
J. A., Lamelas, M. T., Reque, J., Castedo-Dorado, F. (2024). 3DFin: a
software for automated 3D forest inventories from terrestrial point
clouds. *Forestry*, 97, 479–496.
