"""Pure PyTorch reimplementation of region_grow from torch-points-kernels.

Replaces the unmaintained C++/CUDA torch-points-kernels dependency with
torch_cluster.radius() for ball queries and Numba-accelerated union-find
for connected components.

Origin: Ported from tyson-swetnam/SegmentAnyTree (2026-update branch),
        sat/clustering/region_grow.py

This runs on CPU, matching the original usage pattern where all call sites
pass .cpu() tensors. The Numba JIT-compiled union-find provides ~50-100x
speedup over a pure Python loop for the connected component computation.

Dependencies: torch, torch_cluster, numba, numpy
"""

from typing import List, Optional

import numpy as np
import numba
import torch
from torch_cluster import radius


@numba.jit(nopython=True)
def _union_find_batch(
    parent: np.ndarray,
    rank: np.ndarray,
    edge_target: np.ndarray,
    edge_source: np.ndarray,
) -> None:
    """Process all edges through union-find with path compression.

    Modifies parent and rank arrays in place. Compiled to native code
    by Numba for performance on large edge sets (millions of edges).
    """
    for i in range(edge_target.shape[0]):
        a, b = edge_target[i], edge_source[i]
        # Find root of a with path compression
        ra = a
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]
        # Find root of b with path compression
        rb = b
        while parent[rb] != rb:
            parent[rb] = parent[parent[rb]]
            rb = parent[rb]
        # Union by rank
        if ra != rb:
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1


@numba.jit(nopython=True)
def _flatten_roots(parent: np.ndarray) -> np.ndarray:
    """Flatten all parent pointers to their root representatives."""
    n = len(parent)
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = i
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        roots[i] = r
    return roots


def region_grow(
    positions: torch.Tensor,
    predicted_labels: torch.Tensor,
    batch: torch.Tensor,
    ignore_labels: Optional[torch.Tensor] = None,
    radius_value: float = 0.03,
    nsample: int = 200,
    min_cluster_size: int = 10,
    **kwargs,
) -> List[torch.Tensor]:
    """Region growing clustering on point clouds.

    Groups points into clusters by:
    1. Ball query to find neighboring points within a radius
    2. Only connecting points with matching semantic labels
    3. Excluding points with labels in ignore_labels
    4. Connected components via Numba-accelerated union-find

    Parameters
    ----------
    positions : Tensor [N, 3]
        Point positions (CPU).
    predicted_labels : Tensor [N]
        Predicted semantic label per point (CPU).
    batch : Tensor [N]
        Batch index per point (CPU).
    ignore_labels : Tensor, optional
        Labels to exclude from clustering (e.g., ground, low vegetation).
    radius_value : float
        Ball query search radius in meters. Controls how far apart
        two points can be and still be connected. For 0.2m voxel grid,
        typical values are 0.3m (1.5x) to 0.5m (2.5x).
    nsample : int
        Maximum neighbors per point in ball query.
    min_cluster_size : int
        Minimum number of points to form a valid cluster.

    Returns
    -------
    List[Tensor]
        List of 1D tensors, each containing point indices for one cluster.
        Indices reference the original positions array.
    """
    # Handle the 'radius' keyword used by existing call sites
    if "radius" in kwargs:
        radius_value = kwargs["radius"]

    n_points = positions.shape[0]
    if n_points == 0:
        return []

    # Ensure CPU and correct dtypes
    positions = positions.float().cpu()
    predicted_labels = predicted_labels.long().cpu()
    batch = batch.long().cpu()
    if batch.dim() > 1:
        batch = batch.squeeze(-1)

    # Build mask for valid (non-ignored) points
    if ignore_labels is not None:
        ignore_labels = ignore_labels.long().cpu()
        valid_mask = ~torch.isin(predicted_labels, ignore_labels)
    else:
        valid_mask = torch.ones(n_points, dtype=torch.bool)

    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() == 0:
        return []

    # Ball query using torch_cluster.radius()
    valid_pos = positions[valid_indices]
    valid_batch = batch[valid_indices]
    valid_labels = predicted_labels[valid_indices]

    # radius() returns (target_idx, source_idx) in valid_pos index space
    edge_target, edge_source = radius(
        valid_pos,
        valid_pos,
        r=radius_value,
        batch_x=valid_batch,
        batch_y=valid_batch,
        max_num_neighbors=nsample,
    )

    # Filter: only keep edges where both endpoints have the same label
    same_label = valid_labels[edge_target] == valid_labels[edge_source]
    not_self = edge_target != edge_source
    keep = same_label & not_self
    edge_target = edge_target[keep]
    edge_source = edge_source[keep]

    # Union-find connected components (Numba-accelerated)
    n_valid = valid_indices.shape[0]
    parent = np.arange(n_valid, dtype=np.int64)
    rank_arr = np.zeros(n_valid, dtype=np.int64)

    et = edge_target.numpy().astype(np.int64)
    es = edge_source.numpy().astype(np.int64)

    _union_find_batch(parent, rank_arr, et, es)
    roots = torch.from_numpy(_flatten_roots(parent))

    # Group points by root, mapping back to original indices
    clusters: list[torch.Tensor] = []
    unique_roots = torch.unique(roots)
    for root in unique_roots:
        members = valid_indices[roots == root]
        if members.shape[0] >= min_cluster_size:
            clusters.append(members)

    return clusters
