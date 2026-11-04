"""
Score propagation through the leaf-cover matrix.

Alternative to ``algo.propagate.propagate_scores`` that computes the propagated
score of every node directly from its definition in the paper,

    g(x)_v = sum_{w in leaf-cov({v})} f(x)_w,

for all nodes at once as a single matrix product ``scores @ M``, where
``M[i, j] = 1`` iff class node ``i`` is node ``j`` or one of its descendants.

Compared to the level-by-level propagation in ``algo.propagate``, this

* gives a score to every ancestor of the class nodes. The level-based
  propagation skips some internal nodes of DAGs with multiple inheritance
  (79 nodes of the ImageNet taxonomy), which then implicitly score 0;
* counts a leaf that is reachable via several paths exactly once per node;
* needs no per-sample taxonomy traversal, caching or multiprocessing.

``propagate_scores_leaf_cover`` is a drop-in replacement for
``propagate_scores(scores, wordnet, id_to_node)`` as called in ``hcc.py``.
``propagate_scores_matrix`` returns the scores as an array, which avoids
building one dictionary per sample.
"""
from typing import Optional

import numpy as np
import wn

from algo.no_cover import ancestors


def leaf_cover_matrix(
        id_class_node_mapping: dict[int, wn.Synset],
) -> tuple[np.ndarray, list[wn.Synset]]:
    """
    Returns the leaf-cover matrix of the class nodes and the nodes it covers.

    Args:
        * id_class_node_mapping: maps the column index of a class in the score
          matrix to its class node. Indices must be ``0, ..., k-1``.

    Returns:
        * ``M`` of shape ``(k, m)`` with ``M[i, j] = 1`` iff class node ``i`` is
          node ``j`` or a descendant of node ``j``.
        * the ``m`` nodes, i.e. all class nodes and their ancestors, sorted by
          id, in the column order of ``M``.
    """
    k = len(id_class_node_mapping)
    if sorted(id_class_node_mapping) != list(range(k)):
        raise ValueError("Class indices must be 0, ..., k-1.")
    covering = {
        i: frozenset(ancestors(node)) | {node}
        for i, node in id_class_node_mapping.items()
    }
    nodes = sorted(frozenset().union(*covering.values()), key=lambda s: s.id)
    column = {node: j for j, node in enumerate(nodes)}
    matrix = np.zeros((k, len(nodes)))
    for i, covering_nodes in covering.items():
        matrix[i, [column[node] for node in covering_nodes]] = 1.0
    return matrix, nodes


def propagate_scores_matrix(
        scores: np.typing.ArrayLike,
        id_class_node_mapping: dict[int, wn.Synset],
) -> tuple[np.ndarray, list[wn.Synset]]:
    """
    Propagated scores of all nodes as an array.

    Args:
        * scores: base classifier scores of shape ``(n, k)``; column ``i``
          belongs to ``id_class_node_mapping[i]``.
        * id_class_node_mapping: maps column indices to class nodes.

    Returns:
        * propagated scores of shape ``(n, m)``.
        * the ``m`` nodes in column order.
    """
    scores = np.asarray(scores, dtype=float)
    matrix, nodes = leaf_cover_matrix(id_class_node_mapping)
    if scores.ndim != 2 or scores.shape[1] != matrix.shape[0]:
        raise ValueError(
            f"Expected scores of shape (n, {matrix.shape[0]}), got {scores.shape}."
        )
    # numpy linked against Apple Accelerate emits spurious floating point
    # warnings for matmul even when all values are finite, so silence them and
    # check the result explicitly instead
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        propagated = scores @ matrix
    if not np.isfinite(propagated).all() and np.isfinite(scores).all():
        raise FloatingPointError("Non-finite propagated scores for finite input scores.")
    return propagated, nodes


def propagate_scores_leaf_cover(
        scores: np.typing.ArrayLike,
        wordnet: wn.Wordnet,
        class_node_id_mapping: dict[int, wn.Synset],
        return_per_level: Optional[bool] = False,
        n_levels: Optional[int] = None,
) -> dict[int, dict[wn.Synset, float]]:
    """
    Drop-in replacement for ``algo.propagate.propagate_scores`` that gives
    every class node and every ancestor its propagated score.

    The signature matches ``propagate_scores``: ``wordnet`` is not needed and
    ``class_node_id_mapping`` maps column indices to class nodes (``id_to_node``
    in ``hcc.py``). Only the flat output is supported, i.e. ``return_per_level``
    must be falsy (``hcc.py`` passes ``0`` or ``None``) and ``n_levels`` must
    be ``None``.

    Returns:
        maps each sample index to a dict of node to propagated score.
    """
    if return_per_level:
        raise NotImplementedError("Per-level output is not supported.")
    if n_levels is not None:
        raise NotImplementedError("Limiting the number of levels is not supported.")
    propagated, nodes = propagate_scores_matrix(scores, class_node_id_mapping)
    # convert row by row to avoid holding a nested list of all values at once
    return {
        i: dict(zip(nodes, row.tolist()))
        for i, row in enumerate(propagated)
    }
