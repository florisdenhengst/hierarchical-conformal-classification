import numpy as np
import wn
import functools
import algo.no_cover
from algo.utils import get_lca, descendants_or_ancestors, get_parents, max_synsets_depth, disk_cache
import tqdm

def propagate_ground_truths(
        ground_truths: np.typing.ArrayLike,
        id_class_node_mapping: dict[int, wn.Synset],
    ) -> set[wn.Synset]:
    ground_truth_synsets = list(map(lambda x: id_class_node_mapping[x], ground_truths))
    result = []
    print("Propagating ground truths")
    for base_truth in tqdm.tqdm(ground_truth_synsets):
        hypernym_path = algo.utils.hypernym_paths_flat(base_truth)
        hypernym_path.append(base_truth)
        result.append(
            frozenset(sorted(hypernym_path))
        )
    return result

def propagate_scores_sample(
        scores: np.typing.ArrayLike,
        nodes: frozenset[wn.Synset],
        class_nodes: frozenset[wn.Synset],
        id_class_node_mapping: dict[int, wn.Synset],
        level: int,
        )->np.typing.ArrayLike:
    """
    Aggregates all scores one level up:

    Input:
    """
    parents = get_parents(nodes, level)

    parent_scores = {}
    for parent in parents:
        covered_class_nodes = algo.no_cover.covered_nodes(frozenset({parent,}), class_nodes)
        class_ids = [id_class_node_mapping[l] for l in covered_class_nodes]
        class_scores = [scores[l] for l in class_ids]
        parent_scores[parent] = sum(class_scores)
    return parent_scores

@disk_cache()
def propagate_scores(
        scores: np.typing.ArrayLike,
        wordnet: wn.Wordnet,
        class_node_id_mapping: dict[wn.Synset, int],
        return_per_level=False,
        n_levels=None,
    )->np.typing.ArrayLike:
    aggregate_scores = {}
    id_class_node_mapping = {idx: node_id for node_id, idx in class_node_id_mapping.items()}
    class_nodes = frozenset(class_node_id_mapping.values())
    # levels = wn.taxonomy.taxonomy_depth(wordnet, pos='n')
    levels = max_synsets_depth(class_nodes)
    print("propagating scores...")
    i = 0
    for sample in tqdm.tqdm(scores):
        initial_scores =  {class_node_id_mapping[idx]: float(score) for idx, score in enumerate(sample)}
        if return_per_level:
            aggregate_scores[i] = {levels: initial_scores}
        else:
            aggregate_scores[i] = initial_scores
        nodes_i = class_nodes
        for level in range(levels)[::-1]:
            aggregated_scores = propagate_scores_sample(sample, nodes_i, class_nodes, id_class_node_mapping, level)
            if return_per_level:
                aggregate_scores[i][level] = aggregated_scores
            else:
                aggregate_scores[i] |= aggregated_scores
            parents = get_parents(frozenset(nodes_i), level)
            nodes_i = filter_nodes(parents)
            if n_levels is not None and level <= n_levels:
                break
        i += 1
    return aggregate_scores

@functools.cache
def find_descendant_groups(nodes):
    # Initialize a list to store sets of descendant groups
    descendant_groups = []
    
    # Keep track of nodes that are already part of a group
    visited = set()

    # Iterate over all nodes
    for node_a in nodes:
        if node_a in visited:
            continue  # Skip nodes that are already part of a group
        
        # Create a set for the current group
        group = set([node_a])
        
        # Check descendant/ancestor relationships with all other nodes
        for node_b in nodes:
            if node_a != node_b and descendants_or_ancestors(node_a, node_b):
                group.add(node_b)
        
        # Add the group to the list and mark nodes as visited
        descendant_groups.append(group)
        visited.update(group)
    
    return descendant_groups

@functools.cache
def filter_nodes(
        nodes: set[wn.Synset]
) -> list[str]:
    """
    Returns all lowest common ancestors of groups of nodes that are not in the same subgraph. 
    """
    grouped_ancestors = find_descendant_groups(nodes)
    lcas = []
    for g in grouped_ancestors:
        lcas.append(get_lca(frozenset(g)))
    return frozenset(lcas)