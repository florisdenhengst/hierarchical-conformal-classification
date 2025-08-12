import functools
import wn
from typing import Optional
import sys
from algo import utils
from collections import defaultdict

@functools.cache
def filter_candidate_covers(
    candidates: frozenset[wn.Synset],
    lca: wn.Synset,
    ) -> frozenset[wn.Synset]:
    """
    Returns the candidate covers that do not contain any ancestors of ``lca``.
    """
    def has_no_ancestor(candidate, lca):
        return not(any(map(lambda x: covers(x, lca), candidate)))
    return frozenset(filter(
        lambda x: has_no_ancestor(x, lca),
        candidates,
    ))


@functools.cache
def is_descendant_free(subset: set[wn.synset]) -> bool:
    """
    Returns False when `subset` contains elements that are each others' (indirect) hypernym or hyponym and True otherwise.
    """
    visited = set()
    for node in subset:
        if node in visited:
            return False
        # Perform a depth-first search (DFS) to find descendants
        stack = [node]
        while stack:
            current = stack.pop()
            if current in subset and current != node:
                return False
            visited.add(current)
            stack.extend(current.hyponyms())  # add children to stack
    return True

@functools.cache
def covers(
    a: wn.Synset,
    b: wn.Synset
)-> bool:
    """
    Returns True if ``b`` is covered by ``a``, i.e. when ``b`` is a descendant of ``a``.
    """
    return b in descendants(a)
    

@functools.cache
def covered_nodes(synsets: set[wn.Synset], nodes_of_interest: Optional[set[wn.Synset]]=None) -> set[wn.Synset]:
    """ Get all covered nodes of a set of synsets """
    covered = set()
    for synset in synsets:
        covered.add(synset) #include node itself
        covered |= descendants(synset)
        if nodes_of_interest is not None:
            covered &= nodes_of_interest
    return {c for c in covered if c.id != "*INFERRED*"}

@functools.cache
def covers_all_nodes(synsets: set[wn.Synset], nodes: set[wn.Synset]) -> bool:
    """Checks whether ``synsets`` covers all ``nodes``. """
    covered = covered_nodes(synsets)
    return all(node in covered for node in nodes)

@functools.cache
def descendants(node: wn.Synset) -> set[wn.Synset]:
    descendants = set()
    pending = [node]
    while pending:
        synset = pending.pop()
        descendants.add(synset)
        pending.extend(synset.hyponyms())
    return descendants

@functools.cache
def ancestors(node: wn.Synset) -> set[wn.Synset]:
    """
    Returns all ancestors of a node.
    """
    path = set()
    pending = [node]
    while pending:
        synset = pending.pop()
        if synset != node:
            path.add(synset)
        parents = synset.hypernyms()
        if parents:
            pending.extend(parents)
    return path

def top_down_recursive(subset: set[wn.synset], class_nodes: set[wn.synset], valid_subsets: list[set[wn.synsets]]) -> None:
    """Adds all non-overlapping covers in the subgraph defined by ``subsets`` to ``valid_subsets``."""
    subset_frozen = frozenset(subset)
    if not is_descendant_free(subset_frozen):
        return

    if covers_all_nodes(subset_frozen, class_nodes) and subset not in valid_subsets:
        valid_subsets.append(subset_frozen)
        subset = subset.copy()
        # Try removing nodes one by one, starting from the top
        for node in list(subset):  # Use list() to avoid modifying the set during iteration
            # Remove the current node and add its direct descendants
            hyponyms = node.hyponyms()

            subset.remove(node)
            subset.update(hyponyms)  # Add its direct descendants

            top_down_recursive(subset, class_nodes, valid_subsets)

            # Backtrack: restore the original state by removing the descendants and adding the node back
            subset.difference_update(hyponyms)  # Remove the added descendants
            subset.add(node)  # Add the original node back

def descendant_free_covered_class_sets_top_down(wordnet: wn.Wordnet):
    # Identify all class nodes
    class_nodes = utils.class_nodes(wordnet)
    taxonomy_depth = wn.taxonomy.taxonomy_depth(wordnet, pos='n')
    if taxonomy_depth >= sys.getrecursionlimit():
        raise ValueError(f"Recursion limit too small to calculate all no covers, set to value higher than {taxonomy_depth}")

    # Start from the full set of root nodes
    root_nodes = wn.taxonomy.roots(wordnet, pos='n')

    # Initialize valid subsets collection
    valid_subsets = []
    
    # Start with the entire set of root nodes + descendants, collapse recursively
    initial_subset = set(root_nodes)  # Start with the full set of nodes
    top_down_recursive(initial_subset, class_nodes, valid_subsets)
    return valid_subsets



def get_covers_layer_bottom_up(wordnet):
    def flatten_set(xss):
        return {x for xs in xss for x in xs}

    def move_frontier(frontier, candidates):
        non_overlapping = [
            candidate for candidate in candidates
            if covered_nodes(frozenset({candidate,}), frozenset(candidates-{candidate,})) == frozenset({})
            ]
        if non_overlapping:
            new_frontier = non_overlapping
            non_covered_old_frontier = [node for node in frontier if not node in covered_nodes(frozenset(new_frontier))]
            return set(new_frontier + non_covered_old_frontier)
        else:
            return non_overlapping
        
    def get_hypernyms(node):
        candidates = node.hypernyms()
        result = set()
        for c in candidates:
            # parent has only child, traverse up
            if len(c.hyponyms()) == 1:
                candidates.extend(c.hypernyms())
            else:
                result.add(c)
        return frozenset(result)

    root = wn.taxonomy.roots(wordnet)[0]
    leaves = utils.all_leaves(wordnet)
    max_depth = max([wn.taxonomy.max_depth(l) for l in leaves])

    layers = {0: leaves}
    frontier = leaves
    frontier_depth = max_depth
    layer_i = 0
    while frontier:
        layer_i += 1
        candidates = flatten_set([get_hypernyms(node) for node in frontier])
        frontier = move_frontier(frontier, candidates)
        if frontier:
            layers[layer_i] = frozenset(frontier)
    return layers



def get_cover_layer_top_down(wordnet: wn.Wordnet):
    """
    Covers per layer.

    Since e.g. Imagenet's taxonomy is much bigger, we restrict the possible covers, and use a layered-approach:
    - the root nodes
    - all their hyponyms
    etc
    - until all the leave nodes

    if by taking the hyponyms of a node we reach a leaf node, we keep the node in the following covers
    """
    cover = frozenset(wn.taxonomy.roots(wordnet, pos='n'))
    covers = [cover]
    to_keep, to_expand = [], cover
    leaves = wn.taxonomy.leaves(wordnet, pos='n')

    while to_expand:
        new_expand = []
        for synset in to_expand:
            children = [x for x in synset.hyponyms() if x not in to_keep]
            for p in children:
                if p in leaves:
                    to_keep.append(p)
                else:
                    new_expand.append(p)
        covers.append(frozenset(new_expand+to_keep))
        to_expand = new_expand

    return covers

def get_covers(wordnet: wn.Wordnet, per_layer=False):
    if per_layer:
        # root = wn.taxonomy.roots(wordnet, pos='n')[0]
        return list(get_covers_layer_bottom_up(wordnet).values())
    else:
        return descendant_free_covered_class_sets_top_down(wordnet)

def get_leaf_node_cover(wordnet: wn.Wordnet, per_layer=False):
    """
    Returns a set of covers containing a single element: the cover with all leaf nodes in the taxonomy.
    """
    return [utils.all_leaves(wordnet),]