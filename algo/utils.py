import wn
import functools
import json
from typing import Iterable, Any
import os
import pickle
import hashlib
import functools

def flatten(input: Iterable[Iterable[Any]]) -> Iterable[Any]:
    return [
        x
        for xs in input
        for x in xs
    ]

@functools.cache
def get_lca(synsets: frozenset[wn.Synset]) -> wn.Synset:
    """ Find Lowest Common Ancestor of a set of synsets.
    The lowest common ancestor or least common subsumer is the node of highest depth of all nodes that
    have all nodes in ``synsets`` as descendant.

    Returns the node if ``synsets`` has only one element. """
    if len(synsets) == 1:
        # the lowest common ancestor is the node itself
        return next(iter(synsets))
    else:
        # if len(ps) == 2 -> one iteration
        # if len(ps) ==3 -> find lca-1 of first two elements, then lca(lca-1, 3rd elements)
        # then recursively
        synsets = list(synsets)
        pending_synsets = synsets[2:]
        lca = wn.taxonomy.lowest_common_hypernyms(synsets[0], synsets[1], simulate_root=True)[0]
        while pending_synsets:
            synset = pending_synsets.pop()
            lca = wn.taxonomy.lowest_common_hypernyms(lca, synset, simulate_root=True)[0]
        return lca

def max_synsets_depth(nodes: [wn.Synset]) -> int:
    """
    Returns the length of the longest hypernym path for ``nodes
    """
    paths = flatten(map(hypernym_paths, nodes))
    return max(map(len, paths))

@functools.cache
def max_synset_depth(node: wn.Synset) -> int:
    """
    Returns the length of the longest hypernym path for ``node``.
    """
    return max([0, *map(len, hypernym_paths(node))])

@functools.cache
def hypernym_paths_flat(node: wn.Synset):
    return flatten(hypernym_paths(node))

@functools.cache
def hypernym_paths(node: wn.Synset):
    """
    Cached wrapper around wn.taxonomy.hypernym_paths.
    """
    return wn.taxonomy.hypernym_paths(node)

def bonferonni_correct(alpha, n_covers, n):
    # TODO: implement check on size of correction
    
    alpha_adj =  alpha / max(1.0, (n_covers-1))
    if n < max([1/alpha_adj, 1/(1-alpha_adj)]):
        alpha_adj = 1.0 / max(1.0, (n-1))
    return alpha_adj

@functools.cache
def descendants_or_ancestors(
        node_a: str,
        node_b: str,
) -> bool:
    """
    Returns ``True`` if ``node_b`` is on the path from ``node_a`` to its root or if ``node_a`` is on the path from ``node_b`` to its root.
    """
    flat_ancestorpaths_a = [
        x
        for xs in hypernym_paths(node_a)
        for x in xs
    ]
    flat_ancestorpaths_b = [
        x
        for xs in hypernym_paths(node_b)
        for x in xs
    ]
    return node_b in flat_ancestorpaths_a or node_a in flat_ancestorpaths_b

@functools.cache
def get_parents(nodes: frozenset[wn.Synset], level:int=None):
    """
    Returns a set of parents for a set of nodes.
    """
    parents = []
    for node in nodes:
        # select parents of this leaf node
        if level is None:
            parents.extend(node.hypernyms())
        else:
            node_level = max_synset_depth(node)
            if node_level <= level:
                parents.append(node)
            else:
                parents.extend(node.hypernyms())
    parents = frozenset(parents)
    return parents

@functools.cache
def all_leaves(wordnet: wn.Synset) -> set[wn.Synset]:
    """
    Returns bottom noun-synsets, i.e. all synsets with pos=n that have no hyponyms, in `wordnet`
    """
    leaves = set(wn.taxonomy.leaves(wordnet, pos='n'))
    return frozenset(leaves)

def get_wn_name(wordnet: wn.Synset) -> str:
    """
    Returns human-readable name of `wordnet`
    """
    return str(wordnet.lexicons()[0])

def class_nodes(wordnet: wn.Synset) -> set[wn.Synset]:
    wordnet_name = get_wn_name(wordnet)
    if wordnet_name in ("<Lexicon gtzan-en:1.0 [en]>", "<Lexicon hcptest-en:1.0 [en]>", "<Lexicon hcptestdag-en:1.0 [en]>", "<Lexicon dbpedia14-en:1.0 [en]>"):
        return all_leaves(wordnet)
    elif "omw-subset-en" in wordnet_name:
        # assuming dataset is imagenet
        with open("data/imagenet/id_to_label.json", "r") as f:
            id_to_label = json.load(f)
        return frozenset({
            wn.synset(idx) for idx in id_to_label.keys()
        })
    elif "omw-subset200-en" in wordnet_name:
        with open("data/imagenet/id_to_label200.json", "r") as f:
            id_to_label = json.load(f)
        return frozenset({
            wn.synset(idx) for idx in id_to_label.keys()
        })
    raise ValueError("Missing class nodes for Wordnet with name '{}'".format(wordnet_name))


def disk_cache(cache_dir='/tmp/cache'):
    os.makedirs(cache_dir, exist_ok=True)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Serialize args and kwargs to create a unique hash
            key_bytes = pickle.dumps((args, kwargs))
            key_hash = hashlib.md5(key_bytes).hexdigest()
            cache_path = os.path.join(cache_dir, f"{func.__name__}_{key_hash}.pkl")

            # Check for cached result
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

            # Compute and cache result
            result = func(*args, **kwargs)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            return result
        return wrapper
    return decorator