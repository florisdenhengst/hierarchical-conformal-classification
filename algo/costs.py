import wn
from algo.no_cover import covered_nodes
from algo.utils import flatten
import functools
from typing import Optional, Callable, Tuple

@functools.cache
def get_max_cost_cover(beta, cover: frozenset[wn.Synset], class_nodes: frozenset[wn.Synset]) -> float:
    return cost(beta, cover, class_nodes, cover)[0]

@functools.cache
def get_cost_empty_ps(beta, cover: frozenset[wn.Synset], class_nodes: frozenset[wn.Synset]) -> float:
    # returns the max cost for a cover
    return get_max_cost_cover(cover, class_nodes) - 1

def cost(beta,
         prediction_set: list[list[wn.Synset]],
         class_nodes: frozenset[wn.Synset],
         cover: frozenset[wn.Synset],
         cost_function_empty: Optional[Callable] = None) -> Tuple[float, int]:
    if len(prediction_set) == 0:
        if cost_function_empty is None:
            cost_function_empty = lambda *x: 0.0
        return cost_function_empty(beta, cover, class_nodes), len(class_nodes)
    if isinstance(prediction_set, list) and isinstance(prediction_set[0], list):
        prediction_set = flatten(prediction_set)
    if isinstance(class_nodes, list):
        raise ValueError("Class_nodes is a list, should be a set")
    try:
        covered_nodes_ps = covered_nodes(frozenset(prediction_set), class_nodes)
    except:
        print('error in calculating cost')
        raise ValueError('Error!')
    cost = len(prediction_set) + beta * (len(covered_nodes_ps))
    return cost, len(covered_nodes_ps)