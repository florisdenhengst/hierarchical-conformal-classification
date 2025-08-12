import networkx as nx
from itertools import combinations
import pprint
from tree_example import three_layer_tree
from dbpedia14_example import dbpedia14_graph
from dag_example import three_layer_dag
from imagenet_example import imagenet_graph
# from imagenet100_example import imagenet100_graph
import networkx as nx
import functools

def topological_sort(graph):
    # Use NetworkX's topological sort to get nodes in a valid order
    return list(nx.topological_sort(graph))

# @functools.cache
def is_descendant_free(subset, graph):
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
            stack.extend(graph.successors(current))  # Add descendants to stack
    return True

# @functools.cache
def covers_leaf_nodes(subset, leaf_nodes, graph):
    covered = set()
    for node in subset:
        # Include the node itself
        covered.add(node)
        # Add ancestors of the current node
        # for predecessor in nx.predecessor(graph, node):
        for descendant in nx.descendants(graph, node):
            # print('descendant: ', node, "->", predecessor)
            covered.add(descendant)
    # Check if all leaf nodes are covered
    return all(leaf in covered for leaf in leaf_nodes)

def find_root_nodes(graph):
    return [node for node in graph.nodes if graph.in_degree(node) == 0]

# @functools.cache
def get_next_expansion(graph, node):
    direct_descendants = set(graph.successors(node))
    while(len(direct_descendants) == 1):
        direct_descendants = set(graph.successors(direct_descendants.pop()))
    return direct_descendants
    
def top_down_recursive(subset, graph, leaf_nodes, valid_subsets):
    subset_frozen = frozenset(subset)
    if not is_descendant_free(subset_frozen, graph):
        return

    if covers_leaf_nodes(subset_frozen, leaf_nodes, graph) and subset not in valid_subsets:
        valid_subsets.append(subset_frozen)
        subset = subset.copy()
        # Try removing nodes one by one, starting from the top
        for node in list(subset):  # Use list() to avoid modifying the set during iteration
            # Remove the current node and add its direct descendants
            descendants = get_next_expansion(graph, node)
                    
            subset.remove(node)
            subset.update(descendants)  # Add its direct descendants
            
            top_down_recursive(subset, graph, leaf_nodes, valid_subsets)
            
            # Backtrack: restore the original state by removing the descendants and adding the node back
            subset.difference_update(descendants)  # Remove the added descendants
            subset.add(node)  # Add the original node back


def descendant_free_covered_leaf_sets_top_down(graph):
    # Identify all leaf nodes
    leaf_nodes = frozenset([node for node in graph.nodes if graph.out_degree(node) == 0])

    # Start from the full set of root nodes
    root_nodes = find_root_nodes(graph)

    # Initialize valid subsets collection
    valid_subsets = []
    
    # Start with the entire set of root nodes + descendants, collapse recursively
    initial_subset = set(root_nodes)  # Start with the full set of nodes
    top_down_recursive(initial_subset, graph, leaf_nodes, valid_subsets)
    
    return valid_subsets


# dfcp = descendant_free_covered_leaf_sets_top_down(imagenet100_graph[3])
# dfcp = descendant_free_covered_leaf_sets_top_down(three_layer_tree[3])
# dfcp = descendant_free_covered_leaf_sets_top_down(three_layer_dag[3])
# dfcp = descendant_free_covered_leaf_sets_top_down(imagenet_graph[3])
dfcp = descendant_free_covered_leaf_sets_top_down(dbpedia14_graph[3])
pprint.pprint(dfcp)
print(len(dfcp))