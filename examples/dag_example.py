from collections import namedtuple
import networkx as nx
from pyvis.network import Network
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import pandas as pd
import random

ExampleDAG = namedtuple("ExampleDAG", ['scores', 'probs', 'labels', 'graph'])

num_samples = 10
seed = 42

np.random.seed(seed)
random.seed(seed)

# Create a directed graph
G = nx.DiGraph(data='data')

# Add nodes and edges for Top-Level Category T1 (Vertebrates)
G.add_node("root")
G.add_node("T1 (BioAnimals)")
G.add_node("S1 (Mammals)")
G.add_node("S2 (Birds)")
G.add_node("S3 (Insects)")

G.add_edges_from([
    ("T1 (BioAnimals)", "S1 (Mammals)"),
    ("T1 (BioAnimals)", "S2 (Birds)"),
    ("T1 (BioAnimals)", "S3 (Insects)")
])

# Add nodes and edges for Top-Level Category T2 (Invertebrates)
G.add_node("T2 (MoveableAnimals)")
G.add_node("S4 (FlyingAnimals)")
G.add_node("S5 (WalkingAnimals)")
G.add_node("S6 (CrawlingAnimals)")

G.add_edges_from([
    ("T2 (MoveableAnimals)", "S4 (FlyingAnimals)"),
    ("T2 (MoveableAnimals)", "S5 (WalkingAnimals)"),
    ("T2 (MoveableAnimals)", "S6 (CrawlingAnimals)"),
    ("root", "T2 (MoveableAnimals)"),
    ("root", "T1 (BioAnimals)"),
])

# Add nodes and edges for Subclasses under T2
edges = [
    ("S6 (CrawlingAnimals)", "C1 (Platypus)"),
    ("S1 (Mammals)", "C1 (Platypus)"),
    ("S1 (Mammals)", "C2 (Dog)"),
    ("S2 (Birds)", "C3 (Duck)"),
    ("S4 (FlyingAnimals)", "C3 (Duck)"),
    ("S2 (Birds)", "C4 (Kiwi)"),
    ("S5 (WalkingAnimals)", "C4 (Kiwi)"),
    ("S5 (WalkingAnimals)", "C5 (Ant)"),
    ("S3 (Insects)", "C5 (Ant)"),
    ("S5 (WalkingAnimals)", "C6 (Spider)"),
    ("S3 (Insects)", "C6 (Spider)"),
    # ("T1 (BioAnimals)", "C7 (Flying Fish)"),
    # ("S4 (FlyingAnimals)", "C7 (Flying Fish)"),
]
leaf_nodes = sorted(list(set([edge[1] for edge in edges])))
leaf_node_id_mapping = {i: name for i, name in enumerate(leaf_nodes)}
id_leaf_node_mapping = {name: i for i, name in enumerate(leaf_nodes)}
G.add_edges_from(edges)


# # Perform topological sort to get the hierarchical order
# topological_order = list(nx.topological_sort(G))

# # Assign positions for each node based on topological sorting
# # Root at the top, levels descend downwards
# pos = {}
# layer_gap = 200  # vertical gap between layers
# node_gap = 100   # horizontal gap between nodes in the same layer
# layers = {}

# # Arrange nodes into layers based on distance from the root
# for i, node in enumerate(topological_order):
#     layer = nx.shortest_path_length(G, source=topological_order[0], target=node)
#     if layer not in layers:
#         layers[layer] = []
#     layers[layer].append(node)

# # Set positions for nodes in each layer
# for layer, nodes in layers.items():
#     for j, node in enumerate(nodes):
#         pos[node] = (j * node_gap, -layer * layer_gap)

# # Initialize PyVis Network with directed layout
# net = Network(directed=True, height="600px", width="100%", bgcolor="#ffffff", font_color="black")
# net.from_nx(G)

# # Apply custom positions to PyVis nodes
# for node in net.nodes:
#     node_id = node['id']
#     node['x'] = pos[node_id][0]
#     node['y'] = pos[node_id][1]

# # Customize the appearance
# net.set_options("""
# var options = {
#   "edges": {
#     "arrows": {
#       "to": {"enabled": true, "scaleFactor": 0.5}
#     },
#     "color": {"inherit": false}
#   },
#   "nodes": {
#     "shape": "dot",
#     "size": 15,
#     "font": {"size": 14}
#   },
#   "layout": {
#     "hierarchical": {
#       "enabled": true,
#       "sortMethod": "directed"
#     }
#   },
#   "physics": {
#     "enabled": false
#   }
# }
# """)

# # Save and view the graph in the browser
# net.show("topological_dag.html", notebook=False)

predicted_scores = np.random.rand(num_samples, len(leaf_nodes))
predicted_probs = predicted_scores / predicted_scores.sum(axis=1, keepdims=True)

true_labels = predicted_scores.argmax(axis=1)

three_layer_dag = ExampleDAG(predicted_scores, predicted_probs, true_labels, G)