from collections import namedtuple
import networkx as nx
from pyvis.network import Network
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

# Add nodes and edges for Top-Level Category
G.add_node("root")
G.add_node("T1 (Lunch)")
G.add_node("T2 (Breakfast)")

G.add_node("S1 (Sandwiches)")
G.add_node("S2 (Salads)")

G.add_edges_from([
    ("root", "T1 (Lunch)"),
    ("root", "T2 (Breakfast)"),
    ("T1 (Lunch)", "S1 (Sandwiches)"),
    ("T1 (Lunch)", "S2 (Salads)"),
])

# Add nodes and edges for leaf nodes
edges = [
    ("S1 (Sandwiches)", "C1 (Cheese)"),
    ("S1 (Sandwiches)", "C2 (Ham)"),
    ("S1 (Sandwiches)", "C3 (Tuna)"),
    ("S2 (Salads)", "C4 (CeasarSalad)"),
    ("S2 (Salads)", "C5 (GreekSalad)"),
    ("T2 (Breakfast)", "C6 (Omelette)"),
    ("T2 (Breakfast)", "C7 (Pancakes)"),
]
leaf_nodes = sorted(list(set([edge[1] for edge in edges])))
leaf_node_id_mapping = {i: name for i, name in enumerate(leaf_nodes)}
id_leaf_node_mapping = {name: i for i, name in enumerate(leaf_nodes)}
G.add_edges_from(edges)

# Perform topological sort to get the hierarchical order
topological_order = list(nx.topological_sort(G))

# Assign positions for each node based on topological sorting
# Root at the top, levels descend downwards
pos = {}
layer_gap = 200  # vertical gap between layers
node_gap = 100   # horizontal gap between nodes in the same layer
layers = {}

# Arrange nodes into layers based on distance from the root
for i, node in enumerate(topological_order):
    layer = nx.shortest_path_length(G, source=topological_order[0], target=node)
    if layer not in layers:
        layers[layer] = []
    layers[layer].append(node)

# Set positions for nodes in each layer
for layer, nodes in layers.items():
    for j, node in enumerate(nodes):
        pos[node] = (j * node_gap, -layer * layer_gap)

# Initialize PyVis Network with directed layout
net = Network(directed=True, height="600px", width="100%", bgcolor="#ffffff", font_color="black")
net.from_nx(G)

# Apply custom positions to PyVis nodes
for node in net.nodes:
    node_id = node['id']
    node['x'] = pos[node_id][0]
    node['y'] = pos[node_id][1]

# Customize the appearance
net.set_options("""
var options = {
  "edges": {
    "arrows": {
      "to": {"enabled": true, "scaleFactor": 0.5}
    },
    "color": {"inherit": false}
  },
  "nodes": {
    "shape": "dot",
    "size": 15,
    "font": {"size": 14}
  },
  "layout": {
    "hierarchical": {
      "enabled": true,
      "sortMethod": "directed"
    }
  },
  "physics": {
    "enabled": false
  }
}
""")

# # Save and view the graph in the browser
# net.show("topological_tree.pdf", notebook=False)

predicted_scores = np.random.rand(num_samples, len(leaf_nodes))
predicted_probs = predicted_scores / predicted_scores.sum(axis=1, keepdims=True)

true_labels = predicted_scores.argmax(axis=1)

three_layer_tree = ExampleDAG(predicted_scores, predicted_probs, true_labels, G)