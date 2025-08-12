from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

ExampleDAG = namedtuple("ExampleDAG", ['scores', 'probs', 'labels', 'graph'])

num_samples = 10
seed = 42

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to represent the structure
G.add_node("owl:Thing")

# Define the relationships (edges) between nodes
edges = [
    ("owl:Thing", ":Agent"),
    (":Agent", ":Organisation"),
    (":Organisation", ":EducationalInstitution"),
    (":Organisation", ":Company"),
    ("owl:Thing", ":Species"),
    (":Species", ":Eukaryote"),
    (":Eukaryote", ":Animal"),
    (":Animal", ":Person"),
    (":Person", ":OfficeHolder"),
    (":Person", ":Athlete"),
    (":Person", ":Artist"),
    (":Eukaryote", ":Plant"),
    ("owl:Thing", ":MeansOfTransport"),
    ("owl:Thing", ":Work"),
    (":Work", ":Album"),
    (":Work", ":Film"),
    (":Work", ":WrittenWork"),
    ("owl:Thing", ":Settlement"),
    (":Settlement", ":Village"),
    ("owl:Thing", ":Place"),
    (":Place", ":NaturalPlace"),
    ("owl:Thing", ":ArchitecturalStructure"),
    (":ArchitecturalStructure", ":Building")
]


# Add edges to the graph
G.add_edges_from(edges)


leaf_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
predicted_scores = np.random.rand(num_samples, len(leaf_nodes))
predicted_probs = predicted_scores / predicted_scores.sum(axis=1, keepdims=True)

true_labels = predicted_scores.argmax(axis=1)
dbpedia14_graph = ExampleDAG(predicted_scores, predicted_probs, true_labels, G)