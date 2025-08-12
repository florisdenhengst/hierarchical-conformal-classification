from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import wn


ExampleDAG = namedtuple("ExampleDAG", ['scores', 'probs', 'labels', 'graph'])

num_samples = 10
seed = 42

with open('data/imagenet/id_to_label.json') as f:
    leaf_nodes = list(json.load(f).keys())

def build_wordnet_dag(synset_ids, simulate_root=True):
    """
    Constructs a NetworkX DAG from a list of WordNet synset IDs.
    
    Parameters:
    - synset_ids (list): A list of WordNet synset IDs (e.g., ["dog.n.01", "cat.n.01"]).
    
    Returns:
    - nx.DiGraph: A directed acyclic graph representing the WordNet hierarchy for the given synsets.
    """
    dag = nx.DiGraph()
    
    # Helper function to add synsets and their hypernyms recursively to the DAG
    def add_hypernyms(synset):
        if synset not in dag:
            dag.add_node(synset)
            for hypernym in synset.hypernyms():
                dag.add_edge(hypernym, synset)
                add_hypernyms(hypernym)
    
    # Process each synset ID in the input list
    for synset_id in synset_ids:
        synset = wn.synset(synset_id)
        add_hypernyms(synset)
    
    if simulate_root:
        roots = [node for node in dag.nodes if dag.in_degree(node) == 0]
        if len(roots) > 1:
            dag.add_node('root')
            # add a single simulated root node
            for root in roots:
                dag.add_edge('root', root)    
    return dag

G = build_wordnet_dag(leaf_nodes)
predicted_scores = np.random.rand(num_samples, len(leaf_nodes))
predicted_probs = predicted_scores / predicted_scores.sum(axis=1, keepdims=True)

true_labels = predicted_scores.argmax(axis=1)
imagenet_graph = ExampleDAG(predicted_scores, predicted_probs, true_labels, G)

print(len([node for node in G.nodes if G.out_degree(node) == 0]))
print(len([node for node in G.nodes if G.in_degree(node) == 0]))
print(G.number_of_nodes(), G.number_of_edges())