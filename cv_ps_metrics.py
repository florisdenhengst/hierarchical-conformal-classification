# -*- coding: utf-8 -*-
"""
CV Prediction Sets
Other similarity metrics: 
https://wn.readthedocs.io/en/latest/api/wn.similarity.html?highlight=lowest%20common%20ancestor
"""
import json
import os
import pickle
from collections import Counter
import functools
from typing import List
import click
import itertools
import numpy as np
import wn
from tqdm import tqdm
import pandas as pd
from wn.taxonomy import shortest_path, min_depth
from algo.utils import get_lca

ROOT_STR = 'ROOT'

with open('data/imagenet/imagenet_mapping.json', 'r', encoding='utf-8') as OPENFILE:
    MAPPINGS = json.load(OPENFILE)

def get_synsets(words: List[str]) -> List[wn.Synset]:
    """ Get list of elements of type wn.synset from list of strings """
    res = []
    if isinstance(words[0], list):
        words = [y for x in words for y in x]
    for w in words:
        # if w.lower() in MAPPINGS:
            # res.append(wn.synset(MAPPINGS[w.lower()]))
            # res.append(wn.synset(w.lower()))
        synsets = wn.synset(w)
        if synsets:
            res.append(synsets)
        else:
            raise ValueError(f"No known synset for word '{w}'")
    return list(set(res))


def get_dist_lca(synsets: List[wn.Synset]) -> List[int]:
    """ Distance of each element in the set to LCA """
    if len(synsets) == 1:
        return [0,]
    lca = get_lca(synsets=synsets)
    distances_to_lca = [len(shortest_path(s, lca, simulate_root=True)) for s in synsets]
    return distances_to_lca

def get_node_depths(synsets: List[wn.Synset]) -> List[int]:
    """ Node depths of each synset"""
    return [min_depth(s, simulate_root=True) for s in synsets]


def get_lca_depth(synsets: List[wn.Synset], graph_depth: int) -> float:
    """ LCA Depth, normalized by total graph depth"""
    if len(synsets) == 1:
        # print(synsets[0], wn.taxonomy.min_depth(synsets[0], simulate_root=True))
        lca_depth =  wn.taxonomy.min_depth(synsets[0], simulate_root=True)
        assert lca_depth <= graph_depth, \
            f"lca_depth of {lca_depth} higher than max {graph_depth} for {synsets[0].lemmas()[0]}"
    else:
        lca_depth = min_depth(get_lca(synsets=synsets), simulate_root=True)
    return lca_depth/graph_depth

def get_gwp(synsets: List[wn.Synset]) -> float:
    """ Generalized Wu-Palmer specificity"""
    # NOTE: FdH: 
    if len(synsets) == 1:
        return 1
    else:
        lca_depth = min_depth(get_lca(synsets=synsets), simulate_root=True)
        depths = map(lambda x: min_depth(x, simulate_root=True), synsets)
        return (len(synsets)*lca_depth)/(sum(depths))


def traverse(hyponyms: set[wn.Synset]):
    """ Get leaf nodes of a list of hyponyms """
    leaf_nodes = []
    pending = hyponyms
    while pending:
        synset = pending.pop()
        new_hyponyms = synset.hyponyms()
        if not new_hyponyms:  # synset is a hyponym
            leaf_nodes.append(synset)
        else:  # need to get the hyponyms of the hyponyms
            pending.extend(new_hyponyms)
    return leaf_nodes


def get_leaf_node(synset: wn.Synset):
    """ Leaf node in the taxonomy of the synset:
    - If no hyponyms: the synset is a leaf node 
    - If hyponyms: get the hyponyms of the hyponyms and recurse """
    if not synset.hyponyms():
        return [synset]
    return traverse(hyponyms=synset.hyponyms())


def get_nb_leaf_node(synsets: List[wn.Synset]):
    """ Metric """
    leaf_nodes = []
    for synset in synsets:
        leaf_nodes.extend(get_leaf_node(synset))
        # if synset not in cache:
        #     cache[synset] = get_leaf_node(synset)
        # leaf_nodes.extend(cache[synset])
    # return len(set(leaf_nodes)), cache
    return len(set(leaf_nodes))


class PSMetrics:
    """ Describing prediction sets with metrics """
    def __init__(self, lexicon: str = "omw-en:1.4"):
        """ Init main variables.  Metrics available:
        - `dist_lca`: max distance to LCA
        - `lca_depth`: depth of LCA in taxonomy """
        ewn = wn.Wordnet(lexicon)
        self.graph_depth = wn.taxonomy.taxonomy_depth(ewn, 'n') + 1
        print('lexicon depth: ', self.graph_depth)
        # graph_depth = self.graph_depth
        self.metrics_to_func = {
            'dist_lca': get_dist_lca,
            'lca_depth': lambda synsets: get_lca_depth(synsets=synsets, graph_depth=self.graph_depth),
            'node_depths': get_node_depths,
            'gwp': get_gwp,
            'nb_leaf_nodes': get_nb_leaf_node
        }

    def __call__(self, synsets: List[wn.Synset], metric: str):
        """  """
        if metric not in self.metrics_to_func:
            raise ValueError(f"Metric {metric} is not implemented")

        return self.metrics_to_func[metric](synsets=synsets)


def describe_data(results: dict) -> None:
    """ Describe prediction sets """
    curr_r =  {k: v for k, v in results.items() if len(v["set"]) == 1}
    print("Size of prediction set = 1\n-----")
    print(f"#: {len(curr_r)}")
    label_in_set = Counter([info["label"] in info["set"] for _, info in curr_r.items()])
    print(f"Distribution of set == label: {label_in_set}")
    print("===")
    curr_r =  {k: v for k, v in results.items() if len(v["set"]) > 1}
    print("Size of prediction set > 1\n-----")
    print(f"#: {len(curr_r)}")
    len_set_dis = Counter([len(info["set"]) for _, info in curr_r.items()])
    print(f"Distribution of prediction sets size: {len_set_dis}")
    label_in_set = Counter([info["label"] in info["set"] for _, info in curr_r.items()])
    print(f"Distribution of label in set: {label_in_set}")
    print("===")

def is_root(node: wn.Synset):
    return len(node.hypernyms()) == 0

def get_metrics_lcas(results, lexicon):
    """ Get metrics+lcas """
    for id, info in results.items():
        info["synsets"] = get_synsets(words=info["set"])
        if len(info['synsets']) != len(info['set']):
            print(f"{id}: {len(info['synsets'])} != {len(info['set'])} for {info['synsets']} {info['set']}")
        # assert len(info['synsets']) == len(info['set']), f"{id}: {len(info['synsets'])} != {len(info['set'])} for {info['synsets']} {info['set']}"
    results = {k: v for k, v in results.items() if len(v["synsets"]) > 0}
    print("# of prediction sets with more than zero synset + " + \
            f"label in prediction set: {len(results)}")

    psm = PSMetrics(lexicon=lexicon)
    metrics_ = ['dist_lca', 'lca_depth', 'node_depths', 'gwp', 'nb_leaf_nodes']
    metrics_r = {x: [] for x in metrics_}
    lcas, keys = [], []
    fields = next(iter(results.values())).keys()
    tabular_frame = {}
    for f in fields:
        tabular_frame[f] = []

    for key, info in tqdm(results.items()):
        synsets = get_synsets(info["set"])
        lca = get_lca(synsets=synsets)
        for f in fields:
            tabular_frame[f].append(info[f])
        try:
            lcas.append(lca.lemmas()[0])
        except IndexError as e:
            # special case for the root node -- since this is not associated with any synsets
            if is_root(lca):
                lcas.append(ROOT_STR)
            else:
                raise e
        keys.append(key)
        for m in metrics_:
            curr_m = psm(synsets=synsets, metric=m)
            metrics_r[m].append(curr_m)
    results_frame = pd.DataFrame.from_dict(
        {
            "index": results.keys(),
            "set_size": map(len, tabular_frame["set"],),
            "lca": lcas,
            "dist_lca": metrics_r["dist_lca"],
            "lca_depth": metrics_r["lca_depth"],
            "node_depth": metrics_r["node_depths"],
            "gwp": metrics_r["gwp"],
            'nb_leaf_nodes': metrics_r["nb_leaf_nodes"],
            **tabular_frame,
        }
    )
    results_frame = results_frame.set_index("index")
    return results_frame, lcas

def aggregate_lcas(metrics):
    """ Describe metrics """
    lca_aggregations = [
        (min, 'dist_lca_min'),
        (max, 'dist_lca_max'),
        (np.mean, 'dist_lca_mean'),
    ]
    for function, colname in lca_aggregations:
        metrics[colname] = metrics['dist_lca'].apply(function)
    return metrics

def describe_metrics(metrics_frame):
    metrics = [
        'gwp',
        'lca_depth',
        'dist_lca_min',
        'dist_lca_max',
        'dist_lca_mean',
    ]
    aggregations = [min, max, np.mean]
    for metric in metrics:
        print(metric, end=": ")
        for aggregation in aggregations:
            print(aggregation.__name__, end=" ")
            print(aggregation(metrics_frame[metric]), end=" ")
        print()

def add_human_readable_labels(metrics, mapping):
    """ Add labels for 'Label' and 'Set' columns"""
    # metrics = metrics.rename(columns={'label': 'label_id', 'set': 'set_id'})
    metrics['label_id'] = metrics['label']
    metrics['set_id'] = metrics['set']
    with open(mapping, 'r', encoding='utf-8') as openfile:
        mapping = json.load(openfile)
    try:
        metrics['label_label'] = metrics['label_id'].apply(lambda x: mapping[x])
    except KeyError:
        metrics['label_label'] = metrics['label_id']
    metrics['set_label'] = metrics['set_id'].apply(lambda l: [mapping[x] for x in l])
    return metrics


@click.command()
@click.argument("sets_file")
@click.argument("lexicon")
@click.option("--metrics_file", help="Path to metrics file", default="stats/metrics.json")
@click.option("--lca_file", help="Path to file with lowest-common ancestor for prediction sets", default="stats/lcas.json")
@click.option("--mapping", help="mapping wn id to label")
def main(sets_file, metrics_file, lca_file, lexicon, mapping):
    """ Main script """
    with open(sets_file, "r", encoding="utf-8") as openfile:
        results = json.load(openfile)
    describe_data(results=results)

    if not(os.path.exists(metrics_file)) or not os.path.exists(lca_file):
        metrics, lcas = get_metrics_lcas(results=results, lexicon=lexicon)
        metrics = aggregate_lcas(metrics)
        if mapping is not None:
            metrics = add_human_readable_labels(metrics, mapping)
        metrics.to_pickle(metrics_file)
        with open(lca_file, 'wb') as f:
            pickle.dump(lcas, f)
    metrics = pd.read_pickle(metrics_file)

    with open(lca_file, 'rb') as openfile:
        lcas = pickle.load(openfile)

    describe_metrics(metrics)

    num_examples = 5
    
    for idx, row in metrics.sample(num_examples).iterrows():
        print(row)
        # n = np.random.choice(len(lcas))
        print(f'Random {idx}, set size {len(row.set)}')
        print(row)
        print("====")


if __name__ == '__main__':
    # dbpedia14
    # python cv_ps_metrics.py data/dbpedia14/dbpedia_results_0.0001_omw.json dbpedia14-en --metrics_file ./stats/dbpedia14/dbpedia14.pkl --lca_file ./stats/dbpedia14/lcas_dbpedia14.json --mapping ./data/dbpedia14/dbpedia_id2label_omw.json

    # gtzan
    # python cv_ps_metrics.py data/gtzan/gtzan_results_0.2.json gtzan-en --metrics_file ./stats/gtzan/gtzan.pkl --lca_file ./stats/gtzan/lcas_gtzan.json --mapping ./data/gtzan/id_to_label.json

    # imagenet (alpha \in {0.1, 0.01, 0.05})
    # python cv_ps_metrics.py ./data/imagenet/0.1/results1k_omw.json omw-en:1.4 --metrics_file stats/imagenet/0.1/metrics1k_imagenet_0.1.pkl --lca_file stats/imagenet/0.1/lcas1k_imagenet_0.1.json --mapping data/imagenet/id_to_label.json
    # python cv_ps_metrics.py ./data/imagenet/0.01/results1k_omw.json omw-en:1.4 --metrics_file stats/imagenet/0.01/metrics1k_imagenet_0.01.pkl --lca_file stats/imagenet/0.01/lcas1k_imagenet_0.01.json --mapping data/imagenet/id_to_label.json
    
    
    # python cv_ps_metrics.py ./data/imagenet/0.05/results1k_omw.json omw-en:1.4 --metrics_file stats/imagenet/0.05/metrics1k_imagenet_0.05.pkl --lca_file stats/imagenet/0.05/lcas1k_imagenet_0.05.json --mapping data/imagenet/id_to_label.json
    main()
