import json
import random
import click
import wn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import algo.utils
from algo.hierarchical_conformal_prediction import calibrate_covers_mockmodel, optimize_mm, optimize_mm_betas
from algo.propagate import propagate_scores
from algo.no_cover import get_covers, get_leaf_node_cover
import sys
import time

def calibrate():
    pass

def get_scores(dataframe):
    score_columns = [column for column in dataframe.columns if 'label_' in column]
    return dataframe[score_columns].to_numpy()


@click.command()
@click.argument("dataset")
@click.argument("taxonomy")
@click.argument("mapping_file")
@click.argument("output_file")
@click.argument("tree", default=False)
@click.argument("per_layer", default=False)
@click.argument("alpha", default=0.1)
@click.argument("beta", default="1.0")
@click.argument("seed", default=42)
@click.argument("leafs_only", default=False)
@click.argument("method", default="hierarchical")
@click.argument("dynamic", default='True')
@click.argument("loss_approach", default='first')
@click.argument("cp_method", default='lac')
def main(dataset,
         taxonomy,
         mapping_file,
         output_file,
         tree=False,
         per_layer=False,
         alpha=0.1,
         beta=1.0,
         seed=42,
         leafs_only=False,
         method="hierarchical",
         dynamic=True,
         loss_approach='score',
         cp_method='lac'):
    """ Main """
    np.random.seed(seed)
    random.seed(seed)
    # test_size_ratio = 0.2
    test_size_ratio = 0.8# if 'imagenet' in dataset else 0.8
    # test_size_ratio = .5
    loss_approaches = ('costs', 'first', 'last')
    if loss_approach not in loss_approaches:
        raise ValueError('Loss approach should be one of {loss_approaches} but is {loss_approach}')
    
    try:
        beta = float(beta)
    except ValueError:
        if beta == 'betas':
            beta = np.arange(0.0,1.1,step=0.1)
            # extra run for imagenet
            beta = np.append(np.arange(0.01, 0.1, step=0.01), beta)
        else:
            raise ValueError(f"Unknown parameter for beta: {beta}")
    wordnet = wn.Wordnet(taxonomy)
    dataset = pd.read_csv(dataset)
    class_nodes = algo.utils.class_nodes(wordnet)
    node_to_id = {}
    id_to_node = {}
    with open(mapping_file, encoding='utf-8') as f:
        string_mapping = json.load(f)
        for i, string_node in enumerate(string_mapping):
            synset = wn.synset(string_node)
            node_to_id[synset] = i
            id_to_node[i] = synset
    assert all([class_node in node_to_id for class_node in class_nodes])

    # stratified split into calibration and test sets
    scores = get_scores(dataset)
    X_calib, X_test, y_calib, y_test = train_test_split(
        scores,
        dataset.label,
        test_size=test_size_ratio,
        stratify=dataset.label,
        random_state=42,
    )
    n_calib = len(y_calib)
    print(f"y_calib unique labels:{len(np.unique(y_calib))}, y_test unique labels: {len(np.unique(y_test))}")
    
    if leafs_only:
        covers = get_leaf_node_cover(wordnet)
        max_levels_propagate = 0
    else:  
        covers = get_covers(wordnet, per_layer)
        max_levels_propagate = None
    _alpha = alpha
    if(dynamic == 'nobc'):
        dynamic = False
    elif dynamic == "True":
        dynamic =  True
    else:
        dynamic = False
        _alpha = algo.utils.bonferonni_correct(alpha, len(covers), n_calib)
    print('n covers (', per_layer, leafs_only, '): ', len(covers), 'alpha:', _alpha, 'test size (ratio):', len(X_test), f"({test_size_ratio})", 'test size:' 'method:', method, 'dynamic:', dynamic, flush=True)
    node_scores_calib = propagate_scores(X_calib, wordnet, id_to_node, max_levels_propagate)
    node_scores_test = propagate_scores(X_test, wordnet, id_to_node, max_levels_propagate)

    # calibrate on calibration set
    calibrators = calibrate_covers_mockmodel(
        y_calib,
        _alpha,
        covers,
        node_scores_calib,
        id_to_node,
        tree,
        method=method,
        wordnet=wordnet,
        loss_approach=loss_approach,
        cp_method=cp_method,
    )
    # create prediction sets for test set
    if isinstance(beta, float):
        covers, pss, costs, n_covered_leaves = optimize_mm(beta, node_scores_test, calibrators, class_nodes, _alpha, n_calib, tree, dynamic)
        result_frame = pd.DataFrame({
            'costs': costs, 
            'prediction_set_len': map(len, pss),
            'alpha': [alpha,] * len(pss),
            'bf_alpha': [_alpha,] * len(pss),
            'beta': [beta,] * len(pss),
            'prediction_set': pss,
            'best_cover': covers,
            'label_ing': y_test,
            'label': map(lambda x: id_to_node[x], y_test),
            'n_covered_leaves': n_covered_leaves
        })
        mean_covered_leaves = np.array(n_covered_leaves).mean()
        print('costs:', min(costs), max(costs))
        print('ps sizes:', min(map(len, pss)), max(map(len, pss)), sum(map(len, pss))/len(pss))
        print(f'mean covered leaves, {alpha}, {mean_covered_leaves}') 

        # store in file
        result_frame.to_csv(output_file)
    else:
        covers, pss, costs = optimize_mm_betas(beta, node_scores_test, calibrators, class_nodes, _alpha, n_calib, tree, dynamic)
        open(output_file, 'w').close() # create empty file, or remove contents from file if it exists
        for i, b in enumerate(covers):
            result_frame = pd.DataFrame({
                'costs': costs[b], 
                'prediction_set_len': map(len, pss[b]),
                'alpha': [alpha,] * len(pss[b]),
                'bf_alpha': [_alpha,] * len(pss[b]),
                'beta': [b,] * len(pss[b]),
                'prediction_set': pss[b],
                'best_cover': covers[b],
                'label_ing': y_test,
                'label': map(lambda x: id_to_node[x], y_test),
            })
            print('beta:', b)
            print('costs:', min(costs[b]), max(costs[b]))
            print('ps sizes:', min(map(len, pss[b])), max(map(len, pss[b])), sum(map(len, pss[b]))/len(pss[b]))

            # store in file
            if i == 0:
                header = True
            else:
                header = False
            result_frame.to_csv(output_file, mode='a', header=header)


if __name__ == '__main__':
#     # python hcc.py models/gtzan/gtzan_scores.csv gtzan-en data/gtzan/id_to_label.json data/gtzan/hcp_0.1_0.33.csv True False 0.1 .33
#     # python hcc.py models/dbpedia14/dbpedia14_bert.csv dbpedia14-en data/dbpedia14/dbpedia_id2label_omw.json data/dbpedia14/hcp_0.001_0.33.csv True False .001 .33
#     # python hcc.py data/imagenet/resnet152.csv omw-subset-en data/imagenet/id_to_label.json data/imagenet/hcp_0.05_0.33.csv False True 0.05 0.35
    main()

# # beta=0.25
# beta = np.arange(0.0,1.5,step=0.1)
# alpha=0.1
# main(
#     "models/gtzan/gtzan_scores.csv",
#     "gtzan-en",
#     "data/gtzan/id_to_label.json",
#     f"data/gtzan/hcp_{alpha}_betas_cost.csv",
#     tree=False,
#     per_layer=False,
#     alpha=alpha,
#     beta=beta,
#     seed=42,
#     leafs_only=False,
#     dynamic=True,
#     loss_approach='costs')
# # # # beta = 0.33
# # # beta = 0.19
# alpha = .0003
# # # # # alpha = 0.001
# main(
#     "models/dbpedia14/dbpedia14_bert.csv",
#     "dbpedia14-en",
#     "data/dbpedia14/dbpedia_id2label_omw.json",
#     f"data/dbpedia14/hcp_{alpha}_betas_cost.csv",
#     tree=True,
#     per_layer=True,
#     alpha=alpha,
#     beta=beta,
#     seed=42,
#     leafs_only=False,
#     dynamic=True,
#     loss_approach='costs',
# )

# beta = 0.04
# alpha = 0.02
# # # beta = np.arange(0.0,0.1,step=0.01)
# main(
#     "data/imagenet/predictions_val.csv", #dataset
#     "omw-subset-en", # lexicon
#     "data/imagenet/id_to_label.json", # id mapping
#     # f"data/imagenet/hcp_{alpha}_betas_cost.csv", # output file
#     f"data/imagenet/hcp_{alpha}_leafs.csv",
#     tree=False, # tree (False indicates a DAG)
#     per_layer=True, # per_layer (False indicates brute force)
#     alpha=alpha, # target_alpha
#     beta=beta,
#     seed=42,
#     leafs_only=True,
#     method='hierarchical',
#     dynamic=True,
#     loss_approach='first',
#     ) # beta in optimization goal/trade-off set size vs coverage
