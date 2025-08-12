import copy
from collections import defaultdict
import numpy as np
import wn
from algo.propagate import propagate_ground_truths
from algo.no_cover import covered_nodes
from mapie.classification import MapieClassifier
from mapie.hierarchical_classification import HierarchicalMapieClassifier 
from mapie.multi_label_classification import MapieMultiLabelClassifier
import pprint
import scipy
import tqdm
from algo.hcp_mock_model import HCPMockModel
from algo import utils, no_cover
from algo.costs import cost


def format_for_cover(scores, class_node_id_cover):
    # assign prediction scores for each instance
    # Using Mauricio Sadinle, Jing Lei, and Larry Wasserman.
    # "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.",
    # Journal of the American Statistical Association, 114, 2019.
    
    # TODO sort dict by id-mapping
    scores_values_matrix = np.zeros((len(scores.values()), len(next(iter(scores.items()))[1])))
    for node, id in class_node_id_cover.items():
        scores_values_matrix[id, :] = scores[node]
    scores_matrix = scores_values_matrix.transpose()
    # return scipy.special.softmax(scores_matrix, axis=1)
    return scores_matrix

def _to_padded_matrix(frozensets_list, class_node_id_cover):
    """
    Organize frozensets of various lengths into a numpy matrix, padding with copies of the last element.

    :param frozensets_list: List of frozensets to organize.
    :return: Numpy array forming a matrix with padding.
    """
    # Convert frozensets to lists for easier manipulation
    list_sets = [list(map(lambda x: class_node_id_cover[x], fset)) for fset in frozensets_list]

    # Find the length of the longest frozenset for padding
    max_length = max(len(lst) for lst in list_sets)

    # Pad each list with copies of the last element to match max_length
    padded_matrix = np.array([
        lst + [lst[-1]] * (max_length - len(lst)) if lst else [None] * max_length
        for lst in list_sets
    ])

    filtered = padded_matrix[~np.all(padded_matrix == None, axis=1)]

    return filtered.astype(np.int64)

def cover_ground_truths(cover, class_node_id_cover, hierarchical_ground_truths):
    ground_truths = _cover_ground_truths(cover, hierarchical_ground_truths)
    ground_truths = _to_padded_matrix(ground_truths, class_node_id_cover)
    return ground_truths

def _cover_ground_truths(
        cover: frozenset[wn.Synset],
        hierarchical_ground_truths,
):
    return [hgt & cover for hgt in hierarchical_ground_truths]


def cover_scores(
        cover: frozenset[wn.Synset],
        scores_per_node: dict[wn.Synset, float]
)-> set[wn.Synset, float]:
    """
    Returns the scores for all nodes in ``cover`` and a new mapping of synsets to ids for this `cover`.
    """
    result_scores = {}
    synset_id_map = {}
    n = len(next(iter(scores_per_node.values())))
    for i, node in enumerate(cover):
        if node in scores_per_node:
            result_scores[node] = scores_per_node[node]
        else:
            result_scores[node] = [np.float64(0.0),] * n
    #     synset_id_map[node] = i
    # for synset, scores in scores_per_node.items():
    #     if synset in cover:
    #         result_scores[synset] = scores
    #         synset_id_map[synset] = i
    #         i += 1
    return result_scores, synset_id_map

def nonconformity_scores(scores):
    return 1 - scores

def calibrate_cover_naive(
        scores: np.typing.ArrayLike,
        true_labels:np.typing.ArrayLike,
        alpha:float=0.1) -> float:
    """
    Function to calculate the empirical quantile for the desired alpha level
    given the scores and true labels.
    """
    n = scores.shape[0]
    try:
        if len(true_labels.shape) > 1:
            # NOTE take the max in case there are multiple ground truths
            selected_scores = np.take_along_axis(scores, true_labels, axis=1).max(axis=1)
        else:
            selected_scores = scores[np.arange(n), true_labels]
    except Exception as e:
        print('error:', true_labels.shape, scores.shape, true_labels, scores)
        raise e
    nonconformity_cal_scores = nonconformity_scores(selected_scores)
    q_level = min(np.ceil((n+1) * (1-alpha))/n, 1.0)
    qhat = np.quantile(nonconformity_cal_scores, q_level, method='higher')
    return qhat

def calibrate_cover_adaptive(
        scores,
        true_labels,
        alpha):
    n = scores.shape[0]
    cal_pi = scores.argsort(1)[:,::-1]
    cal_srt = np.take_along_axis(cal_pi,cal_pi,axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt,cal_pi.argsort(axis=1),axis=1)[range(n),true_labels]
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')
    return qhat

def _scores_per_node(node_scores):
    scores_per_node = defaultdict(list)
    for inner_dict in node_scores.values():
        for synset, prob in inner_dict.items():
            scores_per_node[synset].append(prob)
    return scores_per_node

def calibrate_covers(
        # base_scores: np.typing.ArrayLike,
        true_labels: np.typing.ArrayLike,
        alpha: float,
        covers: set[set[wn.Synset]],
        node_scores: dict[wn.Synset, float],
        id_class_node_mapping: dict[int, wn.Synset],
        wordnet,
        ) -> dict[set[wn.Synset], float]:
    cutoffs = {} # maps covers to empirical quantiles
    hierarchical_ground_truths = propagate_ground_truths(
        true_labels,
        id_class_node_mapping
    )
    scores_per_node = _scores_per_node(node_scores)
    leafs = utils.all_leaves(wordnet)
    # loop over all covers
    print('Calibrating...')
    for cover in tqdm.tqdm(covers):
        if len(cover) == 1:
            cutoffs[cover] = 1.0
        else:
            scores, _ = cover_scores(cover, scores_per_node)
            class_node_id_cover = {synset: idx for idx, synset in enumerate(cover)}
            id_class_node_cover = {idx: synset for synset, idx in class_node_id_cover.items()}
            costs_per_node = np.array([
                covered_nodes(frozenset({id_class_node_cover[idx]}), leafs) for idx in range(len(id_class_node_cover))
            ])
            {idx: covered_nodes(frozenset({synset}), leafs) for synset, idx in class_node_id_cover.items()}
            if (len(scores) == 0):
                continue
            scores_matrix = format_for_cover(scores, class_node_id_cover)
            ground_truth_ids = cover_ground_truths(cover, class_node_id_cover, hierarchical_ground_truths)
            if ground_truth_ids.max() >= scores_matrix.shape[1]:
                raise ValueError('Wrong index for ground truth')
            # propagate true labels
            cutoffs[cover] = calibrate_cover_naive(scores_matrix, ground_truth_ids, alpha, costs_per_node)
            # calibrate: determine qhat
    # return dict[wn.Synset, qhat]
    return cutoffs


def calibrate_covers_mockmodel(
        # base_scores: np.typing.ArrayLike,
        true_labels: np.typing.ArrayLike,
        alpha: float,
        covers: set[set[wn.Synset]],
        node_scores: dict[wn.Synset, float],
        id_class_node_mapping: dict[int, wn.Synset],
        tree=False,
        method='hierarchical',
        wordnet=None,
        loss_approach='costs',
        cp_method='lac'
        ) -> dict[set[wn.Synset], float]:
    calibrators = {} # maps covers to CP models
    hierarchical_ground_truths = propagate_ground_truths(
        true_labels,
        id_class_node_mapping
    )
    scores_per_node = _scores_per_node(node_scores)
    leafs = utils.all_leaves(wordnet)
    # loop over all covers
    print('calibrating...')
    for cover in tqdm.tqdm(covers):
        if len(cover) == 1:
            calibrators[cover] = (None, None)
        else:
            scores, _ = cover_scores(cover, scores_per_node)
            class_node_id_cover = {synset: idx for idx, synset in enumerate(cover)}
            id_class_node_cover = {id: node for node, id in class_node_id_cover.items()}
            costs_per_node = np.array([
                len(covered_nodes(frozenset({id_class_node_cover[idx]}), leafs)) for idx in range(len(id_class_node_cover))
            ])
            if (len(scores) == 0):
                continue
            scores_matrix = format_for_cover(scores, class_node_id_cover)
            _ground_truth_ids = cover_ground_truths(cover, class_node_id_cover, hierarchical_ground_truths)
            if tree:
                ground_truth_ids = _ground_truth_ids
            else:
                # Number of classes (k)
                k = max(class_node_id_cover.values())+1  # Example: 4 possible classes

                # Create the boolean encoded matrix
                n = len(_ground_truth_ids)
                bool_encoded_matrix = np.zeros((n, k), dtype=bool)

                # Set True where class is present
                for i, row in enumerate(_ground_truth_ids):
                    bool_encoded_matrix[i, row] = True
                ground_truth_ids = bool_encoded_matrix

            if ground_truth_ids.max() >= scores_matrix.shape[1]:
                raise ValueError('Wrong index for ground truth')
            
            n_ground_truths = ground_truth_ids.sum(axis=1)

            if tree:
                hierarchical = False
                ground_truth_ids = _ground_truth_ids.ravel()
                _method='score'
            elif n_ground_truths.max() == 1 and method == 'hierarchical':
                hierarchical = False
                ground_truth_ids = _ground_truth_ids.ravel()
                _method='score'
            elif method == 'hierarchical':
                hierarchical = True
                _method=loss_approach
            elif method == 'multi-label':
                hierarchical = True
                _method = 'multi_label'

            # print(f'n_ground_truths: ({n_ground_truths.min()}, {n_ground_truths.max()}')
            model = HCPMockModel(
                    "id",
                    classes=np.arange(scores_matrix.shape[1]),
                    softmax=False,
                    raw=True
                    )
            cover_ids = np.arange(scores_matrix.shape[0]).reshape(scores_matrix.shape[0], 1)
            model.store_scores(cover_ids, scores_matrix)
            model.fit(cover_ids, y=ground_truth_ids)
            if not hierarchical:
                cp_classifier = MapieClassifier(
                    estimator=model,
                    method=_method,
                    cv='prefit'
                )
            elif _method == 'multi_label': # hierarchical/multilabel = True
                cp_classifier = MapieMultiLabelClassifier(
                    estimator=model,
                    method='crc',
                    metric_control="recall"
                )
            else: # hierarchical/multilabel = True
                cp_classifier = HierarchicalMapieClassifier(
                    estimator=model,
                    method=_method,
                    classes=np.arange(scores_matrix.shape[1]),  
                    cv='prefit',
                    costs_per_node=costs_per_node,
                    clip=(0.0, 0.999999),
                    cp_method=cp_method,
                )
            cp_classifier = cp_classifier.fit(
                cover_ids,
                ground_truth_ids
            )
            # propagate true labels
            calibrators[cover] = (model, cp_classifier)
            # calibrate: determine qhat
    # return dict[wn.Synset, qhat]
    return calibrators

def filter_calibrators(calibrators, lca, class_nodes):
    filtered_calis = {}
    lca_calis = {}
    candidate_covers = list(no_cover.filter_candidate_covers(frozenset(calibrators.keys()), lca))
    for calibrator in calibrators:
        if calibrator in candidate_covers and calibrator != class_nodes:
            filtered_calis[calibrator] = calibrators[calibrator]
        if lca in calibrator:
            lca_calis[calibrator] = calibrators[calibrator]
    return filtered_calis, lca_calis

def infer_mm(
    node_scores: dict[wn.Synset, float],
    calibrators,
    class_nodes,
    alpha,
):
    scores_per_node = _scores_per_node(node_scores)
    prediction_sets_idxs = {}
    prediction_sets_nodes = {}
    # alpha_adj = utils.bonferonni_correct(alpha, len(calibrators), len(node_scores))
    for cover in sorted(calibrators, key=len, reverse=True):
        model, cp_classifier = calibrators[cover]
        
        if len(cover) == 1:
            #    prediction_set_idx = [[0,],] * len(node_scores)
            prediction_set_nodes = [[next(iter(cover))],] * len(node_scores)
        else:
            scores, _ = cover_scores(cover, scores_per_node)
            class_node_id_cover = {synset: idx for idx, synset in enumerate(cover)}
            id_class_node_cover = {id: node for node, id in class_node_id_cover.items()}
            if (len(scores) == 0):
                continue
            scores_matrix = format_for_cover(scores, class_node_id_cover)
            start = model.scores.idx.max() + 1
            stop = scores_matrix.shape[0] + start
            cover_ids = np.arange(start, stop).reshape(scores_matrix.shape[0], 1)
            model.store_scores(cover_ids, scores_matrix)

            result = cp_classifier.predict(cover_ids, alpha=alpha)
            if len(result) == 2:
                preds, prediction_set_mask = result
            elif len(result) == 4:
                _, preds, prediction_set_mask, _ = result
            prediction_set_mask = prediction_set_mask[:, :, 0]

            prediction_set_nodes = [np.array(list(id_class_node_cover.values()))[row_mask].tolist() for row_mask in prediction_set_mask]
        # prediction_sets_idxs[cover] = prediction_set_idx
        prediction_sets_nodes[cover] = prediction_set_nodes
    return prediction_sets_nodes

def get_cover_ps(
    cover_calibrator,
    cover_model,
    cover,
    scores_per_node,
    alpha
):
    scores, _ = cover_scores(cover, scores_per_node)
    class_node_id_cover = {synset: idx for idx, synset in enumerate(cover)}
    id_class_node_cover = {id: node for node, id in class_node_id_cover.items()}
    if (len(scores) == 0):
        raise ValueError('no scores')
    scores_matrix = format_for_cover(scores, class_node_id_cover)
    start = cover_model.scores.idx.max() + 1
    stop = scores_matrix.shape[0] + start
    cover_ids = np.arange(start, stop).reshape(scores_matrix.shape[0], 1)
    cover_model.store_scores(cover_ids, scores_matrix)

    result = cover_calibrator.predict(cover_ids, alpha=alpha)
    if len(result) == 2:
        preds, prediction_set_mask = result
    elif len(result) == 4:
        _, preds, prediction_set_mask, _ = result
    prediction_set_mask = prediction_set_mask[:, :, 0]

    prediction_set_nodes = [np.array(list(id_class_node_cover.values()))[row_mask].tolist() for row_mask in prediction_set_mask]
    return prediction_set_nodes

@utils.disk_cache('/tmp/')
def infer_mm_dynamic(
    node_scores: dict[wn.Synset, float],
    calibrators,
    class_nodes,
    alpha,
    n_calib,
):
    scores_per_node = _scores_per_node(node_scores)
    prediction_sets_idxs = {}
    prediction_sets = {}
    alpha_adj = []
    # loop over all instances in node_scores
    print("inferring...")
    n_no_candidates = 0
    for i, instance_scores in tqdm.tqdm(enumerate(node_scores), total=len(node_scores)):
        prediction_sets[i] = {}
        scores_per_node_i = {node: [scores[i],] for node, scores in scores_per_node.items()}
        
        # determine the coverage for leaves and its lca
        leaves_ps = get_cover_ps(
            calibrators[class_nodes][1],
            calibrators[class_nodes][0],
            class_nodes,
            scores_per_node_i,
            alpha
        )[0]
        prediction_sets[i][class_nodes] = leaves_ps
        if len(leaves_ps) > 0:
            lca_leaves_ps = utils.get_lca(frozenset(leaves_ps))
        else:
            lca_leaves_ps = frozenset({})
        candidate_calibrators, lca_calibrators = filter_calibrators(
            calibrators,
            lca_leaves_ps,
            class_nodes
        )
        lca_ps = frozenset({lca_leaves_ps,})
        for lca_cali in lca_calibrators:
            prediction_sets[i][lca_cali] = lca_ps
        if len(candidate_calibrators) == 0:
            # print('No candidate calibrators')
            n_no_candidates += 1
        alpha_adj.append(utils.bonferonni_correct(alpha, len(calibrators), n_calib)) # root cover
        # filter all covers containing an ancestor to lca > candidate_covers per instance
        
        # loop over all candidate_covers for this instance
        # infer using dynamically adjusted alpha by the length of candidates
        for cover in sorted(candidate_calibrators, key=len, reverse=True):
            model, cp_classifier = calibrators[cover]
            
            if len(cover) == 1:
                #    prediction_set_idx = [[0,],] * len(node_scores)
                prediction_set_nodes = [[next(iter(cover))],]
            else:
                prediction_set_nodes = get_cover_ps(
                    cp_classifier,
                    model,
                    cover,
                    scores_per_node_i,
                    alpha_adj[i]
                )[0]
            # prediction_sets_idxs[cover] = prediction_set_idx
            prediction_sets[i][cover] = prediction_set_nodes
    alpha_adj = np.array(alpha_adj)
    print(f"alpha_adj: {alpha_adj.mean()}, {alpha_adj.min()}, {alpha_adj.max()}")
    print(f"n_no_candidates: {n_no_candidates}")
    return prediction_sets

def infer(
        node_scores: dict[wn.Synset, float],
        cover_cutoffs: dict[frozenset[wn.Synset], float],
        class_nodes,
        tree=False,
          # map covers to empirical quantiles
        ) -> dict[set[wn.Synset], set[wn.Synset]]:
    scores_per_node = _scores_per_node(node_scores)
    prediction_sets_idxs = {}
    prediction_sets = {}
    print("Running inference...")
    for cover, qhat in tqdm.tqdm(cover_cutoffs.items()):
        if len(cover) == 1:
            #    prediction_set_idx = [[0,],] * len(node_scores)
               prediction_set_nodes = [[next(iter(cover))],] * len(node_scores)
        else:
            if cover == class_nodes:
                pass
            scores, _ = cover_scores(cover, scores_per_node)
            class_node_id_cover = {synset: idx for idx, synset in enumerate(cover)}
            id_class_node_cover = {id: node for node, id in class_node_id_cover.items()}
            if (len(scores) == 0):
                continue
            scores_matrix = format_for_cover(scores, class_node_id_cover)

            prediction_set_mask = scores_matrix <= (1-qhat)

            prediction_set_nodes = [np.array(list(id_class_node_cover.values()))[row_mask].tolist() for row_mask in prediction_set_mask]

        prediction_sets_nodes[cover] = prediction_set_nodes
    return prediction_sets_nodes


def optimize(
        beta: float,
        node_scores: dict[wn.Synset, float],
        cover_cutoffs: dict[frozenset[wn.Synset], float],
        class_nodes: frozenset[wn.Synset],
):
    # obtain the prediction sets for all covers
    ps_per_cover = infer(node_scores, cover_cutoffs, class_nodes)
    ps_per_instances = defaultdict(dict)
    print("Formatting...")
    for cover, prediction_sets in tqdm.tqdm(ps_per_cover.items()):
        for i, prediction_set in enumerate(prediction_sets):
            ps_per_instances[i] |= {cover: prediction_set}
    # now map these back to per-instance prediction sets
    best_covers = []
    prediction_sets = []
    costs = []
    covered_leaves = []
    print('Optimizing...')
    for i, instance in tqdm.tqdm(ps_per_instances.items()):
        i_costs = {}
        i_covered_leaves = {}
        for j, (cover, ps) in enumerate(instance.items()):
            j_cost, covered_leaves = cost(beta, ps, class_nodes, cover)
            i_costs[cover] = j_cost
            i_covered_leaves[cover] = covered_leaves
        # select the cover that optimizes objective in the paper (Eq. 5)
        best_cover = min(i_costs, key=i_costs.get)
        best_covers.append(best_cover)
        prediction_sets.append(instance[best_cover])
        costs.append(i_costs[best_cover])
        covered_leaves.append(i_covered_leaves[cover])
    return best_covers, prediction_sets, costs, covered_leaves

def optimize_mm(
        beta: float,
        node_scores: dict[wn.Synset, float],
        cover_cutoffs: dict[frozenset[wn.Synset], float],
        class_nodes: frozenset[wn.Synset],
        alpha,
        n_calib,
        tree=False,
        dynamic=True,
):
    if dynamic:
        ps_per_instances = infer_mm_dynamic(node_scores, cover_cutoffs, class_nodes, alpha, n_calib)
    else:
        # obtain the prediction sets for all covers
        ps_per_cover = infer_mm(node_scores, cover_cutoffs, class_nodes, alpha)
        ps_per_instances = defaultdict(dict)
        for cover, prediction_sets in ps_per_cover.items():
            for i, prediction_set in enumerate(prediction_sets):
                ps_per_instances[i] |= {cover: prediction_set}
        # now map these back to per-instance prediction sets
    best_covers = []
    prediction_sets = []
    costs = []
    n_covered_leaves = []
    print('optimizing...')
    for i, instance in tqdm.tqdm(ps_per_instances.items()):
        i_costs = {}
        i_covered_leaves = {}
        for j, (cover, ps) in enumerate(instance.items()):
            j_cost, covered_leaves = cost(beta, ps, class_nodes, cover, lambda *x: 0) # 0 costs for empty set
            i_costs[cover] = j_cost
            i_covered_leaves[cover] = covered_leaves
        # select the cover that optimizes objective in the paper (Eq. 5)
        best_cover = min(i_costs, key=i_costs.get)
        best_covers.append(best_cover)
        prediction_sets.append(list(instance[best_cover]))
        costs.append(i_costs[best_cover])
        n_covered_leaves.append(i_covered_leaves[cover])
    return best_covers, prediction_sets, costs, n_covered_leaves

def optimize_mm_betas(
        betas: list[float],
        node_scores: dict[wn.Synset, float],
        cover_cutoffs: dict[frozenset[wn.Synset], float],
        class_nodes: frozenset[wn.Synset],
        alpha,
        n_calib,
        tree=False,
        dynamic=True
):
    # obtain the prediction sets for all covers
    if dynamic:
        ps_per_instances = infer_mm_dynamic(node_scores, cover_cutoffs, class_nodes, alpha, n_calib)
    else:
        # obtain the prediction sets for all covers
        ps_per_cover = infer_mm(node_scores, cover_cutoffs, class_nodes, alpha)
        ps_per_instances = defaultdict(dict)
        for cover, prediction_sets in ps_per_cover.items():
            for i, prediction_set in enumerate(prediction_sets):
                ps_per_instances[i] |= {cover: prediction_set}
        # now map these back to per-instance prediction sets

    # now map these back to per-instance prediction sets
    best_covers = {}
    prediction_sets = {}
    costs = {}
    print('optimizing...')
    for beta in tqdm.tqdm(betas):
        print('beta', beta)
        best_covers[beta] = []
        prediction_sets[beta] = []
        costs[beta] = []
        for i, instance in tqdm.tqdm(ps_per_instances.items()):
            i_costs = {}
            for j, (cover, ps) in enumerate(instance.items()):
                i_costs[cover] = cost(beta, ps, class_nodes, cover)
            # select the cover that optimizes objective in the paper (Eq. 5)
            min_value = min(i_costs.values())
            # Get the last key with the min value by reversing the items
            best_cover = next(k for k, v in reversed(list(i_costs.items())) if v == min_value)
            best_covers[beta].append(best_cover)
            prediction_sets[beta].append(list(instance[best_cover]))
            costs[beta].append(i_costs[best_cover])
    return best_covers, prediction_sets, costs

def optimize_mm_alphas(
        beta: float,
        node_scores: dict[wn.Synset, float],
        cover_cutoffs: dict[frozenset[wn.Synset], float],
        class_nodes: frozenset[wn.Synset],
        alphas,
        tree=False,
        dynamic=True
):
    # obtain the prediction sets for all covers
    if dynamic:
        ps_per_instances = infer_mm_dynamic_alphas(node_scores, cover_cutoffs, class_nodes, alphas)
    else:
        # obtain the prediction sets for all covers
        ps_per_cover = infer_mm(node_scores, cover_cutoffs, class_nodes, alphas)
        ps_per_instances = defaultdict(dict)
        for cover, prediction_sets in ps_per_cover.items():
            for i, prediction_set in enumerate(prediction_sets):
                ps_per_instances[i] |= {cover: prediction_set}
        # now map these back to per-instance prediction sets

    # now map these back to per-instance prediction sets
    best_covers = {}
    prediction_sets = {}
    costs = {}
    print('optimizing...')
    for alpha in tqdm.tqdm(alphas):
        best_covers[alpha] = []
        prediction_sets[alpha] = []
        costs[alpha] = []
        for i, instance in tqdm.tqdm(ps_per_instances.items()):
            i_costs = {}
            for j, (cover, ps) in enumerate(instance.items()):
                i_costs[cover] = cost(beta, ps, class_nodes, cover)
            # select the cover that optimizes objective in the paper (Eq. 5)
            min_value = min(i_costs.values())
            # Get the last key with the min value by reversing the items
            best_cover = next(k for k, v in reversed(list(i_costs.items())) if v == min_value)
            best_covers[alpha].append(best_cover)
            prediction_sets[v].append(list(instance[best_cover]))
            costs[alpha].append(i_costs[best_cover])
    return best_covers, prediction_sets, costs