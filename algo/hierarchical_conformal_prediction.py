import copy
import os
from collections import defaultdict
import numpy as np
import wn
from algo.propagate import propagate_ground_truths
from algo.no_cover import covered_nodes
from mapie.classification import MapieClassifier
from mapie.hierarchical_classification import HierarchicalMapieClassifier
from mapie.multi_label_classification import MapieMultiLabelClassifier
from mapie.multiple_testing import WestfallYoungMaxT
import pprint
import scipy
import tqdm
from joblib import Parallel, delayed
from algo.hcp_mock_model import HCPMockModel
from algo import utils, no_cover
from algo.costs import cost

# Not a function arg: infer_mm_dynamic (which optimize_mm_betas also feeds
# through) is wrapped in disk_cache(), which hashes every argument to build
# its cache key, so N_JOBS must stay out of any cached function's signature
# or identical runs would miss the cache purely because of a
# parallelism-degree change. calibrate_covers_mockmodel and optimize_mm_betas
# below read this directly.
N_JOBS = int(os.environ.get("HCC_N_JOBS", "-1"))

# wn opens a single sqlite3 connection per process with check_same_thread=True
# by default, so any wn.* lookup (e.g. covered_nodes) made from a worker
# thread other than the one that first opened the connection raises
# sqlite3.ProgrammingError. calibrate_covers_mockmodel below parallelizes
# with threads, so this must be enabled before any wn query runs.
wn.config.allow_multithreading = True


def format_for_cover(scores, class_node_id_cover):
    # assign prediction scores for each instance
    # Using Mauricio Sadinle, Jing Lei, and Larry Wasserman.
    # "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.",
    # Journal of the American Statistical Association, 114, 2019.
    
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


def _calibrate_one_cover(
        cover,
        scores_per_node,
        hierarchical_ground_truths,
        leafs,
        tree,
        method,
        loss_approach,
        cp_method,
):
    if len(cover) == 1:
        return cover, (None, None)
    scores, _ = cover_scores(cover, scores_per_node)
    class_node_id_cover = {synset: idx for idx, synset in enumerate(cover)}
    id_class_node_cover = {id: node for node, id in class_node_id_cover.items()}
    costs_per_node = np.array([
        len(covered_nodes(frozenset({id_class_node_cover[idx]}), leafs)) for idx in range(len(id_class_node_cover))
    ])
    if (len(scores) == 0):
        return cover, None
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
    return cover, (model, cp_classifier)

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
    hierarchical_ground_truths = propagate_ground_truths(
        true_labels,
        id_class_node_mapping
    )
    scores_per_node = _scores_per_node(node_scores)
    leafs = utils.all_leaves(wordnet)
    # Each cover is fit independently and only returns freshly-created
    # objects (nothing shared is mutated), so this is embarrassingly
    # parallel. We use threads rather than processes: sending freshly fitted
    # MapieClassifier/HCPMockModel objects back across a process boundary
    # means pickling them, and a pickle round-trip of a fitted MAPIE
    # estimator has been observed to silently drop state it sets dynamically
    # during fit() (AttributeError at predict time later on) - not worth the
    # risk. Threads share memory, so nothing needs to be pickled at all.
    print('calibrating...')
    results = Parallel(n_jobs=N_JOBS, backend="threading", return_as="generator")(
        delayed(_calibrate_one_cover)(
            cover, scores_per_node, hierarchical_ground_truths, leafs, tree, method, loss_approach, cp_method,
        )
        for cover in covers
    )
    calibrators = {} # maps covers to CP models
    for cover, result in tqdm.tqdm(results, total=len(covers)):
        if result is not None:
            calibrators[cover] = result
    # return dict[wn.Synset, qhat]
    return calibrators

def westfall_young_correct(calibrators, alpha):
    """
    Corrects a family of calibrated covers for multiple comparisons with the
    single-step Westfall-Young maxT procedure.

    For every calibration instance ``i`` the non-conformity scores of all
    ``m`` covers are reduced to ``t_i = max(s_{i,1}, ..., s_{i,m})``, and every
    cover predictor is re-calibrated on ``{t_i}``, so that it uses the
    ``ceil((n_c+1)(1-alpha))/n_c`` quantile of the maxima as its threshold.
    Coverage then holds simultaneously for all covers, and thus for the cover
    that is selected per test instance by the cost minimization.

    Args:
        * calibrators: maps covers to ``(model, cp_classifier)`` tuples as
          returned by ``calibrate_covers_mockmodel``. All cp_classifiers must
          be calibrated on the same calibration instances, in the same order.
        * alpha: the nominal (family-wise) error level.

    Returns:
        * a dict with the same keys as ``calibrators`` in which every
          cp_classifier is replaced by its corrected copy. The corrected
          predictors are used with the nominal ``alpha``; ``calibrators``
          itself is not modified. Covers without a cp_classifier (single-node
          covers, which always cover the ground truth) are kept as they are.
        * the effective corrected alpha', i.e. the mean over all covers of the
          fraction of calibration scores that exceed the corrected threshold
          (for CRC predictors: their own mean calibration loss at the
          corrected threshold). This is reported for information only.
    """
    corrected = dict(calibrators)
    covers = [cover for cover, (_, cp) in calibrators.items() if cp is not None]
    if len(covers) == 0:
        return corrected, alpha
    correction = WestfallYoungMaxT().fit([calibrators[cover][1] for cover in covers])
    for cover in covers:
        model, cp_classifier = calibrators[cover]
        corrected[cover] = (model, correction.correct(cp_classifier))
    corrected_alpha = correction.get_corrected_alpha(alpha)[0]
    if np.all(np.isnan(corrected_alpha)):
        alpha_adj = alpha
    else:
        alpha_adj = float(np.nanmean(corrected_alpha))
    return corrected, alpha_adj

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
    # get_cover_ps is called once per test instance (infer_mm_dynamic), and
    # nothing downstream needs rows from a previous call - predict() below
    # only ever looks up the ids we're about to store. store_scores() just
    # appends forever though, so without this reset, cover_model.scores
    # grows across the whole test set and every store_scores() call becomes
    # an O(current size) copy, turning the per-cover cost O(n_test^2)
    # instead of O(n_test). Resetting first keeps each call O(batch size).
    cover_model.scores = None
    cover_ids = np.arange(scores_matrix.shape[0]).reshape(scores_matrix.shape[0], 1)
    cover_model.store_scores(cover_ids, scores_matrix)

    result = cover_calibrator.predict(cover_ids, alpha=alpha)
    if len(result) == 2:
        preds, prediction_set_mask = result
    elif len(result) == 4:
        _, preds, prediction_set_mask, _ = result
    prediction_set_mask = prediction_set_mask[:, :, 0]

    prediction_set_nodes = [np.array(list(id_class_node_cover.values()))[row_mask].tolist() for row_mask in prediction_set_mask]
    return prediction_set_nodes

def _infer_mm_dynamic_one(
        i,
        scores_per_node,
        calibrators,
        corrected_calibrators,
        class_nodes,
        alpha,
        alpha_adj,
):
    scores_per_node_i = {node: [scores[i],] for node, scores in scores_per_node.items()}
    instance_ps = {}

    # determine the coverage for leaves and its lca
    leaves_ps = get_cover_ps(
        calibrators[class_nodes][1],
        calibrators[class_nodes][0],
        class_nodes,
        scores_per_node_i,
        alpha
    )[0]
    instance_ps[class_nodes] = leaves_ps
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
        instance_ps[lca_cali] = lca_ps
    has_no_candidates = len(candidate_calibrators) == 0
    # filter all covers containing an ancestor to lca > candidate_covers per instance

    # loop over all candidate_covers for this instance
    # infer with the Westfall-Young corrected predictors: these are calibrated
    # on the max non-conformity score over all covers, so they are used with
    # the nominal alpha (alpha_adj is the effective per-cover alpha')
    for cover in sorted(candidate_calibrators, key=len, reverse=True):
        model, cp_classifier = corrected_calibrators[cover]

        if len(cover) == 1:
            #    prediction_set_idx = [[0,],] * len(node_scores)
            prediction_set_nodes = [[next(iter(cover))],]
        else:
            prediction_set_nodes = get_cover_ps(
                cp_classifier,
                model,
                cover,
                scores_per_node_i,
                alpha
            )[0]
        instance_ps[cover] = prediction_set_nodes
    return i, instance_ps, alpha_adj, has_no_candidates

@utils.disk_cache('/tmp/')
def infer_mm_dynamic(
    node_scores: dict[wn.Synset, float],
    calibrators,
    class_nodes,
    alpha,
    n_calib,
):
    scores_per_node = _scores_per_node(node_scores)
    # Measured, not assumed: threading this loop (one task per instance,
    # guarded by a per-cover lock) was tried and benchmarked against plain
    # sequential execution on both gtzan and a dbpedia14 sample. It came out
    # slower both times (e.g. 116s sequential vs 124s threaded for ~14k
    # dbpedia instances) - each instance's work is dominated by small
    # Python/pandas calls that don't release the GIL for long, so thread
    # dispatch + lock overhead outweighs whatever parallelism is possible.
    # Kept sequential; see calibrate_covers_mockmodel and propagate_scores
    # for loops where parallelizing did pay off.
    # Westfall-Young maxT correction over all m covers, computed once from the
    # calibration non-conformity scores (independent of the test instance)
    corrected_calibrators, alpha_adj_wy = westfall_young_correct(calibrators, alpha)
    print("inferring...")
    prediction_sets = {}
    alpha_adj = []
    n_no_candidates = 0
    for i in tqdm.tqdm(range(len(node_scores))):
        _, instance_ps, alpha_adj_i, has_no_candidates = _infer_mm_dynamic_one(
            i, scores_per_node, calibrators, corrected_calibrators, class_nodes, alpha, alpha_adj_wy
        )
        prediction_sets[i] = instance_ps
        alpha_adj.append(alpha_adj_i)
        if has_no_candidates:
            n_no_candidates += 1
    alpha_adj = np.array(alpha_adj)
    print(f"alpha_adj: {alpha_adj.mean()}, {alpha_adj.std()}, {alpha_adj.min()}, {alpha_adj.max()}")
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

def _optimize_mm_betas_one(beta, ps_per_instances, class_nodes):
    best_covers_beta = []
    prediction_sets_beta = []
    costs_beta = []
    for i, instance in ps_per_instances.items():
        i_costs = {}
        for j, (cover, ps) in enumerate(instance.items()):
            i_costs[cover] = cost(beta, ps, class_nodes, cover)
        # select the cover that optimizes objective in the paper (Eq. 5)
        min_value = min(i_costs.values())
        # Get the last key with the min value by reversing the items
        best_cover = next(k for k, v in reversed(list(i_costs.items())) if v == min_value)
        best_covers_beta.append(best_cover)
        prediction_sets_beta.append(list(instance[best_cover]))
        costs_beta.append(i_costs[best_cover])
    return beta, best_covers_beta, prediction_sets_beta, costs_beta

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

    # Each beta redoes a full, independent pass over ps_per_instances, which
    # by this point holds only plain data (covers/prediction sets - the
    # fitted calibrators were already consumed by infer_mm_dynamic above),
    # so processes are safe here. cost() is pure-Python/GIL-bound, so
    # threads would buy little; processes give real parallelism at the cost
    # of pickling ps_per_instances once per beta (fine for the typical
    # beta-sweep sizes this is used for).
    best_covers = {}
    prediction_sets = {}
    costs = {}
    print('optimizing...')
    results = Parallel(n_jobs=N_JOBS, backend="loky", return_as="generator")(
        delayed(_optimize_mm_betas_one)(beta, ps_per_instances, class_nodes) for beta in betas
    )
    for beta, best_covers_beta, prediction_sets_beta, costs_beta in tqdm.tqdm(results, total=len(betas)):
        best_covers[beta] = best_covers_beta
        prediction_sets[beta] = prediction_sets_beta
        costs[beta] = costs_beta
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