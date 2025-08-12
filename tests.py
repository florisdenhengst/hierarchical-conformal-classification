import algo
from algo.hierarchical_conformal_prediction import calibrate_covers, calibrate_covers_mockmodel, infer, infer_mm, optimize
from algo.no_cover import get_covers
from algo.propagate import propagate_scores
from algo.utils import get_wn_name
from collections import Counter
from cv_ps_metrics import get_nb_leaf_node
import pprint
import numpy as np
import wn
import random

def setup_seeds():
    np.random.seed(42)
    random.seed(42)

def setup_lexicons():
    LEXICONS = [
        ('gtzan-en', 'models/gtzan/gtzan_lexicon.xml'),
        ('dbpedia14-en', 'models/dbpedia14/dbpedia14_lexicon.xml'),
        ('hcptest-en', 'models/test/test_lexicon.xml'),
        ('omw-subset-en', 'models/imagenet/lexicon.xml')
    ]
    for lexicon, file in LEXICONS:
        try:
            wn.remove(lexicon)
        except wn.Error:  # the lexicon was never added
            pass
        wn.add(file)

def _setup_scores_propagate(wordnet: wn.Wordnet, k_classes: int, n_samples: int=10, probs=True):
    """
    Helper function to create probabilities to test the propagate functions.
    """
    scores = np.random.rand(n_samples, k_classes)
    if probs:
        # normalize to probabilities
        scores = scores / scores.sum(axis=1, keepdims=True)
    return scores

def test_nb_leaf_node():
    """
    Test code for determining the # of covered leaf nodes in some set.
    """
    # refresh the installed wn lexicons    
    INPUT_OUTPUTS_STR = [
        # gtzan dataset examples
        ## single leaf node
        (["gtzan-en-pop-synset-0-n"],
        1),
        ## multiple leaf nodes
        (["gtzan-en-hiphop-synset-0-n",
        "gtzan-en-pop-synset-0-n"],
        2), 
        ## single parent node with two children
        (["gtzan-en-pop_hiphop-synset-0-n"],
        2), 
        ## root node
        (["gtzan-en-entity-synset-0-n"],
        10), 
        # dbpedia14 examples
        ## example with a single leaf node
        (["dbpedia14-en-artist-synset-0-n"],
        1),
        ## example with multiple leaf nodes
        (["dbpedia14-en-artist-synset-0-n",
        "dbpedia14-en-naturalplace-synset-0-n"],
        2),
        ## example with single parent node having one child
        (["dbpedia14-en-architecturalstructure-synset-0-n"], # only has building as child
        1),
        ## example with single parent node having two children
        (["dbpedia14-en-organisation-synset-0-n"], # has company and educiationalinstitution as children
        2),
        ## example with root input
        (["dbpedia14-en-entity-synset-0-n"],
        14), #

        # imagenet examples
        ## example with only leaf node inputs
        (["omw-subset-en-02102177-n"], # Welsh springer spaniel
        1),
        ## example with one parent input
        (["omw-subset-en-02101861-n"], # springer spaniel
        2),
        ## example with root
        (["omw-subset-en-00001740-n"], # entity
        64958), # imagenet subset has 1K classes but 64958 leaf nodes
    ]
    INPUT_OUTPUTS = [(list(map(wn.synset, input)), n_output) for input, n_output in INPUT_OUTPUTS_STR]
    failures = []
    for input, expected_output in INPUT_OUTPUTS:
        actual_output = get_nb_leaf_node(input)
        if expected_output != actual_output:
            failures.append([
                input,
                expected_output,
                actual_output]
            )
    if len(failures) > 0:
        print("====test_nb_leaf_node====")
        pprint.pprint(failures)
    else:
        print(".", end="")

def test_no_covers_unique():
    """
    Tests that the output of no_covers are all unique.
    """
    INPUTS = [
        (wn.Wordnet("dbpedia14-en"),),
        (wn.Wordnet("gtzan-en"),),
        (wn.Wordnet("hcptest-en"),),
        (wn.Wordnet("omw-subset-en"),), #takes too long
    ]
    failures = []
    for input in INPUTS:
        output = get_covers(*input)
        c = Counter(output)
        if max(c.values()) > 1:
            failures.append(input)
    if len(failures) > 0:
        print("====test_no_covers_unique====")
        pprint.pprint(failures)
    else:
        print(".", end="")

def test_no_covers_all_classnodes():
    """
    Tests that the output of no_covers contain all class nodes:
    """
    INPUTS = [
        (wn.Wordnet("dbpedia14-en"),),
        (wn.Wordnet("gtzan-en"),),
        (wn.Wordnet("hcptest-en"),),
        (wn.Wordnet("omw-subset-en"),),
    ]
    failures = []
    for input in INPUTS:
        covers = get_covers(*input)
        all_synsets = {e for cover in covers for e in cover}
        all_class_nodes = algo.utils.class_nodes(input[0])
        difference = all_class_nodes.difference(all_synsets)
        if len(difference) > 0:
            failures.append((get_wn_name(input[0]), difference, covers))
    if len(failures) > 0:
        print("====test_no_covers_all_classnodes====")
        pprint.pprint(failures)
    else:
        print(".", end="")

def test_propagate_layers_prob_sums():
    """
    Tests that the sums of all scores sum to one for normalized prediction scores.
    """
    INPUTS = [
        (wn.Wordnet("hcptest-en"), 7),
        (wn.Wordnet("dbpedia14-en"), 14),
        (wn.Wordnet("gtzan-en"), 10),
        (wn.Wordnet("omw-subset-en"), 1000),
    ]
    failures = []
    for wordnet, k_classes in INPUTS:
        failed = False
        scores = _setup_scores_propagate(wordnet, k_classes, 100, True)
        class_nodes = algo.utils.class_nodes(wordnet)
        # all_leaves = algo.utils.all_leaves(wordnet)
        class_node_id_mapping = {i: name for i, name in enumerate(sorted(class_nodes))}

        aggregated_probabilities = propagate_scores(scores, wordnet, class_node_id_mapping, return_per_level=True)

        for i, instance in aggregated_probabilities.items():
            for l, scores_map in instance.items():
                scores_map = instance[l]
                scores = scores_map.values()
                summed = sum(scores)
                if summed < 0.98: # approximately to account for rounding issues
                    failures.append((get_wn_name(wordnet), i, l, summed, '>'))
                    failed = True
                if ((l == 0 or l == max(instance.keys()))
                    and not np.isclose(summed, 1.0)): # approximately to account for rounding issues
                    failures.append((get_wn_name(wordnet), i, l, summed))
                    failed = True
        if failed:
            pprint.pprint(aggregated_probabilities)
        else:
            print(".", end="")
    if len(failures) > 0:
        print("====test_propagate_layers_prob_sums====")
        pprint.pprint(failures)

def test_propagate_layers_equal():
    """
    Tests that the predicted scores per layer are equal for score propagation.
    """
    INPUTS = [
        (wn.Wordnet("hcptest-en"), 7),
        (wn.Wordnet("dbpedia14-en"), 14),
        (wn.Wordnet("gtzan-en"), 10),
        (wn.Wordnet("omw-subset-en"), 1000),
    ]
    failures = []
    for wordnet, k_classes in INPUTS:
        failed = False
        scores = _setup_scores_propagate(wordnet, k_classes, 100, True)
        class_nodes = algo.utils.class_nodes(wordnet)
        # all_leaves = algo.utils.all_leaves(wordnet)
        class_node_id_mapping = {i: name for i, name in enumerate(sorted(class_nodes))}

        aggregated_probabilities = propagate_scores(scores, wordnet, class_node_id_mapping, return_per_level=True)

        for i, instance in aggregated_probabilities.items():
            instance_map = {}
            for l, scores_map in instance.items():
                for n, score in scores_map.items():
                    if n in instance_map:
                        if not np.isclose(score, instance_map[n]):
                            failures.add(get_wn_name(wordnet), i, n, score, instance_map[n])
                    else:
                        instance_map[n] = score
        if failed:
            pprint.pprint(aggregated_probabilities)
        else:
            print('.', end="")
    if len(failures) > 0:
        print("===test_propagate_layers_equal===")
        pprint.pprint(failures)
    else:
        print(".", end="")


def test_propagate_layers_all():
    """
    Tests that all synsets get a prediction score for score propagation.
    """
    INPUTS = [
        (wn.Wordnet("hcptest-en"), 7),
        (wn.Wordnet("dbpedia14-en"), 14),
        (wn.Wordnet("gtzan-en"), 10),
        #(wn.Wordnet("omw-subset-en"), 1000), #not relevant for omw-subset-en as it has nodes that are not part of a cover
    ]
    failures = []
    for wordnet, k_classes in INPUTS:
        failed = False
        scores = _setup_scores_propagate(wordnet, k_classes, 1, True)
        class_nodes = algo.utils.class_nodes(wordnet)
        # all_leaves = algo.utils.all_leaves(wordnet)
        class_node_id_mapping = {i: name for i, name in enumerate(sorted(class_nodes))}

        aggregated_probabilities = propagate_scores(scores, wordnet, class_node_id_mapping)

        for i, instance in aggregated_probabilities.items():
            if len(instance) != len(wordnet.synsets()):
                failures.append(
                    (get_wn_name(wordnet), i, instance, len(instance), len(wordnet.synsets()))
                )
                failed=True
        if failed:
            pprint.pprint(aggregated_probabilities)
        else:
            print('.', end="")
    if len(failures) > 0:
        print("===test_propagate_layers_all===")
        pprint.pprint(failures)
    else:
        print(".", end="")

def test_calibrate_covers():
    INPUTS = [
        # (wn.Wordnet("hcptest-en"), 7),
        # (wn.Wordnet("dbpedia14-en"), 14),
        # (wn.Wordnet("gtzan-en"), 10),
        (wn.Wordnet("omw-subset-en"), 1000),
    ]
    n = 5
    alpha = 0.5
    for wordnet, k_classes in INPUTS:
        failures = []
        scores = _setup_scores_propagate(wordnet, k_classes, 2, True)
        class_nodes = algo.utils.class_nodes(wordnet)
        class_node_id_mapping = {i: name for i, name in enumerate(sorted(class_nodes))}
        # true_labels = np.arange(k_classes)
        true_labels = np.array([4,3])
        covers = get_covers(wordnet)
        node_scores = propagate_scores(scores, wordnet, class_node_id_mapping)
        qhats = calibrate_covers(
            true_labels,
            alpha,
            covers,
            node_scores,
            class_node_id_mapping
        )

        for cover in covers:
            if cover not in qhats:
                failures.add("missing cover: ", get_wn_name(wordnet), cover)
            else:
                if qhats[cover] > 1.0 or qhats[cover] <= 0.0:
                    failures.add(f"illegal value qhat of {qhats[cover]}:", get_wn_name(wordnet), cover)
        if len(failures) > 0:
            print(f"==={get_wn_name(wordnet)}===")
            pprint.pprint(failures)
        else:
            print(".", end="")


def test_calibrate_covers_mm():
    INPUTS = [
        (wn.Wordnet("hcptest-en"), 7),
        (wn.Wordnet("dbpedia14-en"), 14),
        (wn.Wordnet("gtzan-en"), 10),
        (wn.Wordnet("omw-subset-en"), 1000),
    ]
    alpha = 0.5
    for wordnet, k_classes in INPUTS:
        n = k_classes
        failures = []
        scores = _setup_scores_propagate(wordnet, k_classes, n, True)
        class_nodes = algo.utils.class_nodes(wordnet)
        class_node_id_mapping = {i: name for i, name in enumerate(sorted(class_nodes))}
        true_labels = np.arange(k_classes)
        # true_labels = np.array([4,3])
        covers = get_covers(wordnet)
        node_scores = propagate_scores(scores, wordnet, class_node_id_mapping)
        calibrators = calibrate_covers_mockmodel(
            true_labels,
            alpha,
            covers,
            node_scores,
            class_node_id_mapping
        )

        result = infer_mm(
            node_scores,
            calibrators,
            class_nodes,
            alpha,
        )
        pprint.pprint(result)
        for cover in covers:
            if cover not in result:
                failures.append(("missing cover: ", get_wn_name(wordnet), cover))
            # else:
            #     if qhats[cover] > 1.0 or qhats[cover] <= 0.0:
            #         failures.add(f"illegal value qhat of {qhats[cover]}:", get_wn_name(wordnet), cover)
        if len(failures) > 0:
            print(f"==={get_wn_name(wordnet)}===")
            pprint.pprint(failures)
        else:
            print(".", end="")

def test_infer_covers():
    INPUTS = [
        (wn.Wordnet("hcptest-en"), 7),
        (wn.Wordnet("dbpedia14-en"), 14),
        (wn.Wordnet("gtzan-en"), 10),
        (wn.Wordnet("omw-subset-en"), 1000),
    ]
    n = 100
    qhat = 0.5
    for wordnet, k_classes in INPUTS:
        failures = []
        scores = _setup_scores_propagate(wordnet, k_classes, n, True)
        class_nodes = algo.utils.class_nodes(wordnet)
        class_node_id_mapping = {i: name for i, name in enumerate(sorted(class_nodes))}
        # true_labels = np.arange(k_classes)
        covers = get_covers(wordnet)
        node_scores = propagate_scores(scores, wordnet, class_node_id_mapping)
        qhats = {cover: qhat for cover in covers}
        result = infer(node_scores, qhats, class_nodes)
        # NOTE: we only check whether it runs correclty for now
        print(".", end="")

def test_optimize():
    INPUTS = [
        (wn.Wordnet("hcptest-en"), 7),
        (wn.Wordnet("dbpedia14-en"), 14),
        (wn.Wordnet("gtzan-en"), 10),
        (wn.Wordnet("omw-subset-en"), 1000),
    ]
    n = 1000
    qhat = 0.1
    beta = 1
    for wordnet, k_classes in INPUTS:
        failures = []
        scores = _setup_scores_propagate(wordnet, k_classes, n, True)
        class_nodes = algo.utils.class_nodes(wordnet)
        class_node_id_mapping = {i: name for i, name in enumerate(sorted(class_nodes))}
        # true_labels = np.arange(k_classes)
        covers = get_covers(wordnet)
        node_scores = propagate_scores(scores, wordnet, class_node_id_mapping)
        qhats = {cover: qhat for cover in covers}
        covers, pss, costs = optimize(
            beta,
            node_scores,
            qhats,
            frozenset(class_nodes))
        pprint.pprint(pss)
        pprint.pprint(costs)
        # NOTE: we only check whether it runs correclty for now
        print(".", end="")

setup_seeds()
setup_lexicons()
# test_nb_leaf_node()
# test_no_covers_unique()
# test_no_covers_all_classnodes()
# test_propagate_layers_prob_sums()
# test_propagate_layers_equal()
# test_propagate_layers_all()
test_calibrate_covers_mm()
# test_infer_covers()
# test_optimize()
# print()

# test_propagate_ground_truth()
