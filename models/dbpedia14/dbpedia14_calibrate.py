from datasets import load_dataset
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import json
import os
import sys
sys.path.append(os.getcwd() + '/..')
from mock_model import MockModel
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score
from sklearn.model_selection import train_test_split
from operator import itemgetter
import sklearn

np.random.seed(42)

# eval_accuracy = .00001
target_alpha = .00001

def index_for_arange(start: float, step: float, value: float):
    """
    Returns at which location/index a known ``value`` for an ``np.arange`` generated with ``start`` and ``step`` resides. 
    """
    return int((value - start) / step)

def cp_stats(
    y_pred_set: ArrayLike,
    y_test: ArrayLike,
    max_set_size: int=None,
)-> float:
    """
    Calculates evaluation statistics for a given matrix of n_inputs x m_classes `y_pred_set`:
    * coverage of the true class
    * the % of inputs with a single class
    * the % of empty sets
    * the average set size
    """
    set_size = y_pred_set.sum(axis=1)
    single = (set_size == 1)
    empty = (set_size == 0)
    ps_size = y_pred_set.sum(axis=1).mean()
    return classification_coverage_score(y_test, y_pred_set), single.mean(), empty.mean(), ps_size.mean()

dbpedia = load_dataset('fancyzhx/dbpedia_14', split='test')
scores = pd.read_csv("models/dbpedia14/dbpedia14_bert.csv")
scores["id"] = scores.index
labels = scores.label.unique()
score_columns = [f"label_{i}" for i in range(len(labels))] + ['id',]
model = MockModel(
    scores[score_columns],
    "id",
    classes=labels,
    softmax=False
)

model = model.fit(
    scores['id'],
    scores['label']
)

X_calib, X_test, y_calib, y_test = train_test_split(
    scores[score_columns],
    scores['label'],
    test_size=0.2,
    stratify=scores['label'],
    random_state=42
)
cp_classifier = MapieClassifier(
    estimator=model,
    method='score',
    cv='prefit'
)
cp_classifier = cp_classifier.fit(
    X_calib['id'],
    y_calib
)


y_pred = model.predict(X_calib)

print('accuracy:', sklearn.metrics.accuracy_score(
    y_calib,
    y_pred
))


start, stop, step = 0.001, 0.01, 0.001
target_alpha = 0.005
alphas = np.arange(start, stop, step)
target_index = index_for_arange(start, step, target_alpha)

_, ps = cp_classifier.predict(X_test['id'], alpha=alphas)

coverage, single, empty, size = cp_stats(ps[:, :, target_index], y_test)
print(f"alpha: {alphas[target_index]:.7g}")
print(f"coverage: {coverage}")
print(f"% single: {single}")
print(f"% empty: {empty}")
print(f"average size: {size}")

dbpedia_labels = [label.lower() for label in ['Company', 'EducationalInstitution', 'Artist', 'Athlete', 'OfficeHolder', 'MeanOfTransportation', 'Building', 'NaturalPlace', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'WrittenWork']]

sets = {}

print(X_test.index)
print(ps[:, :, target_index])
for id, set in zip(X_test.index, ps[:, :, target_index]):
    predicted_classes = np.where(set)[0]
    # TODO FdH: ensure that there is at least one element in the set
    if len(predicted_classes) == 0: continue
    labels = itemgetter(*predicted_classes)(dbpedia_labels)
    if isinstance(labels, tuple):
        labels = list(labels)
    else:
        labels = [labels]
    sets[id] = {
        'label': dbpedia_labels[int(y_test[id])],
        'content': dbpedia[id]['content'],
        'set': labels
    }

result_file = f"data/dbpedia14/dbpedia_results_{alphas[target_index]:.7g}.json"
with open(result_file, 'w+') as f:
    json.dump(sets, f, sort_keys=True, indent=4,)
print(result_file)
