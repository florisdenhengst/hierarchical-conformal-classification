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

np.random.seed(42)

eval_accuracy = 0.60
id_to_label = {
    1: 'pop',
    2: 'metal',
    3: 'disco',
    4: 'blues',
    5: 'reggae',
    6: 'classical',
    7: 'rock',
    8: 'hiphop',
    9: 'country',
    10: 'jazz'
}
gtzan_labels = list(id_to_label.values())
label_to_synset = {
    "blues": "gtzan-en-blues-synset-0-n",
    "classical": "gtzan-en-classical-synset-0-n",
    "country": "gtzan-en-country-synset-0-n",
    "disco": "gtzan-en-disco-synset-0-n",
    "hiphop": "gtzan-en-hiphop-synset-0-n",
    "jazz": "gtzan-en-jazz-synset-0-n",
    "metal": "gtzan-en-metal-synset-0-n",
    "pop": "gtzan-en-pop-synset-0-n",
    "reggae": "gtzan-en-reggae-synset-0-n",
    "rock": "gtzan-en-rock-synset-0-n"
}

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

scores = pd.read_csv("models/gtzan/gtzan_scores.csv")
# scores["id"] = scores.index
labels = [1 + l for l in id_to_label.keys()]
score_columns = [f"label_{i}" for i in range(len(labels))] + ['id',]
model = MockModel(
    scores[score_columns],
    "id",
    # classes=labels,
    softmax=False
)

model = model.fit(
    scores['idx'],
    scores['label']
)

X_calib, X_test, y_calib, y_test = train_test_split(
    scores[score_columns],
    scores['label'],
    test_size=0.8,
    stratify=scores['label'],
    random_state=42,
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

start, stop, step = 0.01, 0.3, 0.01
alphas = np.arange(start, stop, step)
target_alpha = 0.1
target_index = index_for_arange(start, step, target_alpha)

_, ps = cp_classifier.predict(X_test['id'], alpha=alphas)

coverage, single, empty, size = cp_stats(ps[:, :, target_index], y_test)
print(f"alpha: {alphas[target_index]:.7g}")
print(f"coverage: {coverage}")
print(f"% single: {single}")
print(f"% empty: {empty}")
print(f"average size: {size}")

sets = {}

for idx, set in zip(X_test.index, ps[:, :, target_index]):
    predicted_classes = np.where(set)[0]
    if len(predicted_classes) == 0: continue
    labels = itemgetter(*predicted_classes)(gtzan_labels)
    if isinstance(labels, tuple):
        labels = list(labels)
    else:
        labels = [labels]
    sets[idx] = {
        'label': gtzan_labels[int(y_test[idx])],
        'content': scores.iloc[idx]['idx'],
        'set': [label_to_synset[l] for l in labels]
    }

result_file = f"data/gtzan/gtzan_results_{alphas[target_index]:.7g}.json"
with open(result_file, 'w') as f:
    json.dump(sets, f, sort_keys=True, indent=4,)
