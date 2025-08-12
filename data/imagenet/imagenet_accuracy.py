from datasets import load_dataset
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import json
import os
import sys
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + 'model/gtzan')
from mock_model import MockModel
from sklearn.model_selection import train_test_split
import sklearn
from operator import itemgetter

np.random.seed(42)

scores = pd.read_csv("data/imagenet/predictions_val.csv")
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

y_pred = model.predict(X_calib)

print('accuracy:', sklearn.metrics.accuracy_score(
    y_calib,
    y_pred
))