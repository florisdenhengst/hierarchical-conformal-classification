import pandas as pd
import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multioutput import _MultiOutputEstimator
from sklearn.utils.multiclass import unique_labels

class HCPMockModel(BaseEstimator, ClassifierMixin):
    """
    A mock sklearn classifier that returns precomputed predictions stored in a ``lookup_table`` for hierarchical
    conformal prediction with single labels eg a Tree.

    Args:
        * input_column (str): the index column when performing a lookup
        * softmax (bool, optional): whether to apply softmax to the precomputed predictions. Defaults to False
        * raw (bool, optional): whether to normalize results to sum to 1. Defaults to False
    """
    
    def __init__(self, input_column, classes, softmax=False, raw=False):
        # self.lookup_table = None
        self.input_column = input_column
        self.softmax = softmax
        self.raw = raw
        self.classes_ = np.array(classes)
        self.scores = None
        self.proba_columns = None
        self.cv = "prefit"
    
    def store_scores(self, idxs, scores):
        column_names = [f"Col_{i}" for i in range(scores.shape[1])]
        if self.scores is None:
            self.scores = pd.DataFrame(scores, columns=column_names)
            self.scores['idx'] = idxs
            self.proba_columns = column_names
        else:
            # merge to dataframe
            new_df = pd.DataFrame(scores, columns=column_names)
            new_df['idx'] = idxs
            # Append the new matrix
            self.scores = pd.concat([self.scores, new_df], ignore_index=True)

    def fit_transform(self, X):
        return self.predict_proba(X)
    
    def fit(self, X, y):
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)
        if self.classes_ is None:
            # Store the classes seen during fit
            self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        return self
    
    def predict(self, X, *args, **kwargs):
        #index = np.argmax(self.predict_proba(X), axis=1)
        return self.predict_proba(X)

    def predict_proba(self, X):
        scores = self.scores[self.scores.idx.isin(X.ravel())][self.proba_columns]
        if self.softmax:
            scores = scipy.special.softmax(scores, axis=1)
        elif not self.raw:
            # normalize outputs to sum to 1 by scaling only
            scores = scores.divide(scores.sum(axis=1),axis=0).to_numpy()
        else:
            scores = scores.to_numpy()
        return scores
    
    def score(self, X, y):
        raise NotImplementedError()


# class HCPMultiLabelMockModel(BaseEstimator, ClassifierMixin, _MultiOutputEstimator):
#     """
#     A mock sklearn classifier that returns precomputed predictions stored in a ``lookup_table`` for hierarchical
#     conformal prediction with multiple labels, e.g. a DAG.

#     Args:
#         * input_column (str): the index column when performing a lookup
#         * softmax (bool, optional): whether to apply softmax to the precomputed predictions. Defaults to False
#         * raw (bool, optional): whether to normalize results to sum to 1. Defaults to False
#     """
    
#     def __init__(self, input_column, classes, softmax=False, raw=False):
#         # self.lookup_table = None
#         self.input_column = input_column
#         self.softmax = softmax
#         self.raw = raw
#         self.classes_ = np.array(classes)
#         self.scores = None
#         self.proba_columns = None
    
#     def store_scores(self, idxs, scores):
#         column_names = [f"Col_{i}" for i in range(scores.shape[1])]
#         if self.scores is None:
#             self.scores = pd.DataFrame(scores, columns=column_names)
#             self.scores['idx'] = idxs
#             self.proba_columns = column_names
#         else:
#             # merge to dataframe
#             new_df = pd.DataFrame(scores, columns=column_names)
#             new_df['idx'] = idxs
#             # Append the new matrix
#             self.scores = pd.concat([self.scores, new_df], ignore_index=True)

#     def fit_transform(self, X):
#         return self.predict_proba(X)
    
#     def fit(self, X, y):
#         # Check that X and y have correct shape
#         # X, y = check_X_y(X, y)
#         if self.classes_ is None:
#             # Store the classes seen during fit
#             self.classes_ = unique_labels(y)

#         self.X_ = X
#         self.y_ = y
#         return self
    
#     def predict(self, X):
#         index = np.argmax(self.predict_proba(X), axis=1)
#         return self.classes_[index]

#     def predict_proba(self, X):
#         scores = self.scores[self.scores.idx.isin(X.ravel())][self.proba_columns]
#         if self.softmax:
#             scores = scipy.special.softmax(scores, axis=1)
#         elif not self.raw:
#             # normalize outputs to sum to 1 by scaling only
#             scores = scores.divide(scores.sum(axis=1),axis=0).to_numpy()
#         else:
#             scores = scores.to_numpy()
#         return scores
    
#     def score(self, X, y):
#         raise NotImplementedError()