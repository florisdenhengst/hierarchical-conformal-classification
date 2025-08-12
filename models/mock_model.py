import pandas as pd
import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

class MockModel(BaseEstimator, ClassifierMixin):
    """
    A mock sklearn classifier that returns precomputed predictions stored in a ``lookup_table``.

    Args:
        * lookup_table (pandas.DataFrame): a pandas dataframe of predictions for each input.
        * input_column (str): the index column when performing a lookup
        * softmax (bool, optional): whether to apply softmax to the precomputed predictions. Defaults to False
        * raw (bool, optional): whether to normalize results to sum to 1. Defaults to False
    """
    
    def __init__(self, lookup_table, input_column, classes=None, softmax=False, raw=False):
        self.lookup_table = lookup_table
        self.input_column = input_column
        self.softmax = softmax
        self.raw = raw
        if classes is None:
            self.classes_ = None
        else:
            self.classes_ = np.array(classes)
    
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
    
    def predict(self, X):
        index = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[index]

    def predict_proba(self, X):
        tmp = pd.merge(X, self.lookup_table, on=self.input_column, how='left')
        tmp = tmp.drop(self.input_column, axis=1)
        if self.softmax:
            tmp = scipy.special.softmax(tmp, axis=1)
        elif not self.raw:
            # normalize outputs to sum to 1 by scaling only
            tmp = tmp.divide(tmp.sum(axis=1),axis=0).to_numpy()
        else:
            tmp = tmp.to_numpy()
        return tmp
    
    def score(self, X, y):
        raise NotImplementedError()