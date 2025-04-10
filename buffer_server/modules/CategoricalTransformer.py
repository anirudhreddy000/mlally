
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class ThresholdCategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.thresholds = {}
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        for col_index in range(X.shape[1]):
            col_values = X[:, col_index]
            value_counts = pd.Series(col_values).value_counts()
            threshold_index = len(value_counts) // 4
            self.thresholds[col_index] = value_counts.index[threshold_index]
            self.frequent_categories_[col_index] = value_counts[value_counts.index >= self.thresholds[col_index]].index
        return self

    def transform(self, X):
        X_transformed = []
        for col_index in range(X.shape[1]):
            col_values = X[:, col_index]
            transformed_col = pd.Series(col_values).apply(lambda x: x if x in self.frequent_categories_[col_index] else 'Other')
            X_transformed.append(transformed_col.values)
        return np.column_stack(X_transformed)
