import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, normal_columns, skewed_columns, z_threshold=3):
        self.normal_columns = normal_columns
        self.skewed_columns = skewed_columns
        self.z_threshold = z_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, np.ndarray):
            df = pd.DataFrame(X)
        else:
            raise TypeError("Input must be a DataFrame or NumPy array")


        for col in self.normal_columns:
            if col in df.columns:
                z_scores = np.abs(zscore(df[col].dropna()))
                df = df[(z_scores < self.z_threshold) | df[col].isna()]

        for col in self.skewed_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                filter = (df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR)) | df[col].isna()
                df = df[filter]

        return df.values