from sklearn.decomposition import TruncatedSVD
class ConditionalSVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.svd = None

    def fit(self, X, y=None):
        n_features = X.shape[1]
        if n_features > 1:
            self.svd = TruncatedSVD(n_components=min(self.n_components, n_features))
            self.svd.fit(X)
        return self

    def transform(self, X):
        if self.svd:
            return self.svd.transform(X)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)