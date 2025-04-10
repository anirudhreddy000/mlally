import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.porter = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.apply(self._preprocess_text)
        elif isinstance(X, list):
            return [self._preprocess_text(text) for text in X]
        else:
            raise ValueError("Unsupported input type. Must be pandas DataFrame or list.")

    def _preprocess_text(self, text):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in self.stop_words and word not in string.punctuation]
        tokens = [self.porter.stem(word) for word in tokens]
        return ' '.join(tokens)