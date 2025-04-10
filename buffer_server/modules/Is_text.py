import numpy as np
import string

def is_text_column(series):
    if series.dtype != 'object':
        return False

    unique_ratio = series.nunique() / len(series)

    sample_values = series.dropna().astype(str)
    if len(sample_values) > 100:
        sample_values = sample_values.sample(100)

    avg_length = sample_values.apply(len).mean()
    contains_space = sample_values.str.contains(' ').mean()
    contains_punctuation = sample_values.apply(lambda x: any(char in string.punctuation for char in x)).mean()
    contains_numeric = sample_values.apply(lambda x: any(char.isdigit() for char in x)).mean()
    avg_word_length = sample_values.apply(lambda x: np.mean([len(word) for word in x.split()])).mean()
    char_variety = sample_values.apply(lambda x: len(set(x))).mean()

    if (unique_ratio > 0.3 or
        (avg_length > 5 and (contains_space > 0.1 or contains_punctuation > 0.1)) or
        (avg_word_length > 3 and char_variety > 10) or
        (contains_numeric < 0.5 and contains_space > 0.1)):
        return True
    return False
