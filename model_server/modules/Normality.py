from scipy.stats import shapiro, skew

def test_normality(data, columns):
    normality_results = {}
    for col in columns:
        stat, p = shapiro(data[col].dropna())
        normality_results[col] = p
    return normality_results

def compute_skewness(data, columns):
    skewness_results = {}
    for col in columns:
        skewness = skew(data[col].dropna())
        skewness_results[col] = skewness
    return skewness_results

def summarize_results(normality_results, skewness_results, alpha=0.05):
    normal_columns = []
    skewed_columns = []

    for col in normality_results.keys():
        p_value = normality_results[col]
        skewness = skewness_results[col]

        if p_value >= alpha:
            normal_columns.append(col)
        else:
            skewed_columns.append(col)

    return normal_columns, skewed_columns
