from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
models = {
    'SVC': (SVC(), {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}),
    'RandomForest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 10, 20]}),
    'AdaBoost': (AdaBoostClassifier(algorithm='SAMME'), {'classifier__n_estimators': [50, 100, 200], 'classifier__learning_rate': [0.01, 0.1, 1]}),
    'LogisticRegression': (LogisticRegression(max_iter=1000), {'classifier__C': [0.1, 1, 10], 'classifier__solver': ['lbfgs', 'liblinear']}),
    'KNN': (KNeighborsClassifier(), {'classifier__n_neighbors': [3, 5, 7, 9], 'classifier__weights': ['uniform', 'distance']}),
    'DecisionTree': (DecisionTreeClassifier(), {'classifier__max_depth': [None, 10, 20, 30], 'classifier__min_samples_split': [2, 10, 20]}),
    'GradientBoosting': (GradientBoostingClassifier(), {'classifier__n_estimators': [50, 100, 200], 'classifier__learning_rate': [0.01, 0.1, 1], 'classifier__max_depth': [3, 5, 7]}),
    'NaiveBayes': (GaussianNB(), {}),
    }