import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB


class ModelClassification:
    def __init__(self, models,pipe):
        self.models = models
        self.best_model_name = None
        self.pipe=pipe
        self.best_model = None
        self.results = {}

    def evaluate_models(self, X_train, X_test, y_train, y_test):
        best_accuracy = 0

        for model_name, (model, params) in self.models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.results[model_name] = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model_name = model_name

        return self.best_model_name, self.results

    def perform_grid_search(self, X_train, y_train):
        if not self.best_model_name:
            raise ValueError("No best model identified. Please run evaluate_models first.")

        best_model_type, param_grid = self.models[self.best_model_name]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', best_model_type)
        ])

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.best_score_

    def evaluate_best_model(self, X_test, y_test):
        if not self.best_model:
            raise ValueError("No best model identified. Please run perform_grid_search first.")

        y_pred = self.best_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)

        return final_accuracy

    def save_best_model(self, file_path):
        if not self.best_model:
            raise ValueError("No best model to save. Please run perform_grid_search first.")
        joblib.dump((self.best_model,self.pipe), file_path)
        print(f"Entire model saved to {file_path}")

    def retrain(self,X_train_tra,y_train,prev_model,file_path):
        ensemble_model = VotingClassifier(estimators=[
        ('old', prev_model),
        ('new', self.best_model)
        ], voting='soft')
        ensemble_model.fit(X_train_tra,y_train)
        joblib.dump((ensemble_model,self.pipe), file_path)
        print(f"Entire model saved to {file_path}")