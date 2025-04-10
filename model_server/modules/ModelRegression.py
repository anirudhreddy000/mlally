from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class ModelSelector:
    def __init__(self):
        self.pipelines = {
            'xgb': Pipeline([('xgb', XGBRegressor(objective='reg:squarederror'))]),
            'gbr': Pipeline([('gbr', GradientBoostingRegressor())]),
            'rf': Pipeline([('rf', RandomForestRegressor())]),
            'enet': Pipeline([('enet', ElasticNet())]),
            'svr': Pipeline([('svr', SVR())])
        }
        
        self.param_grids = {
            'xgb': {
                'xgb__n_estimators': [50, 100, 200],
                'xgb__max_depth': [3, 5, 7, 9],
                'xgb__learning_rate': [0.01, 0.05, 0.1, 0.3],
                'xgb__subsample': [0.5, 0.7, 0.9, 1.0],
                'xgb__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
            },
            
            'gbr': {
                'gbr__n_estimators': [50, 100, 200],
                'gbr__max_depth': [3, 5, 7, 9],
                'gbr__learning_rate': [0.01, 0.05, 0.1, 0.3],
                'gbr__subsample': [0.5, 0.7, 0.9, 1.0]
            },
            'rf': {
                'rf__n_estimators': [50, 100, 200],
                'rf__max_depth': [None, 10, 20, 30],
                'rf__min_samples_split': [2, 5, 10],
                'rf__min_samples_leaf': [1, 2, 4],
                'rf__bootstrap': [True, False]
            },
            'enet': {
                'enet__alpha': [0.01, 0.1, 1, 10, 100],
                'enet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                'enet__max_iter': [1000, 2000, 5000]
            },
            'svr': {
                'svr__C': [0.1, 1, 10, 100],
                'svr__kernel': ['linear', 'rbf', 'poly'],
                'svr__gamma': ['scale', 'auto']
            }
        }

        self.best_estimators = {}
        self.best_model = None
        self.best_score = float('-inf')
        self.best_params = None
        self.best_name = None

    def fit(self, X_train, y_train):
        for name, pipeline in self.pipelines.items():
            print(f"Running GridSearchCV for {name}...")
            param_grid = self.param_grids[name]
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=2, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.best_estimators[name] = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best cross-validation R² score for {name}: {grid_search.best_score_}")

            if grid_search.best_score_ > self.best_score:
                self.best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
                self.best_name = name

    def score(self, X_test, y_test):
        if self.best_model is not None:
            test_r2 = self.best_model.score(X_test, y_test)
            print(f"\nBest model: {self.best_name}")
            print(f"Best cross-validation R² score: {self.best_score}")
            print(f"Best parameters: {self.best_params}")
            print(f"Test R² score: {test_r2}")
            return test_r2
        else:
            print("No model has been fitted yet.")
            return None


