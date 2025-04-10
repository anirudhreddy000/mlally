import numpy as np
import pandas as pd
import os
import httpx
from io import BytesIO
from fastapi import FastAPI,UploadFile,File,HTTPException
import joblib
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import *
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from modules.OutlierRemoval import OutlierRemover
from modules.CategoricalTransformer import ThresholdCategoricalTransformer
from modules.Tostring import convert_to_string
from modules.TextPreProcess import TextPreprocessor
from modules.ConditionalSVD import ConditionalSVD
from modules.Is_text import is_text_column
from modules.Normality import test_normality,compute_skewness,summarize_results
from modules.ModelClassification import ModelClassification
from modules.model_selection.ClassificationModels import models
from modules.ModelRegression import ModelSelector


app=FastAPI()

@app.post("/upload_csv")
async def upload_csv_file(file: UploadFile = File(...)):
    csv_bytes=await file.read()
    try:
        df = pd.read_csv(BytesIO(csv_bytes))
    except Exception as e:
        return {"error": "Failed to read CSV file. Error: " + str(e)}


    X=df.drop(columns=['placed'])
    y=df['placed']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    from sklearn.preprocessing import LabelEncoder
    if y_train.dtype == 'object' or len(np.unique(y_train)) <= 2:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        y_is_categorical = True
    else:
        y_is_categorical = False

    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(include=[object]).columns
    columns = X_train.select_dtypes(include=[np.number]).columns
    normality_results = test_normality(X_train, columns)
    skewness_results = compute_skewness(X_train, columns)
    normal_columns, skewed_columns = summarize_results(normality_results, skewness_results)
    text_cols = [col for col in categorical_cols if is_text_column(X_train[col])]
    cat_cols = [col for col in categorical_cols if col not in text_cols]

    numerical_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=2)),
        ('outlier_removal', OutlierRemover(normal_columns, skewed_columns)),
        ('scaler', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('threshold_transformer', ThresholdCategoricalTransformer()),
        ('onehot', OneHotEncoder( handle_unknown='ignore'))
    ])

    text_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('to_list', FunctionTransformer(convert_to_string, validate=False)),
        ('lalala', TextPreprocessor()),
        ('tfidf', TfidfVectorizer())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_pipeline, numerical_cols),
            ('categorical', categorical_pipeline, cat_cols),
        ] + ([('text', text_pipeline, text_cols)] if text_cols else [])
    )

    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        #('svd', ConditionalSVD(n_components=20)),
        #('minmax_scaler', MinMaxScaler())
    ])

    transformed_data_train = final_pipeline.fit_transform(X_train)
    transformed_data_test = final_pipeline.transform(X_test)

    if issparse(transformed_data_train):
        transformed_data_train = transformed_data_train.toarray()
        transformed_data_test = transformed_data_test.toarray()


    X_train_tra = pd.DataFrame(transformed_data_train)
    X_test_tra = pd.DataFrame(transformed_data_test)

  #  selector = ModelSelector(final_pipeline)
  #  selector.fit(X_train_tra, y_train)
  #  selector.score(X_test_tra, y_test)

    model_classifier = ModelClassification(models,final_pipeline)

    best_model_name, results = model_classifier.evaluate_models(X_train_tra, X_test_tra, y_train, y_test)

    print("Model Accuracies:")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")
    best_params, best_score = model_classifier.perform_grid_search(X_train_tra, y_train)

    print("\nBest Model:")
    print(best_model_name)
    print("\nGridSearchCV Results for the Best Model:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")

    final_accuracy = model_classifier.evaluate_best_model(X_test_tra, y_test)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")

    os.makedirs('pkl_buffer_server', exist_ok=True)

    csv_filename = file.filename.split('.')[0]
    model_filename = f"pkl_buffer_server/model_re_{csv_filename}.pkl"

    
    #selector.save_best_model(model_filename)

    model,pipe=joblib.load('pkl_buffer_server\model_placement.pkl')

    model_classifier.retrain(X_train_tra,y_train,model,model_filename)

    async with httpx.AsyncClient() as client:
        try:
            with open(model_filename, 'rb') as model_file:
                files = {
                    'model': (model_filename, model_file, 'application/octet-stream')
                }
                response = await client.post("http://127.0.0.1:8005/receive_files", files=files)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"HTTP error occurred: {exc.response.content.decode()}")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"An error occurred while sending files: {str(exc)}")
       # finally:
           # os._exit(0)
    return {'final_accuracy': final_accuracy}




