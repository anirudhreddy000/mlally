from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Literal
from io import BytesIO
import pandas as pd
import numpy as np
import os
import httpx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.OutlierRemoval import OutlierRemover
from modules.CategoricalTransformer import ThresholdCategoricalTransformer
from modules.Tostring import convert_to_string
from modules.TextPreProcess import TextPreprocessor
from modules.ConditionalSVD import ConditionalSVD
from modules.Is_text import is_text_column
from modules.Normality import test_normality, compute_skewness, summarize_results
from modules.ModelClassification import ModelClassification
from modules.ModelRegression import ModelSelector
from modules.model_selection.ClassificationModels import models

app = FastAPI()

@app.post("/upload_csv")
async def upload_csv_file(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: Literal["regression", "classification"] = Form(...)
):
    csv_bytes = await file.read()
    try:
        df = pd.read_csv(BytesIO(csv_bytes))
    except Exception as e:
        return {"error": "Failed to read CSV file. Error: " + str(e)}

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if y_train.dtype == 'object' or len(np.unique(y_train)) <= 2:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(include=[object]).columns
    normality_results = test_normality(X_train, numerical_cols)
    skewness_results = compute_skewness(X_train, numerical_cols)
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
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    text_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('to_list', FunctionTransformer(convert_to_string, validate=False)),
        ('text_clean', TextPreprocessor()),
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
        # Add ConditionalSVD or MinMaxScaler if needed
    ])

    transformed_train = final_pipeline.fit_transform(X_train)
    transformed_test = final_pipeline.transform(X_test)

    if issparse(transformed_train):
        transformed_train = transformed_train.toarray()
        transformed_test = transformed_test.toarray()

    X_train_tra = pd.DataFrame(transformed_train)
    X_test_tra = pd.DataFrame(transformed_test)

    os.makedirs("pkl_buffer_server", exist_ok=True)
    model_filename = f"pkl_buffer_server/model_{file.filename.split('.')[0]}.pkl"

    if task_type == "regression":
        selector = ModelSelector(final_pipeline)
        selector.fit(X_train_tra, y_train)
        selector.score(X_test_tra, y_test)
        selector.save_best_model(model_filename)
    else:
        model_classifier = ModelClassification(models, final_pipeline)
        model_classifier.evaluate_models(X_train_tra, X_test_tra, y_train, y_test)
        best_params, best_score = model_classifier.perform_grid_search(X_train_tra, y_train)
        final_accuracy = model_classifier.evaluate_best_model(X_test_tra, y_test)
        model_classifier.save_best_model(model_filename)

    import httpx

    try:
        with open(model_filename, 'rb') as f:
            files = {"model_file": (os.path.basename(model_filename), f, "application/octet-stream")}
            async with httpx.AsyncClient() as client:
                response = await client.post("http://127.0.0.1:8005/receive_files", files=files)
                response.raise_for_status()
                return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending model: {str(e)}")
