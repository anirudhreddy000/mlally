from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import logging
import joblib
import pandas as pd

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# Globals to store model components
model = None
pipe = None
preprocessor = None
model_path = None

@app.post("/receive_files")
async def receive_files(model_file: UploadFile = File(...)):
    save_dir = 'pkl_model_server'
    os.makedirs(save_dir, exist_ok=True)

    global model, pipe, preprocessor, model_path

    model_path = os.path.join(save_dir, os.path.basename(model_file.filename))

    try:
        # Read file content once
        content = await model_file.read()
        with open(model_path, "wb") as f:
            f.write(content)

        # Load the model and pipeline
        loaded_obj = joblib.load(model_path)
        if isinstance(loaded_obj, tuple) and len(loaded_obj) == 2:
            model, pipe = loaded_obj
            preprocessor = pipe.named_steps['preprocessor']
            logging.info(f"Model and pipeline loaded successfully from {model_path}")
        else:
            raise HTTPException(status_code=500, detail="Model format is invalid. Expected (model, pipeline) tuple.")

        return {"message": "Files received successfully"}

    except Exception as e:
        logging.error(f"Error while saving/loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



@app.get("/input_schema")
async def input_schema():
    global preprocessor
    if preprocessor is None:
        raise HTTPException(status_code=400, detail="Model not loaded yet.")

    try:
        numerical_cols = list(preprocessor.transformers_[0][2])
        categorical_cols = list(preprocessor.transformers_[1][2])
        text_cols = list(preprocessor.transformers_[2][2]) if len(preprocessor.transformers_) > 2 else []

        return {
            "numerical": [str(col) for col in numerical_cols],
            "categorical": [str(col) for col in categorical_cols],
            "text": [str(col) for col in text_cols]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch schema: {str(e)}")

@app.post('/web_prediction')
async def web_prediction(data: dict):
    global model, pipe, preprocessor

    if model is None or pipe is None:
        return {'error': 'Model is not loaded. Please upload a model file first.'}

    try:
        # Infer column types from the preprocessor
        numerical_cols = preprocessor.transformers_[0][2]  # First transformer - numerical
        categorical_cols = preprocessor.transformers_[1][2]  # Second transformer - categorical
        text_cols = preprocessor.transformers_[2][2] if len(preprocessor.transformers_) > 2 else []

        required_columns = numerical_cols + categorical_cols + text_cols
        missing_columns = [col for col in required_columns if col not in data]

        if missing_columns:
            return {'error': f'Missing columns: {", ".join(missing_columns)}'}

        input_df = pd.DataFrame([data])
        preprocessed_data = pipe.transform(input_df)
        prediction = model.predict(preprocessed_data)
        prediction_native = prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0]

        return {'prediction': prediction_native}

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return {'error': str(e)}
