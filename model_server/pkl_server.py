from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load the model and preprocessing pipeline
model, pipe = joblib.load('pkl_model_server\model_diabetes-dataset.pkl')
preprocessor = pipe.named_steps['preprocessor']

# Extract column types from preprocessor
numerical_cols = []
categorical_cols = []
text_cols = []

for name, transformer, columns in preprocessor.transformers:
    if name == 'numerical':
        numerical_cols.extend(columns)
    elif name == 'categorical':
        categorical_cols.extend(columns)
    elif name == 'text':
        text_cols.extend(columns)

# Define the Placement schema dynamically (using Pydantic)
class_dict = {'__annotations__': {}}
for col in numerical_cols:
    class_dict['__annotations__'][col] = float
for col in categorical_cols + text_cols:
    class_dict['__annotations__'][col] = str

Placement = type('Placement', (BaseModel,), class_dict)

@app.post('/predict_placement')
async def predict_placement(placement: Placement):
    try:
        input_dict = placement.dict()
        input_df = pd.DataFrame([input_dict])
        preprocessed_data = pipe.transform(input_df)
        print(preprocessed_data)
        prediction = model.predict(preprocessed_data)
        prediction_native = prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0]
        return {'prediction': prediction_native}
    except Exception as e:
        return {'error': str(e)}

@app.get('/placement_schema', response_model=dict)
def get_placement_schema():
    return Placement.schema()['properties']

# New endpoint that does not use the Placement schema
@app.post('/web_prediction')
async def web_prediction(data: dict):
    try:
        # Check if the input data contains all required columns
        required_columns = numerical_cols + categorical_cols + text_cols
        missing_columns = [col for col in required_columns if col not in data]
        
        if missing_columns:
            return {'error': f'Missing columns: {", ".join(missing_columns)}'}

        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess the data and make a prediction
        preprocessed_data = pipe.transform(input_df)
        print(preprocessed_data)
        prediction = model.predict(preprocessed_data)
        prediction_native = prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0]
        
        return {'prediction': prediction_native}
    
    except Exception as e:
        return {'error': str(e)}
