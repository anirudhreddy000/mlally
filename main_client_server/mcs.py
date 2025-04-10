import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, create_model, ValidationError
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from json.decoder import JSONDecodeError
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def home():
    return {'Welcome to ': 'MAIN CLIENT SERVER !!!'}

@app.post('/send_data')
async def send_data(request: Request):
    try:
        request_data = await request.json()
    except JSONDecodeError:
        logging.error("Invalid JSON in request body")
        return {"error": "Invalid JSON in request body"}

    logging.info(f"Received request data: {request_data}")

    async with httpx.AsyncClient() as client:
        try:
            schema_response = await client.get("http://127.0.0.1:8005/placement_schema")
            schema_response.raise_for_status()
            schema = schema_response.json()

            logging.info(f"Received schema: {schema}")

            field_definitions = {}
            for field, props in schema.items():
                if props['type'] == 'number':
                    field_definitions[field] = (float, ...)
                elif props['type'] == 'integer':
                    field_definitions[field] = (int, ...)
                elif props['type'] == 'string':
                    field_definitions[field] = (str, ...)
                elif props['type'] == 'boolean':
                    field_definitions[field] = (bool, ...)

            DynamicModel = create_model('DynamicModel', **field_definitions)

            try:
                dynamic_data = DynamicModel(**request_data)
            except ValidationError as e:
                logging.error(f"Validation error: {e.errors()}")
                raise HTTPException(status_code=422, detail=e.errors())

            payload = {field: request_data.get(field, None) for field in schema.keys()}

            logging.info(f"Constructed payload: {payload}")

            response = await client.post("http://127.0.0.1:8005/predict_placement", json=payload)
            response.raise_for_status()
            response_data = response.json()

            response_body = {key: payload.get(key, None) for key in schema.keys()}
            response_body.update(response_data) 

            return response_body
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP status error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logging.error(f"Unhandled exception: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/send_training")
async def send_training_data(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...)
):
    try:
        form_data = {
            "target_column": target_column,
            "task_type": task_type
        }

        files = {
            "file": (file.filename, await file.read(), file.content_type)
        }

        async with httpx.AsyncClient() as client:
            response = await client.post("http://127.0.0.1:8003/upload_csv", data=form_data, files=files)
            return response.json()

    except Exception as e:
        return {"error": str(e)}


@app.post('/send_images')
async def send_images(file: UploadFile = File(...)):
    async with httpx.AsyncClient() as client:
        try:
            logging.info("Sending file: %s", file.filename)
            file_content = await file.read()
            files = {'train_file': (file.filename, file_content, file.content_type)}
            response = await client.post("http://127.0.0.1:8004/train_model", files=files)
            response.raise_for_status()
            response_accuracy = response.json()
            logging.info("Received response: %s", response_accuracy)
            return response_accuracy
        except httpx.HTTPStatusError as exc:
            logging.error("HTTP status error: %s", exc.response.text)
            raise HTTPException(status_code=exc.response.status_code, detail=f"HTTP error occurred: {exc.response.content.decode()}")
        except Exception as exc:
            logging.error("An error occurred: %s", str(exc))
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(exc)}")
        