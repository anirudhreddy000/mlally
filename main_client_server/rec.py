import logging
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import pandas as pd
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def receive_csv(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: Literal["classification", "regression"] = Form(...)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode()))

        if target_column not in df.columns:
            return {"error": f"Column '{target_column}' not found in CSV"}

        logger.info(f"✅ Received File: {file.filename}")
        logger.info(f"📊 Target Column: {target_column}")
        logger.info(f"🧠 Task Type: {task_type}")
        logger.info(f"📄 Columns: {list(df.columns)}")
        logger.info(f"🔢 First 5 values in '{target_column}':\n{df[target_column].head()}")
        logger.info(f"📐 Data shape: {df.shape}")
        logger.info("✅ Done.")

        return {
            "columns": list(df.columns),
            "target_column": target_column,
            "task": task_type,
            "sample_values": df[target_column].head(5).tolist(),
            "rows": len(df),
        }

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return {"error": str(e)}
