from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import fitz
import torch
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

# Load models
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
gen_tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-783M").to(device)

# Persistent variables
stored_chunks = []
stored_index = None
stored_metadata = ""
stored_instruction = ""

# Utility functions
def extract_text_from_pdf_file(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return " ".join(page.get_text() for page in doc)

def extract_metadata(text, lines=30):
    return "\n".join(text.splitlines()[:lines])

def chunk_text(text, max_length=512):
    paragraphs = text.split('\n\n')
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_length:
            current_chunk += para + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def get_embeddings(texts, batch_size=32):
    return embed_model.encode(texts, convert_to_numpy=True, batch_size=batch_size, 
                              show_progress_bar=False, normalize_embeddings=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def search_index(query, index, chunks, k=3):
    query_embedding = get_embeddings([query])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]], D[0].tolist()

def join_chunks(chunks, max_tokens=800):
    result, token_count = "", 0
    for chunk in chunks:
        tokens = len(chunk.split())
        if token_count + tokens > max_tokens:
            break
        result += chunk + " "
        token_count += tokens
    return result.strip()

def generate_answer(query, retrieved_chunks, similarities, instruction, metadata=None):
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    similarity_threshold = 0.15

    keywords = query.lower()
    if any(key in keywords for key in ["author", "abstract", "summary"]) and metadata:
        context = metadata
    elif avg_similarity < similarity_threshold:
        prompt = (
            f"{instruction}\n\n"
            f"The user's question doesn't seem related to the document. Respond kindly and informatively.\n"
            f"User message: {query}\n\nResponse:"
        )
        context = ""
    else:
        context = join_chunks(retrieved_chunks)

    if context:
        prompt = (
            f"{instruction}\n\n"
            f"You are a highly intelligent assistant with access to a document.\n"
            f"Use the context below to answer the question clearly and accurately.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )

    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = gen_model.generate(**inputs, max_new_tokens=350)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.get("/")
def read_root():
    return {"message": "Welcome mate"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), instruction: str = Form(...)):
    global stored_chunks, stored_index, stored_metadata, stored_instruction
    try:
        file_bytes = await file.read()
        text = extract_text_from_pdf_file(file_bytes)
        stored_metadata = extract_metadata(text)
        stored_chunks = chunk_text(text)
        embeddings = get_embeddings(stored_chunks)
        stored_index = build_faiss_index(embeddings)
        stored_instruction = instruction
        return JSONResponse({"message": "PDF processed successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def ask_question(req: QueryRequest):
    global stored_chunks, stored_index, stored_metadata, stored_instruction
    if not stored_chunks or stored_index is None:
        return JSONResponse(status_code=400, content={"error": "No PDF has been uploaded yet."})
    try:
        top_chunks, similarities = search_index(req.query, stored_index, stored_chunks)
        answer = generate_answer(req.query, top_chunks, similarities, stored_instruction, stored_metadata)
        return JSONResponse({"response": answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
