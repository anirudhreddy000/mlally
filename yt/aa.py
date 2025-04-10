import fitz  # PyMuPDF
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load the embedding model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_length=512):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk + sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def get_embedding(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state[:, 0]  # [CLS] token
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(query, index, chunks, k=3):
    query_embedding = get_embedding([query])
    D, I = index.search(query_embedding, k)
    results = [chunks[i] for i in I[0]]
    return results

# === Usage Example ===
pdf_path = "harry.pdf"  # Replace with uploaded file path
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)

print(f"Total chunks: {len(chunks)}")

embeddings = get_embedding(chunks)
index = build_faiss_index(embeddings)

# Chat loop
print("\nChat with your PDF! Type 'exit' to stop.\n")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    results = search_index(query, index, chunks)
    print("Bot:")
    for r in results:
        print("-", r)
