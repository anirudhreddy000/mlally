import fitz
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
gen_tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-783M").to(device)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join(page.get_text() for page in doc)

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

def generate_answer(query, retrieved_chunks, similarities):
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    similarity_threshold = 0.25
    
    if avg_similarity < similarity_threshold:
        prompt = (
            f"You are a helpful AI assistant. The user's question doesn't seem to be related "
            f"to any specific document content. Respond in a friendly and helpful manner.\n\n"
            f"User message: {query}\n\n"
            f"Response:"
        )
    else:
        context = " ".join(retrieved_chunks)
        prompt = (
            f"You are a highly intelligent assistant with access to the context from a document.\n"
            f"Use the context below to answer the question clearly, informatively, and in detail. "
            f"Make sure your answer is based on the context provided and relevant to the user's question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
    
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = gen_model.generate(**inputs, max_new_tokens=350)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    pdf_path = "harry.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)
    
    print("Chat with your PDF (type 'exit' to stop)")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        top_chunks, similarities = search_index(query, index, chunks)
        answer = generate_answer(query, top_chunks, similarities)
        print(f"Bot: {answer}")

if __name__ == "__main__":
    main()