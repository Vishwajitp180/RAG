from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
import uuid
import fitz  # for PDF
import docx  # for DOCX
import email  # for EML
from email import policy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

UPLOAD_DIR = "uploaded_docs"
EMBED_DIM = 384
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(EMBED_DIM)
corpus_chunks = []

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


@app.get("/")
def home():
    return {"message": "LLM-Powered Document Query System (PDF, DOCX, EML)"}


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def extract_text_from_pdf(file_path: str) -> List[str]:
    doc = fitz.open(file_path)
    full_text = "\n".join([page.get_text() for page in doc])
    return chunk_text(full_text)


def extract_text_from_docx(file_path: str) -> List[str]:
    doc = docx.Document(file_path)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    return chunk_text(full_text)


def extract_text_from_eml(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        msg = email.message_from_file(f, policy=policy.default)
        if msg.is_multipart():
            parts = [part.get_payload(decode=True).decode('utf-8', errors='ignore') 
                     for part in msg.walk() if part.get_content_type() == 'text/plain']
            full_text = "\n".join(parts)
        else:
            full_text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    return chunk_text(full_text)


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if ext == "pdf":
        chunks = extract_text_from_pdf(file_path)
    elif ext == "docx":
        chunks = extract_text_from_docx(file_path)
    elif ext == "eml":
        chunks = extract_text_from_eml(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or EML.")

    embeddings = model.encode(chunks)
    index.add(np.array(embeddings))
    corpus_chunks.extend(chunks)

    return {
        "filename": file.filename,
        "size": os.path.getsize(file_path),
        "chunks": len(chunks),
        "message": f"{ext.upper()} file processed successfully"
    }


@app.post("/query")
async def query_knowledge(question: str):
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")

    query_vec = model.encode([question])
    D, I = index.search(np.array(query_vec), k=3)
    matched_chunks = [corpus_chunks[i] for i in I[0] if i < len(corpus_chunks)]

    combined_context = "\n---\n".join(matched_chunks)
    response = {
        "query": question,
        "top_matches": matched_chunks,
        "combined_context": combined_context,
        "explanation": "This context can be passed to an LLM for a full answer."
    }
    return JSONResponse(content=response)
