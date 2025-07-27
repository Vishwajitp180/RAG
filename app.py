from fastapi import FastAPI, UploadFile, File
import fitz  # PyMuPDF

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, FastAPI!"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    contents = await file.read()

    # Save the uploaded file temporarily
    with open(file.filename, "wb") as f:
        f.write(contents)
    
    # Extract text using PyMuPDF
    doc = fitz.open(file.filename)
    text = ""
    for page in doc:
        text += page.get_text()

    # Optional: Delete the file after processing
    doc.close()

    return {
        "filename": file.filename,
        "size": len(contents),
        "text_preview": text[:500]  # return only first 500 chars
    }

# uvicorn app:app --reload

