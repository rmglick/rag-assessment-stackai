from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.ingestion import ingest_pdf
from app.models import IngestFileResult, IngestResponse
from app.vector_store import VectorStore

load_dotenv()

app = FastAPI(title="RAG Assessment API")
store = VectorStore()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results: List[IngestFileResult] = []
    for file in files:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' is not a PDF. Only .pdf files are accepted.",
            )

        pdf_bytes = await file.read()
        chunk_count = await ingest_pdf(file.filename, pdf_bytes, store)
        results.append(IngestFileResult(filename=file.filename, chunks_created=chunk_count))

    return IngestResponse(files_ingested=len(results), results=results)
