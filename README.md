# RAG Assessment

A Retrieval-Augmented Generation backend built with FastAPI and the Mistral AI API.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then add your MISTRAL_API_KEY
uvicorn app.main:app --reload
```

## API Endpoints

### `POST /ingest`

Upload one or more PDF files (multipart/form-data, field name `files`).

Returns the number of files ingested and chunks created per file.

## Design Notes

_To be filled in._
