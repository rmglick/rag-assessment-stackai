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

## Design Notes

### Data ingestion: text extraction and chunking
PDFs are parsed page-by-page with `pdfplumber`, then chunked using fixed-size windows (400 tokens, ~50 token overlap), with no forced reset at page boundaries. Fixed-size chunking was chosen over sentence-based (variable size, needs a tokenizer dependency) or semantic chunking (circular — requires embeddings to decide where to chunk before chunks exist). Overlap recovers context that would otherwise be lost when a sentence is split across a boundary. Each chunk retains its source filename and page number(s) for citation purposes.

### Query intent detection
A regex pre-filter catches high-confidence chitchat (greetings, thanks) for free with zero latency; anything ambiguous falls through to a lightweight Mistral classification call returning one of `chitchat`, `knowledge_query`, or `unclear`. `unclear` queries get a clarifying question rather than being silently treated like chitchat. If the classification call fails, the system defaults to `knowledge_query` — failing toward search is safer than failing toward silence. Known limitation: the classifier has no awareness of the knowledge base's actual contents, so generic phrasing can occasionally be misclassified; document-specific phrasing resolves this.

### Query rewriting
A single Mistral call removes conversational filler (e.g., "can you tell me") that doesn't help retrieval. Query expansion (generating an alternative phrasing to broaden recall) is implemented but defaults to off, since it doubles retrieval calls without measured evidence it improves results for this use case.

### Hybrid search: keyword + semantic
Keyword search uses a from-scratch Okapi BM25 implementation (k1=1.5, b=0.75) over a hand-built inverted index, which rewards rare/distinguishing terms and normalizes for chunk length. Semantic search embeds the query via Mistral and ranks chunks by cosine similarity over an in-memory numpy vector store. The two result sets (top 20 each) are merged using Reciprocal Rank Fusion (k=60) rather than weighted score averaging, since BM25 and cosine similarity scores live on incomparable scales — combining ranks avoids needing to normalize two different distributions. This gives literal-match precision (BM25) and conceptual recall (semantic) in one ranked list.

### Evidence threshold and citations
If the top result's cosine similarity falls below a configurable threshold (default 0.3), the result set is flagged `insufficient_evidence` and generation is skipped entirely, returning a refusal instead of a guess. The generation prompt instructs the model to cite claims using position numbers (`[1]`, `[2]`...) matching the numbered context blocks; these are resolved server-side to the actual chunk's real filename/page metadata, rather than trusting the model to transcribe a chunk ID correctly. This makes citations reliable even if the model's output is slightly malformed.

### Generation and hallucination check
The final answer is generated via a single Mistral call constrained to the retrieved context, with citations required per claim. A second, lightweight Mistral call scans the generated answer against the source chunks and flags any unsupported sentences (`flagged_claims`), surfaced to the user without blocking the response — a non-blocking transparency check rather than a hard gate, given the added latency/cost of a second call.
