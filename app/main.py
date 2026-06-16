from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.embeddings import get_embeddings
from app.generation import generate_answer
from app.ingestion import ingest_pdf
from app.intent import classify_intent, rewrite_query
from app.keyword_search import KeywordIndex
from app.models import (
    IngestFileResult,
    IngestResponse,
    IntentLabel,
    QueryRequest,
    QueryResponse,
)
from app.retrieval import merge_and_rerank
from app.vector_store import VectorStore

load_dotenv()

app = FastAPI(title="RAG Assessment API")
store = VectorStore()
keyword_index = KeywordIndex()

_CHITCHAT_REPLY = (
    "Hello! I'm a document Q&A assistant. "
    "Ask me anything about the documents in my knowledge base."
)
_UNCLEAR_REPLY = (
    "Could you clarify what you're looking for? "
    "I can help answer specific questions about the documents in the knowledge base."
)


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
        chunk_count = await ingest_pdf(file.filename, pdf_bytes, store, keyword_index)
        results.append(IngestFileResult(filename=file.filename, chunks_created=chunk_count))

    return IngestResponse(files_ingested=len(results), results=results)


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    intent = await classify_intent(body.query)

    # Chitchat and unclear intents short-circuit before any retrieval or generation.
    # Chitchat gets a conversational reply; unclear gets a clarifying question.
    # Neither makes a search or generation Mistral call.
    if intent.label == IntentLabel.CHITCHAT:
        return QueryResponse(
            original_query=body.query,
            intent=intent,
            answer=_CHITCHAT_REPLY,
        )

    if intent.label == IntentLabel.UNCLEAR:
        return QueryResponse(
            original_query=body.query,
            intent=intent,
            answer=_UNCLEAR_REPLY,
        )

    rewritten, _ = await rewrite_query(body.query)

    # Semantic search uses the rewritten query (cleaner phrasing improves embedding match).
    # BM25 uses the original query — the rewriter sometimes substitutes abstract terms
    # that don't appear verbatim in source documents, hurting keyword recall.
    query_vector = (await get_embeddings([rewritten]))[0]
    semantic_results = store.semantic_search(query_vector, top_k=20)
    keyword_results = keyword_index.search(body.query, top_k=20)

    ranked_chunks, insufficient_evidence = merge_and_rerank(
        keyword_results=keyword_results,
        semantic_results=semantic_results,
        store=store,
    )

    answer_result = await generate_answer(body.query, ranked_chunks, insufficient_evidence)

    return QueryResponse(
        original_query=body.query,
        rewritten_query=rewritten,
        intent=intent,
        insufficient_evidence=insufficient_evidence,
        answer=answer_result.answer,
        citations=answer_result.citations,
        flagged_claims=answer_result.flagged_claims,
        results=ranked_chunks,
    )
