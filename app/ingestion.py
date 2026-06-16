import io
from typing import List, Tuple

import pdfplumber

from app.embeddings import get_embeddings
from app.keyword_search import KeywordIndex
from app.vector_store import VectorStore

# 1 token ≈ 0.75 words; targets are ~400 tokens and ~50 tokens respectively.
_CHUNK_SIZE_WORDS = 530
_OVERLAP_WORDS = 67


def extract_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """Return a list of (1-based page number, text) for every non-empty page."""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((page_num, text))
    return pages


def chunk_text(
    text: str,
    chunk_size: int = _CHUNK_SIZE_WORDS,
    overlap: int = _OVERLAP_WORDS,
) -> List[str]:
    """
    Split text into fixed-size word chunks with overlap.

    Strategy — fixed-size with overlap:
      Words are used as a token proxy (1 token ≈ 0.75 words), so chunk_size=530
      targets ~400 tokens and overlap=67 targets ~50 tokens.  A sliding window
      steps by (chunk_size - overlap) words each iteration; the overlap carries
      the tail of the previous chunk into the next so a sentence severed at a
      boundary is still captured by at least one chunk.

    Tradeoffs vs. alternatives:
      - Sentence-based: better semantic coherence (no mid-sentence cuts), but
        produces variable-length chunks and requires a sentence tokenizer.
      - Page-based: respects document structure but pages vary wildly in density
        (50–2000+ words), causing inconsistent retrieval.
      - Semantic chunking: best coherence, but requires embedding every sentence
        first — expensive and circular in a RAG pipeline.
    """
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks


async def ingest_pdf(
    filename: str,
    pdf_bytes: bytes,
    store: VectorStore,
    keyword_index: KeywordIndex,
) -> int:
    """
    Extract, chunk, embed, and store all content from a single PDF.
    Adds each chunk to both the vector store (for semantic search) and the
    keyword index (for BM25 search). Returns the number of chunks created.
    """
    pages = extract_pages(pdf_bytes)

    chunk_pairs: List[Tuple[str, dict]] = []
    for page_num, page_text in pages:
        for chunk in chunk_text(page_text):
            chunk_pairs.append((
                chunk,
                {
                    "source": filename,
                    "page": page_num,
                    "chunk_index": len(chunk_pairs),
                },
            ))

    if not chunk_pairs:
        return 0

    texts = [text for text, _ in chunk_pairs]
    vectors = await get_embeddings(texts)

    for (text, metadata), vector in zip(chunk_pairs, vectors):
        chunk_id = store.add(text, vector, metadata)
        keyword_index.add(chunk_id, text)

    return len(chunk_pairs)
