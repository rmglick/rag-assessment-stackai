from typing import List, Tuple

from app.models import RankedChunk
from app.vector_store import VectorStore

_RRF_K = 60
_DEFAULT_TOP_K = 5
_DEFAULT_THRESHOLD = 0.70


def merge_and_rerank(
    keyword_results: List[Tuple[int, float]],
    semantic_results: List[Tuple[int, float]],
    store: VectorStore,
    top_k: int = _DEFAULT_TOP_K,
    similarity_threshold: float = _DEFAULT_THRESHOLD,
) -> Tuple[List[RankedChunk], bool]:
    """
    Merge BM25 and cosine-similarity results using Reciprocal Rank Fusion (RRF).

    RRF score for a chunk = Σ 1/(rank + k) across each result list it appears in.
    k=60 is the standard empirical default from the original RRF paper; it dampens
    the influence of very high-ranked results, making the fusion robust to either
    signal dominating on any single query.

    Why RRF over weighted score averaging:
      BM25 scores are unbounded (typically 2–20+) while cosine similarities are
      bounded [0, 1]. Averaging them directly would require per-corpus normalization
      to prevent the higher-magnitude signal from drowning out the other. RRF avoids
      this entirely by fusing rank positions, which are naturally on the same ordinal
      scale regardless of the underlying scoring function.

    Insufficient-evidence flag:
      After fusion, the highest cosine similarity among the top results is checked
      against `similarity_threshold` (default 0.70). If below it, `insufficient_evidence`
      is True and the caller should decline to answer rather than generate a response
      from weakly-matched chunks.
      Note: mistral-embed has a naturally high baseline similarity (~0.64–0.68 even
      for unrelated documents), so 0.30 is too permissive. 0.70 was calibrated against
      observed score distributions: truly relevant results score 0.71+, irrelevant
      ones cluster at 0.64–0.68.
    """
    rrf: dict[int, float] = {}
    for rank, (chunk_id, _) in enumerate(keyword_results):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (rank + _RRF_K)
    for rank, (chunk_id, _) in enumerate(semantic_results):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (rank + _RRF_K)

    top_ids = sorted(rrf, key=rrf.__getitem__, reverse=True)[:top_k]

    semantic_score_map = {chunk_id: score for chunk_id, score in semantic_results}
    best_semantic = max(
        (semantic_score_map.get(cid, 0.0) for cid in top_ids), default=0.0
    )
    insufficient_evidence = best_semantic < similarity_threshold

    ranked: List[RankedChunk] = []
    for chunk_id in top_ids:
        entry = store.get(chunk_id)
        ranked.append(RankedChunk(
            chunk_id=chunk_id,
            text=entry.text,
            rrf_score=rrf[chunk_id],
            semantic_score=semantic_score_map.get(chunk_id, 0.0),
            metadata=entry.metadata,
        ))

    return ranked, insufficient_evidence
