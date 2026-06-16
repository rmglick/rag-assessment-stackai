import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "it",
    "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "what", "which", "who", "how", "when", "where", "why",
    "not", "no", "so", "if", "then", "than", "up", "out", "about", "into",
}

_PUNCT_RE = re.compile(r"[^\w\s]")


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords and single-char tokens."""
    text = _PUNCT_RE.sub(" ", text.lower())
    return [t for t in text.split() if t not in _STOPWORDS and len(t) > 1]


class KeywordIndex:
    """
    Inverted index with Okapi BM25 scoring.

    Parameters k1=1.5 and b=0.75 are standard BM25 defaults:
      k1 controls term-frequency saturation (higher = more weight to repeated terms).
      b controls document-length normalization (1.0 = full normalization, 0.0 = none).
    """

    K1 = 1.5
    B = 0.75

    def __init__(self) -> None:
        # term → [(chunk_id, term_frequency), ...]
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._doc_lengths: List[int] = []
        self._total_length: int = 0

    @property
    def _n_docs(self) -> int:
        return len(self._doc_lengths)

    @property
    def _avg_dl(self) -> float:
        return self._total_length / self._n_docs if self._n_docs else 1.0

    def add(self, chunk_id: int, text: str) -> None:
        """
        Index a chunk. chunk_id must equal the current number of indexed chunks
        (i.e. chunks must be added in the same order as VectorStore entries).
        """
        assert len(self._doc_lengths) == chunk_id, (
            f"chunk_id mismatch: expected {len(self._doc_lengths)}, got {chunk_id}"
        )
        tokens = tokenize(text)
        self._doc_lengths.append(len(tokens))
        self._total_length += len(tokens)

        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        for term, freq in tf.items():
            self._postings[term].append((chunk_id, freq))

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Score chunks against the query with BM25 and return top_k (chunk_id, score) pairs.

        IDF formula: log((N − df + 0.5) / (df + 0.5) + 1)
        The +1 prevents negative IDF when a term appears in more than half the corpus.
        """
        if not self._n_docs:
            return []

        scores: Dict[int, float] = {}
        avg_dl = self._avg_dl
        n = self._n_docs

        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for chunk_id, tf in postings:
                dl = self._doc_lengths[chunk_id]
                tf_norm = (tf * (self.K1 + 1.0)) / (
                    tf + self.K1 * (1.0 - self.B + self.B * dl / avg_dl)
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * tf_norm

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
