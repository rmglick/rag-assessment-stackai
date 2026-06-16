from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class VectorEntry:
    text: str
    vector: np.ndarray
    metadata: dict


class VectorStore:
    def __init__(self) -> None:
        self._entries: List[VectorEntry] = []

    def add(self, text: str, vector: List[float], metadata: dict) -> int:
        """Append a chunk and return its assigned chunk_id (list index)."""
        chunk_id = len(self._entries)
        self._entries.append(VectorEntry(
            text=text,
            vector=np.array(vector, dtype=np.float32),
            metadata=metadata,
        ))
        return chunk_id

    def get(self, chunk_id: int) -> VectorEntry:
        return self._entries[chunk_id]

    def semantic_search(
        self, query_vector: List[float], top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """Return (chunk_id, cosine_similarity) pairs for the top_k closest chunks."""
        if not self._entries:
            return []

        query = np.array(query_vector, dtype=np.float32)
        query = query / np.linalg.norm(query)

        # Batch cosine similarity across all stored vectors.
        matrix = np.stack([e.vector for e in self._entries])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / norms
        scores = matrix @ query

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self._entries)
