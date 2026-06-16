from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class VectorEntry:
    text: str
    vector: np.ndarray
    metadata: dict


class VectorStore:
    def __init__(self):
        self._entries: List[VectorEntry] = []

    def add(self, text: str, vector: List[float], metadata: dict) -> None:
        self._entries.append(VectorEntry(
            text=text,
            vector=np.array(vector, dtype=np.float32),
            metadata=metadata,
        ))

    def search(self, query_vector: List[float], top_k: int = 5) -> List[dict]:
        if not self._entries:
            return []

        query = np.array(query_vector, dtype=np.float32)
        query = query / np.linalg.norm(query)

        # Stack all stored vectors and batch-compute cosine similarity.
        matrix = np.stack([e.vector for e in self._entries])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / norms
        scores = matrix @ query

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "text": self._entries[i].text,
                "score": float(scores[i]),
                "metadata": self._entries[i].metadata,
            }
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._entries)
