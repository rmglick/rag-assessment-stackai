import os
from typing import List

import httpx

_MISTRAL_EMBEDDINGS_URL = "https://api.mistral.ai/v1/embeddings"
_MISTRAL_EMBED_MODEL = "mistral-embed"
_BATCH_SIZE = 32


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Fetch embeddings from the Mistral API, batching to stay within API limits."""
    api_key = os.environ["MISTRAL_API_KEY"]
    all_embeddings: List[List[float]] = []

    async with httpx.AsyncClient() as client:
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = await client.post(
                _MISTRAL_EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": _MISTRAL_EMBED_MODEL, "input": batch},
                timeout=60.0,
            )
            response.raise_for_status()

            data = response.json()
            ordered = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend(item["embedding"] for item in ordered)

    return all_embeddings
