import os
from typing import Any

import httpx

_MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"


async def chat_complete(
    messages: list[dict],
    model: str = "mistral-small-latest",
    temperature: float = 0.1,
    json_mode: bool = False,
    timeout: float = 10.0,
) -> str:
    """Call the Mistral chat completions API and return the response content string."""
    api_key = os.environ["MISTRAL_API_KEY"]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            _MISTRAL_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
