import json
import logging
from typing import List, Optional

from app.llm import chat_complete
from app.models import AnswerFormat, AnswerResult, Citation, RankedChunk

logger = logging.getLogger(__name__)

_INSUFFICIENT_ANSWER = (
    "I don't have enough information in the knowledge base to answer this question confidently."
)
_ERROR_ANSWER = "I encountered an error generating a response — please try again."

_LEGAL_DISCLAIMER = (
    "\n\n---\n"
    "Disclaimer: This answer is for informational purposes only and does not constitute "
    "legal advice. Consult a qualified attorney for guidance specific to your situation."
)
_MEDICAL_DISCLAIMER = (
    "\n\n---\n"
    "Disclaimer: This answer is for informational purposes only and does not constitute "
    "medical advice. Consult a qualified healthcare professional for guidance specific to your situation."
)

_GENERATE_SYSTEM_PLAIN = """\
You are a precise, document-grounded Q&A assistant.

Answer the user's question using ONLY the numbered context chunks provided. Rules:

1. Every claim must be traceable to a context chunk. Add inline citations using the
   chunk's bracket number, e.g. [1] or [2].
2. If the context doesn't fully answer the question, say so explicitly — do not fill
   gaps with outside knowledge or make inferences beyond what the text supports.
3. Be concise and factual. Prefer direct answers over lengthy summaries.

Context chunks are labeled [1], [2], [3]... Use those same numbers in citations.

Return ONLY valid JSON in this exact shape:
{
  "answer": "Your answer with inline citations like [1] or [2].",
  "citations": [{"chunk_id": <bracket number, e.g. 1>, "filename": "<source filename>", "page": <page number>}]
}
Include a citation entry only for chunks you actually reference. Do not repeat the
same chunk_id twice in the citations list.
"""

_GENERATE_SYSTEM_LIST = """\
You are a precise, document-grounded Q&A assistant.

The user's question calls for a list of items. Answer using ONLY the numbered context
chunks provided, formatted as a bulleted list. Rules:

1. Each bullet must be a distinct item traceable to a context chunk.
2. Add an inline citation after each bullet: "- Item text [1]"
3. If the context doesn't cover all expected items, add a final note saying so.
4. Do not add information beyond what the context states.

Context chunks are labeled [1], [2], [3]... Use those same numbers in citations.

Return ONLY valid JSON:
{
  "answer": "- First item [1]\\n- Second item [2]\\n- Third item [1]",
  "citations": [{"chunk_id": <bracket number>, "filename": "<source filename>", "page": <page number>}]
}
Include a citation entry only for chunks you actually reference.
"""

_GENERATE_SYSTEM_TABLE = """\
You are a precise, document-grounded Q&A assistant.

The user's question calls for a comparison or structured breakdown. Answer using ONLY
the numbered context chunks provided, formatted as a markdown table. Rules:

1. Choose column headers appropriate to the comparison being made.
2. Only include rows/cells traceable to a context chunk. Add inline citations [1] in cells.
3. If the context doesn't support all columns or rows, leave the cell blank or note "N/A".
4. Do not add information beyond what the context states.

Context chunks are labeled [1], [2], [3]... Use those same numbers in citations.

Return ONLY valid JSON:
{
  "answer": "| Column A | Column B |\\n|---|---|\\n| value [1] | value [2] |",
  "citations": [{"chunk_id": <bracket number>, "filename": "<source filename>", "page": <page number>}]
}
Include a citation entry only for chunks you actually reference.
"""

_HALLUCINATION_SYSTEM = """\
You are a fact-checker for a document Q&A system. You are given a generated answer
and the source context chunks it was based on.

Identify any sentences in the answer that are NOT fully supported by the provided
context chunks. A sentence is unsupported if it asserts something that cannot be
directly traced to any numbered chunk — including claims that sound plausible but go
beyond what the text actually states.

Return ONLY valid JSON: {"flagged_claims": ["<sentence>", ...]}
Return an empty array if every sentence in the answer is supported by the context.
"""

_FORMAT_TO_SYSTEM = {
    AnswerFormat.PLAIN: _GENERATE_SYSTEM_PLAIN,
    AnswerFormat.LIST: _GENERATE_SYSTEM_LIST,
    AnswerFormat.TABLE: _GENERATE_SYSTEM_TABLE,
}


def _build_context(chunks: List[RankedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] Source: {chunk.metadata['source']}, Page {chunk.metadata['page']}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


async def _check_hallucinations(answer: str, chunks: List[RankedChunk]) -> List[str]:
    """
    Post-hoc hallucination check via a second Mistral call.

    Asks the model to identify sentences in the generated answer that aren't
    traceable to the provided context chunks. Result is surfaced in `flagged_claims`
    for transparency but does NOT block or modify the answer.

    Cost/latency trade-off: adds one additional chat completion round-trip per
    knowledge_query response (~200-400 ms, ~500-1500 tokens depending on answer and
    context length). Acceptable for low-traffic or assessment use. In production,
    gate this behind a flag or run it asynchronously and surface results out-of-band
    (e.g. a /check/{request_id} polling endpoint) to keep the primary response path
    fast. Failures are swallowed — a broken hallucination check should never surface
    as a user-facing error.
    """
    context = _build_context(chunks)
    try:
        raw = await chat_complete(
            messages=[
                {"role": "system", "content": _HALLUCINATION_SYSTEM},
                {
                    "role": "user",
                    "content": f"Context chunks:\n{context}\n\nAnswer to check:\n{answer}",
                },
            ],
            model="mistral-small-latest",
            temperature=0.0,
            json_mode=True,
            timeout=15.0,
        )
        return json.loads(raw).get("flagged_claims", [])
    except Exception as exc:
        logger.warning("Hallucination check failed (%s); skipping", exc)
        return []


async def generate_answer(
    query: str,
    ranked_chunks: List[RankedChunk],
    insufficient_evidence: bool,
    answer_format: AnswerFormat = AnswerFormat.PLAIN,
    policy_flag: Optional[str] = None,
) -> AnswerResult:
    """
    Generate a grounded answer from ranked context chunks.

    If insufficient_evidence is True, returns a fixed refusal without any LLM call.

    Otherwise:
      1. Select system prompt based on answer_format (plain/list/table).
      2. Build numbered context blocks ([1], [2]...) from the top ranked chunks.
      3. Call mistral-small-latest with the format-appropriate grounding prompt.
      4. Parse answer text and citations; map 1-based context positions back to
         actual chunk_ids (so citations are always ground-truth even if the model
         mis-transcribes filename or page).
      5. Append a legal or medical disclaimer when policy_flag indicates it.
      6. Run a lightweight hallucination check as a second Mistral pass.

    Error handling: if the generation call fails or returns unparseable JSON,
    returns a graceful fallback answer rather than propagating a 500.
    """
    if insufficient_evidence or not ranked_chunks:
        return AnswerResult(answer=_INSUFFICIENT_ANSWER, citations=[], flagged_claims=[])

    system = _FORMAT_TO_SYSTEM.get(answer_format, _GENERATE_SYSTEM_PLAIN)
    context = _build_context(ranked_chunks)

    try:
        raw = await chat_complete(
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Context chunks:\n{context}\n\nQuestion: {query}",
                },
            ],
            model="mistral-small-latest",
            temperature=0.1,
            json_mode=True,
            timeout=30.0,
        )
        data = json.loads(raw)
    except Exception as exc:
        logger.error("generate_answer failed (%s); returning fallback", exc)
        return AnswerResult(answer=_ERROR_ANSWER, citations=[], flagged_claims=[])

    answer = data.get("answer", "")

    # Append professional-advice disclaimer when the query was flagged.
    if policy_flag == "legal":
        answer += _LEGAL_DISCLAIMER
    elif policy_flag == "medical":
        answer += _MEDICAL_DISCLAIMER

    # Map 1-based context positions → actual chunk metadata.
    # Pull filename/page from the real chunk rather than trusting the model's
    # transcription, which can be inconsistent.
    citations: List[Citation] = []
    seen: set = set()
    for c in data.get("citations", []):
        pos = c.get("chunk_id")
        if not isinstance(pos, int) or pos in seen or not (1 <= pos <= len(ranked_chunks)):
            continue
        seen.add(pos)
        chunk = ranked_chunks[pos - 1]
        citations.append(Citation(
            chunk_id=chunk.chunk_id,
            filename=chunk.metadata["source"],
            page=chunk.metadata["page"],
        ))

    flagged_claims = await _check_hallucinations(answer, ranked_chunks)

    return AnswerResult(answer=answer, citations=citations, flagged_claims=flagged_claims)
