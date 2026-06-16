import json
import logging
import re
from typing import List, Tuple

from app.llm import chat_complete
from app.models import IntentLabel, IntentResult

logger = logging.getLogger(__name__)

# Patterns we are certain are chitchat — matched before any LLM call.
# Deliberately narrow: only include patterns with essentially zero false-positive risk.
# Anything ambiguous goes to the LLM rather than being short-circuited here.
_CHITCHAT_RE = re.compile(
    r"""
    ^(
        (hi|hello|hey|howdy|greetings)[!,. ]*
        | (bye|goodbye|see\s+you|later)[!,. ]*
        | (thanks?(\s+you)?|thank\s+you)(\s+(a\s+lot|so\s+much|very\s+much))?[!,. ]*
        | how\s+are\s+you[?!. ]*
        | (good\s+)?(morning|afternoon|evening)[!,. ]*
        | (ok|okay|sure|great|awesome|cool|nice|got\s+it|sounds\s+good)[!,. ]*
        | (yes|no|yeah|nope|yep)[!,. ]*
        | you('re|\s+are)\s+(welcome|great|helpful|amazing|fantastic)[!,. ]*
    )$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CLASSIFY_SYSTEM = """\
You are an intent classifier for a document question-answering system.

Classify the user message into exactly one of three labels:

"knowledge_query"
  The user is asking a question that likely requires searching a document knowledge base.
  Examples: "what does the warranty cover?", "how do I reset my password?",
            "what are the cancellation terms?", "can you explain the refund process?"

"chitchat"
  The user is making small talk, expressing an emotion, or interacting conversationally
  with no underlying information need. A conversational reply is appropriate.
  Examples: "you're so helpful!", "that makes sense", "haha okay", "interesting!"

"unclear"
  The user's message may contain an information need but is too vague for retrieval
  to return anything useful. A clarifying question should be asked — NOT a conversational
  reply and NOT a search attempt.
  Examples: "can you help me?", "tell me more", "what about that thing?", "and the other one?"

Behavior distinction — important:
  "chitchat" → respond conversationally (no retrieval, no clarifying question).
  "unclear"  → ask a clarifying question, e.g. "Could you tell me more about what
                you're looking for?" The user likely has an information need but hasn't
                expressed it specifically enough for retrieval to be useful.

Return ONLY valid JSON: {"label": "<label>"}
"""

_REWRITE_SYSTEM = """\
You are a search query optimizer for a document retrieval system.

Rewrite the user's question to improve semantic matching against document passages:
- Strip conversational filler ("can you tell me", "I was wondering", "I'd like to know")
- Use concise, noun-heavy phrasing that mirrors how technical documents are written
- Expand obvious abbreviations if their meaning is unambiguous from context
- Do NOT add information that is not present in the original query

Single-turn limitation: this system has no conversation history. Queries containing
unresolvable pronouns (e.g. "what about it?") should be returned unchanged.

{expansion_instruction}

Return ONLY valid JSON: {shape}
"""

_SHAPE_SINGLE = '{"rewritten": "<rewritten query>"}'
_SHAPE_EXPANDED = '{"rewritten": "<rewritten query>", "alternatives": ["<alt1>", "<alt2>"]}'


def _is_obvious_chitchat(query: str) -> bool:
    return bool(_CHITCHAT_RE.match(query.strip()))


async def classify_intent(query: str) -> IntentResult:
    """
    Classify the query as "chitchat", "knowledge_query", or "unclear".

    Caller behavior by label:
      chitchat        → respond conversationally; skip retrieval entirely.
      knowledge_query → rewrite and search the knowledge base.
      unclear         → ask a clarifying question (e.g. "Could you clarify what
                        you'd like to know?"). Unlike chitchat, the user likely has
                        an information need but hasn't expressed it specifically enough
                        for retrieval to return anything useful. Do not search; do not
                        respond as if it were small talk.

    Two-stage approach:
      1. Regex pre-filter — catches high-confidence chitchat at ~0 ms with no API cost.
         Only patterns with essentially zero false-positive risk are included; anything
         ambiguous goes to the LLM rather than being decided here.
      2. Mistral LLM (mistral-small-latest, JSON mode, temp=0) — handles everything else.

    Failure fallback: if the LLM call fails or returns an unparseable response, this
    defaults to "knowledge_query". Failing toward search is safer than failing silently —
    worst case the user gets empty results rather than no response at all.
    """
    if _is_obvious_chitchat(query):
        return IntentResult(label=IntentLabel.CHITCHAT, requires_search=False)

    try:
        raw = await chat_complete(
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": query},
            ],
            model="mistral-small-latest",
            temperature=0.0,
            json_mode=True,
            timeout=8.0,
        )
        label = IntentLabel(json.loads(raw)["label"])
    except Exception as exc:
        logger.warning("classify_intent failed (%s); defaulting to knowledge_query", exc)
        label = IntentLabel.KNOWLEDGE_QUERY

    return IntentResult(
        label=label,
        requires_search=(label == IntentLabel.KNOWLEDGE_QUERY),
    )


async def rewrite_query(query: str, expand: bool = False) -> Tuple[str, List[str]]:
    """
    Rewrite a query to improve retrieval quality via a single Mistral call.

    Strips conversational filler, uses document-style phrasing, and expands obvious
    abbreviations in one round trip.

    Single-turn limitation: pronoun resolution requires conversation history, which is
    not yet implemented. Queries like "what about it?" are passed through unchanged.

    expand=False (default): returns a single rewritten query, no alternatives.
    expand=True: also returns up to 2 alternative phrasings (query expansion).
      Trade-off: alternatives improve recall when a passage uses different terminology
      than the query (e.g. "warranty" vs. "guarantee"), but each alternative adds a
      retrieval call downstream. Only enable when eval data justifies the latency cost.

    Returns (rewritten_query, alternatives) where alternatives=[] when expand=False.
    """
    expansion_instruction = (
        "Also generate 1-2 alternative phrasings of the rewritten query that approach "
        "the same information need from a different angle."
        if expand
        else "Do not generate alternative phrasings."
    )
    system = _REWRITE_SYSTEM.format(
        expansion_instruction=expansion_instruction,
        shape=_SHAPE_EXPANDED if expand else _SHAPE_SINGLE,
    )

    try:
        raw = await chat_complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            model="mistral-small-latest",
            temperature=0.1,
            json_mode=True,
            timeout=10.0,
        )
        data = json.loads(raw)
        rewritten = data.get("rewritten", query)
        alternatives = data.get("alternatives", []) if expand else []
    except Exception as exc:
        logger.warning("rewrite_query failed (%s); returning original query", exc)
        rewritten, alternatives = query, []

    return rewritten, alternatives
