import json
import logging
import re
from typing import List, Tuple

from app.llm import chat_complete
from app.models import AnswerFormat, IntentLabel, IntentResult

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

# High-confidence PII patterns. Deliberately conservative — only formats where
# the pattern itself is the clearest signal (SSN, credit card). Loose digit
# sequences are excluded to avoid false positives on policy numbers, dates, etc.
_PII_RE = re.compile(
    r"""
    \b\d{3}[-\s]\d{2}[-\s]\d{4}\b    # SSN: 123-45-6789
    | \b(?:\d{4}[-\s]?){3}\d{4}\b     # Credit card: 1234-5678-9012-3456
    """,
    re.VERBOSE,
)

_CLASSIFY_SYSTEM = """\
You are an intent classifier for a document question-answering system.

Return JSON with three fields: "label", "answer_format", and "policy_flag".

LABEL — exactly one of:
  "knowledge_query": user is asking a question that likely requires searching a document
    knowledge base. Examples: "what does the warranty cover?", "what are the cancellation terms?"
  "chitchat": user is making small talk with no underlying information need.
    Examples: "you're so helpful!", "that makes sense", "haha okay"
  "unclear": user's message may contain an information need but is too vague for retrieval.
    Examples: "can you help me?", "tell me more", "what about that thing?"

  Distinction: "chitchat" → respond conversationally (no search).
               "unclear"  → ask a clarifying question (no search, not small talk).

ANSWER_FORMAT — best format for knowledge_query responses (use "plain" for chitchat/unclear):
  "plain": a direct factual answer — single fact, short paragraph, or brief explanation.
  "list":  question asks for multiple distinct items.
    Examples: "what are all the covered services?", "list the steps to...", "what are the requirements?"
  "table": question asks for a comparison or multi-attribute breakdown.
    Examples: "compare plan A and plan B", "what are the differences between X and Y?"

POLICY_FLAG — null unless the query requests personalized professional advice:
  "legal":   user is asking for specific legal advice, legal interpretation of their situation,
             or liability assessment — NOT just a general question about a legal topic.
  "medical": user is asking for a specific medical diagnosis, treatment recommendation,
             or prescription — NOT just a general health/medical information question.
  Use null for all general informational questions, even about legal or medical topics.

Return ONLY valid JSON:
{"label": "<label>", "answer_format": "<plain|list|table>", "policy_flag": <null|"legal"|"medical">}
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


def _has_pii(query: str) -> bool:
    return bool(_PII_RE.search(query))


async def classify_intent(query: str) -> IntentResult:
    """
    Classify the query as "chitchat", "knowledge_query", or "unclear".

    Also returns answer_format ("plain", "list", "table") and policy_flag
    ("pii", "legal", "medical", or None) in a single pass.

    Three-stage approach:
      1. PII regex — catches SSNs and credit card numbers before any LLM call.
         Returns policy_flag="pii"; caller should refuse the query.
      2. Chitchat regex — catches high-confidence greetings/affirmations at ~0 ms.
      3. Mistral LLM (JSON mode, temp=0) — classifies label, format, and policy flag
         for everything else in one call.

    Failure fallback: defaults to knowledge_query on any LLM error.
    """
    if _has_pii(query):
        return IntentResult(
            label=IntentLabel.KNOWLEDGE_QUERY,
            requires_search=False,
            policy_flag="pii",
        )

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
        data = json.loads(raw)
        label = IntentLabel(data["label"])
        answer_format = AnswerFormat(data.get("answer_format", "plain"))
        policy_flag = data.get("policy_flag") or None
    except Exception as exc:
        logger.warning("classify_intent failed (%s); defaulting to knowledge_query", exc)
        label = IntentLabel.KNOWLEDGE_QUERY
        answer_format = AnswerFormat.PLAIN
        policy_flag = None

    return IntentResult(
        label=label,
        requires_search=(label == IntentLabel.KNOWLEDGE_QUERY),
        answer_format=answer_format,
        policy_flag=policy_flag,
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
