from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class IngestFileResult(BaseModel):
    filename: str
    chunks_created: int


class IngestResponse(BaseModel):
    files_ingested: int
    results: List[IngestFileResult]


class IntentLabel(str, Enum):
    CHITCHAT = "chitchat"
    KNOWLEDGE_QUERY = "knowledge_query"
    UNCLEAR = "unclear"


class AnswerFormat(str, Enum):
    PLAIN = "plain"
    LIST = "list"
    TABLE = "table"


class IntentResult(BaseModel):
    label: IntentLabel
    requires_search: bool
    answer_format: AnswerFormat = AnswerFormat.PLAIN
    policy_flag: Optional[str] = None  # "pii", "legal", "medical", or None


class RankedChunk(BaseModel):
    chunk_id: int
    text: str
    rrf_score: float
    semantic_score: float
    metadata: Dict[str, Any]


class Citation(BaseModel):
    chunk_id: int
    filename: str
    page: int


class AnswerResult(BaseModel):
    answer: str
    citations: List[Citation] = []
    flagged_claims: List[str] = []


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    original_query: str
    rewritten_query: Optional[str] = None
    intent: IntentResult
    insufficient_evidence: bool = False
    answer: Optional[str] = None
    citations: List[Citation] = []
    flagged_claims: List[str] = []
    results: List[RankedChunk] = []
