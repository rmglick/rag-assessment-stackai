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


class IntentResult(BaseModel):
    label: IntentLabel
    requires_search: bool


class RankedChunk(BaseModel):
    chunk_id: int
    text: str
    rrf_score: float
    semantic_score: float
    metadata: Dict[str, Any]


class QueryRequest(BaseModel):
    query: str


class QuerySearchResponse(BaseModel):
    original_query: str
    rewritten_query: Optional[str] = None
    intent: IntentResult
    insufficient_evidence: bool = False
    results: List[RankedChunk] = []
