from enum import Enum
from typing import List, Optional

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


class QueryRequest(BaseModel):
    query: str


class QueryDebugResponse(BaseModel):
    original_query: str
    intent: IntentResult
    rewritten_query: Optional[str] = None
    alternatives: List[str] = []
