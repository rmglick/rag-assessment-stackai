from pydantic import BaseModel
from typing import List, Optional


class IngestFileResult(BaseModel):
    filename: str
    chunks_created: int


class IngestResponse(BaseModel):
    files_ingested: int
    results: List[IngestFileResult]
