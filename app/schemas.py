from typing import List

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int


class AskRequest(BaseModel):
    document_id: str
    question: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=10)


class Source(BaseModel):
    chunk_index: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    answer_found: bool
    sources: List[Source]
