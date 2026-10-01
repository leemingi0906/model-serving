"""3. Pydantic으로 스키마 정의 (HAIC 프로젝트의 schemas.py와 같은 자리)"""
from pydantic import BaseModel, Field


class Tag(BaseModel):
    name: str = Field(..., min_length=1)


class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=50)
    content: str = Field(..., min_length=1)
    tags: list[Tag] = Field(default_factory=list)


class NoteResponse(BaseModel):
    id: int
    title: str
    content: str
    tag_count: int
