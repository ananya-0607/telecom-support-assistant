"""Validated contracts for prepared evidence."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from telecom_support.taxonomy import Category, Product, Severity, Sentiment

class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    source_file: str
    source_version: str
    taxonomy_version: str
    search_text: str = Field(min_length=1)

class TicketRecord(Record):
    source_type: Literal["ticket"] = "ticket"
    ticket_id: str
    created_at: str
    category: Category
    product: Product
    severity: Severity
    sentiment: Sentiment
    customer_complaint: str
    conversation: str
    resolution_steps: str
    resolution_summary: str
    split: Literal["train", "test"]
    origin: Literal['historical', 'agent_reviewed_generated'] = 'historical'
    generated_response: dict | None = None

class KBChunk(Record):
    source_type: Literal["kb"] = "kb"
    document_id: str
    document_name: str | None = None
    content_hash: str | None = Field(default=None, min_length=64, max_length=64)
    chunk_number: int = Field(ge=1)
    title: str | None = None
    pages: list[int]
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    categories: list[Category] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    metadata_status: Literal["pending", "classified", "reviewed"] = "pending"
    chunking_version: str
    metadata_model: str | None = None
    metadata_prompt_version: str | None = None
    metadata_cache_key: str | None = None

class TextBlock(BaseModel):
    text: str
    pages: list[int]
