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

class KBChunk(Record):
    source_type: Literal["kb"] = "kb"
    document_id: str
    chunk_number: int = Field(ge=1)
    section_heading: str | None
    pages: list[int]
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    categories: list[Category] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    metadata_status: Literal["pending", "classified", "reviewed"] = "pending"
    metadata_scope: Literal["unknown", "general", "specific"] = "unknown"
    chunking_version: str

class TextBlock(BaseModel):
    text: str
    pages: list[int]
    heading: str | None = None
