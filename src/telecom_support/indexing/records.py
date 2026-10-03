"""Validate prepared records and create token-bounded searchable passages."""
import uuid
from pathlib import Path
from telecom_support.ingestion.schemas import TicketRecord, KBChunk, TextBlock
from telecom_support.ingestion.chunking import chunk_blocks

def load_records(index_dir):
    records = []
    for filename, schema in [("historical_tickets.jsonl", TicketRecord),
                             ("kb_sections_enriched.jsonl", KBChunk)]:
        path = Path(index_dir) / filename
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            try:
                record = schema.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"Invalid record in {filename}, line {number}") from exc
            if isinstance(record, TicketRecord) and record.split != "train":
                raise ValueError("Evaluation tickets must not enter the search index.")
            if isinstance(record, KBChunk) and record.metadata_status == "pending":
                raise ValueError("Finish KB classification before building embeddings.")
            records.append(record)
        if not lines:
            raise ValueError(f"Empty source file: {filename}")
    if len({r.source_id for r in records}) != len(records):
        raise ValueError("Duplicate source IDs in prepared records.")
    return records

def make_passages(records, tokenizer):
    passages = []
    for record in records:
        ticket = isinstance(record, TicketRecord)
        # Resolution columns and predicted KB titles are deliberately excluded.
        text = record.conversation if ticket else record.text
        if not text.strip():
            raise ValueError(f"Empty embedding text: {record.source_id}")
        pieces = chunk_blocks(
            [TextBlock(text=text, pages=[] if ticket else record.pages)], tokenizer,
            record.source_id, record.source_file, record.source_version,
            record.taxonomy_version)
        for number, piece in enumerate(pieces, 1):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                f"telecom-passage-v1:{record.source_id}:{number}:{piece.text}"))
            passages.append({"passage_id": point_id, "source_id": record.source_id,
                "source_type": record.source_type, "text": piece.text,
                "token_count": piece.token_count,
                "categories": [record.category.value] if ticket else [c.value for c in record.categories],
                "products": [record.product.value] if ticket else [p.value for p in record.products]})
    return passages
