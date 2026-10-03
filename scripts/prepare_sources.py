"""Prepare Excel and all KB PDFs. No LLM calls or vector-index updates.

Run: python scripts/prepare_sources.py
First run downloads MiniLM tokenizer files, not model weights.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def save_jsonl(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    temporary.replace(path)


def main():
    try:
        from telecom_support.ingestion.chunking import (
            CHUNKING_VERSION, MAX_TOKENS, OVERLAP_TOKENS, TOKENIZER_MODEL,
            chunk_blocks, document_identity, load_tokenizer,
        )
        from telecom_support.ingestion.excel import prepare_excel
        from telecom_support.ingestion.pdf import extract_pdf
        from telecom_support.taxonomy import TAXONOMY
        taxonomy = TAXONOMY
        tickets_path = ROOT / "data/processed/conversations.xlsx"
        training, evaluation = prepare_excel(tickets_path, tickets_path.relative_to(ROOT).as_posix(), taxonomy)
        pdfs = sorted(p for p in (ROOT / "data/knowledge_base").rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
        if not pdfs:
            raise ValueError("No PDFs found in data/knowledge_base.")
        print("Loading MiniLM tokenizer (first run may download tokenizer files)...", flush=True)
        output = ROOT / "data/indexes"
        tokenizer = load_tokenizer(output / "tokenizer_cache")
        records, documents = [], []
        for path in pdfs:
            relative = path.relative_to(ROOT).as_posix()
            version = hashlib.sha256(path.read_bytes()).hexdigest()
            blocks, details = extract_pdf(path)
            prepared = chunk_blocks(blocks, tokenizer, document_identity(relative), relative, version, taxonomy["version"])
            if not prepared:
                raise ValueError(f"No chunks generated for {relative}")
            records.extend(prepared)
            documents.append({"source_file": relative, "source_version": version, "chunks": len(prepared), **details})
            print(f"{path.name}: {details['pages']} pages, {len(prepared)} chunks")
        summary = {
            "schema_version": 2, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "taxonomy_version": taxonomy["version"],
            "training_tickets": len(training), "evaluation_tickets": len(evaluation),
            "total_tickets": len(training) + len(evaluation),
            "kb_documents": len(documents), "kb_pages": sum(d["pages"] for d in documents),
            "kb_chunks": len(records), "documents": documents,
            "ticket_source_version": hashlib.sha256(tickets_path.read_bytes()).hexdigest(),
            "chunking": {"version": CHUNKING_VERSION, "tokenizer": TOKENIZER_MODEL,
                         "max_tokens_including_heading_and_special_tokens": MAX_TOKENS,
                         "overlap_tokens_max": OVERLAP_TOKENS},
            "notes": [
                "KB category/product metadata is pending; no LLM calls made.",
                "Evaluation tickets must not enter retrieval or few-shot prompts.",
                "No embeddings, incremental index updates, or source manifest yet.",
                "Cover/contents/reference text retained; inspect and curate before indexing.",
                "Text-PDF extraction does not verify complex tables or column reading order.",
            ],
        }
        output.mkdir(parents=True, exist_ok=True)
        save_jsonl(output / "historical_tickets.jsonl", training)
        save_jsonl(output / "evaluation_tickets.jsonl", evaluation)
        save_jsonl(output / "kb_sections.jsonl", records)
        (output / "ingestion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except ImportError:
        print("Missing dependencies. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Historical tickets: {len(training)}; evaluation tickets: {len(evaluation)}")
    print(f"KB chunks: {len(records)}; category/product metadata: pending")
    print(f"Saved to {output}")
    print("PASS: Preparation complete. No source files modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
