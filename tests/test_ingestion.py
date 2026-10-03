"""Behavior checks without model downloads or API calls."""
import json
import re
from pathlib import Path
import pandas as pd
import pytest
from telecom_support.ingestion.chunking import chunk_blocks, document_identity
from telecom_support.ingestion.excel import prepare_excel
from telecom_support.ingestion.pdf import extract_pdf
from telecom_support.ingestion.schemas import TextBlock

class WordTokenizer:
    def encode(self, text, add_special_tokens=True, truncation=False):
        return list(range(len(text.split()) + (2 if add_special_tokens else 0)))
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}

def chunks(blocks):
    return chunk_blocks(blocks, WordTokenizer(), "doc", "manual.pdf", "a"*64, "1", max_tokens=48, overlap_tokens=8)

def test_plain_blocks_combine_without_section_boundaries():
    blocks = [TextBlock(text="Check service.", pages=[1]),
              TextBlock(text="Check bill.", pages=[2]),
              TextBlock(text="Verify result.", pages=[3])]
    result = chunks(blocks)
    assert len(result) == 1
    assert result[0].pages == [1, 2, 3]
    assert [c.source_id for c in result] == [c.source_id for c in chunks(blocks)]
    assert all(c.metadata_status == "pending" and not c.categories for c in result)

def test_cross_page_and_long_text():
    result = chunks([TextBlock(text="Check wired service.", pages=[1]), TextBlock(text="Compare Wi-Fi.", pages=[2])])
    assert result[0].pages == [1, 2]
    text = " ".join(f"word{i}" for i in range(150))
    result = chunks([TextBlock(text=text, pages=[7])])
    assert all(c.token_count <= 48 and c.pages == [7] for c in result)
    assert " ".join(c.text for c in result).split() == text.split()

def test_heading_budget_and_identity():
    blocks = [TextBlock(text=" ".join([f"item{i}"]*10)+".", pages=[i+1]) for i in range(10)]
    assert all(c.token_count <= 48 for c in chunks(blocks))
    assert document_identity("a.pdf") != document_identity("b.pdf")

def test_overlap_is_bounded_and_new_content_is_not_lost():
    blocks = [TextBlock(text=f"Check item{i} now.", pages=[i+1]) for i in range(30)]
    result = chunks(blocks)
    assert len(result) > 1
    for left, right in zip(result, result[1:]):
        left_blocks, right_blocks = left.text.split("\n\n"), right.text.split("\n\n")
        shared = [text for text in right_blocks if text in left_blocks]
        assert shared and left_blocks[-len(shared):] == right_blocks[:len(shared)]
        assert len(" ".join(shared).split()) <= 8
    assert all(block.text in "\n\n".join(c.text for c in result) for block in blocks)

def test_pdf_without_headings_and_blank_page(monkeypatch):
    class Page:
        def __init__(self, text): self.text = text
        def extract_text(self): return self.text
    class Reader:
        is_encrypted = False
        pages = [Page("Page 1\nGeneral troubleshooting.\nCheck service first."), Page("Page 2\nVerify result.")]
    monkeypatch.setattr("telecom_support.ingestion.pdf.PdfReader", lambda path: Reader())
    blocks, details = extract_pdf(Path("manual.pdf"))
    assert details["pages"] == 2
    assert "Check service first." in " ".join(b.text for b in blocks)
    Reader.pages = [Page("")]
    with pytest.raises(ValueError, match="OCR"):
        extract_pdf(Path("scanned.pdf"))

def test_ticket_split_and_invalid_labels(tmp_path):
    taxonomy = json.loads((Path(__file__).parents[1]/"config/taxonomy.json").read_text())
    base = dict(created_at="2026-01-01", category="Slow speed", product="Fiber Broadband & Gateway",
                severity="Medium", sentiment="Neutral", conversation="Customer and agent",
                resolution_steps="Check connection", resolution_summary="Verified")
    rows = [dict(base, ticket_id="T-1", customer_complaint="Slow downloads", split="train"),
            dict(base, ticket_id="T-2", customer_complaint="Slow uploads", split="test")]
    path = tmp_path/"tickets.xlsx"
    pd.DataFrame(rows).to_excel(path, sheet_name="Tickets", index=False)
    train, test = prepare_excel(path, "tickets.xlsx", taxonomy)
    assert {r.ticket_id for r in train} == {"T-1"}
    assert {r.ticket_id for r in test} == {"T-2"}
    rows[1]["split"] = "invalid"
    pd.DataFrame(rows).to_excel(path, sheet_name="Tickets", index=False)
    with pytest.raises(ValueError):
        prepare_excel(path, "tickets.xlsx", taxonomy)
