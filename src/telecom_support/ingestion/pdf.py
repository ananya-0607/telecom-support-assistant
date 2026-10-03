"""Text PDF extraction; headings are hints, not unique identities."""
import re
from collections import Counter
from pypdf import PdfReader
from .schemas import TextBlock

def heading_line(text):
    return len(text) <= 120 and bool(
        re.match(r"^(?:\d+(?:\.\d+)+\s+\S|Chapter\s+\d+\b|Section\s+\d+\b)", text, re.I)
        or (len(text.split()) <= 12 and text.isupper() and any(c.isalpha() for c in text)))

def extract_pdf(path):
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError(f"Encrypted PDF: {path.name}")
    if not reader.pages:
        raise ValueError(f"Empty PDF: {path.name}")
    raw_pages = [page.extract_text() or "" for page in reader.pages]
    lines_by_page = [[re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()] for text in raw_pages]
    edges = Counter()
    for lines in lines_by_page:
        edges.update(set(lines[:2] + lines[-2:]))
    repeated = {line for line, count in edges.items() if len(raw_pages) >= 3
                and count >= max(3, int(len(raw_pages) * .6 + .999)) and not heading_line(line)}
    blocks, warnings = [], []
    heading = None
    for number, (raw, lines) in enumerate(zip(raw_pages, lines_by_page), 1):
        if not raw.strip():
            raise ValueError(f"No text on {path.name} page {number}; blank/image pages need review or OCR.")
        if "\ufffd" in raw:
            warnings.append(f"Page {number}: replacement characters; inspect extraction.")
        if len(raw.strip()) < 40:
            warnings.append(f"Page {number}: very little text; inspect extraction.")
        paragraph = []
        def flush():
            if paragraph:
                blocks.append(TextBlock(text=" ".join(paragraph), pages=[number], heading=heading))
                paragraph.clear()
        for index, line in enumerate(lines):
            edge = index < 2 or index >= len(lines) - 2
            if edge and re.fullmatch(r"(?:Page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?", line, re.I):
                continue
            if edge and line in repeated:
                continue
            if heading_line(line):
                flush()
                heading = line
                continue
            if re.match(r"^(?:\d+[.)]\s+|[\u2022\u25cf*-]\s+)", line):
                flush()
            paragraph.append(line)
            if re.search(r"[.!?:]$", line):
                flush()
        flush()
    if not blocks:
        raise ValueError(f"No usable text extracted from {path.name}.")
    return blocks, {"pages": len(raw_pages), "page_text_characters": [len(t.strip()) for t in raw_pages],
                    "warnings": warnings, "extraction_method": "pypdf plain text; complex layouts require review"}
