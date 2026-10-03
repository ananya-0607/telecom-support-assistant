"""Token-bounded blocks with page-preserving overlap."""
import hashlib
import re
from .schemas import KBChunk, TextBlock

TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 240
OVERLAP_TOKENS = 32
CHUNKING_VERSION = "text-token-v2"

def load_tokenizer(cache_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, cache_dir=str(cache_dir), use_fast=True)

def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))

def document_identity(relative_path):
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]

def chunk_blocks(blocks, tokenizer, document_id, source_file, source_version, taxonomy_version,
                 max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    if not 0 <= overlap_tokens < max_tokens // 2:
        raise ValueError("Overlap must be less than half the token budget.")
    chunks, current = [], []
    def render(parts):
        body = "\n\n".join(part.text for part in parts)
        return body, body
    def emit():
        if not current:
            return
        body, search = render(current)
        number = len(chunks) + 1
        processing_id = hashlib.sha256(
            f"{CHUNKING_VERSION}:{TOKENIZER_MODEL}:{max_tokens}:{overlap_tokens}:{search}".encode("utf-8")
        ).hexdigest()[:12]
        chunks.append(KBChunk(
            source_id=f"KB-{document_id}-{source_version[:16]}-{number:04d}-{processing_id}",
            document_id=document_id, source_file=source_file, source_version=source_version,
            taxonomy_version=taxonomy_version, chunk_number=number,
            pages=sorted({p for b in current for p in b.pages}), text=body, search_text=search,
            token_count=token_count(tokenizer, search),
            chunking_version=f"{CHUNKING_VERSION}:{max_tokens}:{overlap_tokens}"))
    def split_block(block, budget):
        if token_count(tokenizer, block.text) <= budget:
            return [block]
        parts = []
        for sentence in re.split(r"(?<=[.!?])\s+", block.text):
            if token_count(tokenizer, sentence) <= budget:
                parts.append(TextBlock(text=sentence, pages=block.pages))
                continue
            offsets = tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            start = 0
            while start < len(offsets):
                end = min(start + max(1, budget - 2), len(offsets))
                fragment = sentence[offsets[start][0]:offsets[end-1][1]].strip()
                while end > start + 1 and token_count(tokenizer, fragment) > budget:
                    end -= 1
                    fragment = sentence[offsets[start][0]:offsets[end-1][1]].strip()
                if token_count(tokenizer, fragment) > budget:
                    raise ValueError("Cannot fit source text within token budget.")
                parts.append(TextBlock(text=fragment, pages=block.pages))
                start = end
        return parts
    for block in blocks:
        budget = max_tokens - 4
        for part in split_block(block, budget):
            if current and token_count(tokenizer, render(current + [part])[1]) > max_tokens:
                emit()
                tail = []
                for previous in reversed(current):
                    candidate = [previous] + tail
                    if token_count(tokenizer, "\n\n".join(p.text for p in candidate)) > overlap_tokens:
                        break
                    tail = candidate
                current = tail
                if current and token_count(tokenizer, render(current + [part])[1]) > max_tokens:
                    current = []
            current.append(part)
    emit()
    if any(c.token_count > max_tokens for c in chunks):
        raise ValueError("Chunk exceeds token budget.")
    return chunks
