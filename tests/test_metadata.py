"""Metadata checks using simulated responses only."""
import json
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from telecom_support.ingestion.schemas import KBChunk
from telecom_support.ingestion.metadata import (
    MetadataResult, cache_key, classify_chunk, enrich_chunk, load_cache, save_cache,
)


def chunk():
    return KBChunk(source_id="kb-1", source_file="guide.pdf", source_version="version",
                   taxonomy_version="2", search_text="Check broadband", document_id="doc",
                   chunk_number=1, title=None, pages=[5], text="Check broadband",
                   token_count=6, chunking_version="test")


def result(**changes):
    data = dict(title="Broadband diagnostic checks", categories=["Broadband intermittent drops"], products=["Fiber Broadband & Gateway"])
    data.update(changes)
    return MetadataResult(**data)


def test_enrichment_preserves_evidence():
    original = chunk()
    enriched = enrich_chunk(original, result(), "model", "key")
    assert enriched.text == original.text and enriched.pages == [5]
    assert enriched.source_id == original.source_id
    assert enriched.title == "Broadband diagnostic checks"
    assert enriched.search_text == original.search_text
    assert original.metadata_status == "pending"


def test_general_guidance_has_empty_labels():
    general = result(categories=[], products=[], title="General safety")
    enriched = enrich_chunk(chunk(), general, "model", "key")
    assert enriched.categories == [] and enriched.products == []
    assert "retrieval_eligible" not in enriched.model_dump()
    assert "review_required" not in enriched.model_dump()


def test_invalid_labels_and_duplicate_labels():
    with pytest.raises(ValidationError): result(categories=["invented"])
    with pytest.raises(ValidationError): result(title="   ")
    with pytest.raises(ValidationError): result(products=["Fiber Broadband & Gateway"] * 2)


def test_cache_roundtrip_and_invalidation(tmp_path, monkeypatch):
    c = chunk()
    key = cache_key(c, "model")
    path = tmp_path / "cache.jsonl"
    save_cache(path, {key: result()})
    assert load_cache(path)[key] == result()
    assert cache_key(c.model_copy(update={"text": "changed"}), "model") != key
    assert cache_key(c, "other-model") != key
    import telecom_support.ingestion.metadata as module
    monkeypatch.setattr(module, "PROMPT_VERSION", "new")
    assert cache_key(c, "model") != key
    monkeypatch.setitem(module.TAXONOMY, "version", "changed")
    assert cache_key(c, "model") != key


def test_simulated_api_and_incomplete_response():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=result().model_dump_json()))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert classify_chunk(client, chunk(), "model") == result()
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    client.chat.completions.create = lambda **kwargs: SimpleNamespace(choices=[])
    with pytest.raises(ValueError): classify_chunk(client, chunk(), "model")
