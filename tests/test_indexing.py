"""Offline checks for evidence integrity, holdout exclusion and token coverage."""
import sqlite3
import pytest
from telecom_support.indexing.records import load_records, make_passages
from telecom_support.indexing.storage import write_database, fetch_evidence
from telecom_support.ingestion.schemas import TicketRecord, KBChunk

def sources():
    common = dict(source_file='example', source_version='v1', taxonomy_version='2', search_text='unused')
    ticket = TicketRecord(**common, source_id='T-1', ticket_id='T-1', created_at='2026-01-01',
        category='Slow speed', product='Fiber Broadband & Gateway', severity='Low',
        sentiment='Neutral', customer_complaint='slow', conversation='Customer reports slow internet.',
        resolution_steps='SECRET RESOLUTION COLUMN', resolution_summary='fixed', split='train')
    kb = KBChunk(**common, source_id='KB-1', document_id='doc', chunk_number=1,
        pages=[1], text='Check the cable.', token_count=6, chunking_version='test',
        title='Invented title excluded from embedding', metadata_status='classified')
    return ticket, kb

def save_sources(path, ticket, kb):
    (path/'historical_tickets.jsonl').write_text(ticket.model_dump_json()+'\n', encoding='utf-8')
    (path/'kb_sections_enriched.jsonl').write_text(kb.model_dump_json()+'\n', encoding='utf-8')

class WordTokenizer:
    def encode(self, text, add_special_tokens=True, truncation=False):
        return list(range(len(text.split()) + (2 if add_special_tokens else 0)))
    def __call__(self, text, **kwargs):
        import re
        return {'offset_mapping': [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]}

def test_holdout_and_pending_rejected(tmp_path):
    ticket, kb = sources()
    save_sources(tmp_path, ticket.model_copy(update={'split': 'test'}), kb)
    with pytest.raises(ValueError, match='Evaluation'):
        load_records(tmp_path)
    save_sources(tmp_path, ticket, kb.model_copy(update={'metadata_status': 'pending'}))
    with pytest.raises(ValueError, match='classification'):
        load_records(tmp_path)

def test_embedding_text_and_sql_join(tmp_path):
    ticket, kb = sources()
    passages = make_passages([ticket, kb], WordTokenizer())
    assert passages[0]['text'] == ticket.conversation
    assert passages[1]['text'] == kb.text
    assert all('SECRET' not in p['text'] and 'Invented' not in p['text'] for p in passages)
    assert passages[1]['categories'] == []  # General guidance remains indexed.
    path = tmp_path/'evidence.sqlite3'
    write_database(path, [ticket, kb], passages, {'model':'test'})
    result = fetch_evidence(path, passages[0]['passage_id'])
    assert result['source']['resolution_steps'] == ticket.resolution_steps
    assert result['passage_text'] == ticket.conversation
    with pytest.raises(KeyError):
        fetch_evidence(path, 'missing')
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT count(*) FROM sources').fetchone()[0] == 2

def test_long_conversation_coverage_and_stable_ids():
    ticket, _ = sources()
    words = [f'w{i}' for i in range(700)]
    ticket = ticket.model_copy(update={'conversation': ' '.join(words)})
    passages = make_passages([ticket], WordTokenizer())
    assert len(passages) > 1
    assert all(p['token_count'] <= 240 and p['source_id'] == ticket.source_id for p in passages)
    assert set(' '.join(p['text'] for p in passages).split()) == set(words)
    assert passages == make_passages([ticket], WordTokenizer())

def test_missing_sql_parent_rolls_back(tmp_path):
    ticket, kb = sources()
    passages = make_passages([ticket, kb], WordTokenizer())
    with pytest.raises(sqlite3.IntegrityError):
        write_database(tmp_path/'bad.sqlite3', [ticket], passages, {})
