"""PDF replacement tests: vector reuse, stale-evidence removal and rollback."""
import json
import numpy as np
from telecom_support.ingestion import additions
from telecom_support.ingestion.schemas import TextBlock, KBChunk
from telecom_support.ingestion.metadata import MetadataResult
from telecom_support.indexing.records import load_records
from test_additions import jobs


def long_text(label):
    return label+' '+ ' '.join(f'{label}{i}' for i in range(220))


def prepare_document(jobs, monkeypatch):
    blocks = [TextBlock(text=long_text(name), pages=[number]) for number,name in enumerate(['obsolete','unchanged','removed'], 1)]
    calls = []
    monkeypatch.setattr(additions, 'extract_pdf', lambda path:(blocks, {'pages':max(b.pages[0] for b in blocks)}))
    def classify(client, chunk, model):
        calls.append(chunk.text)
        return MetadataResult(title='Broadband checks', categories=['Slow speed'], products=['Fiber Broadband & Gateway'])
    monkeypatch.setattr(additions, 'classify_chunk', classify)
    first = jobs.submit('pdf', b'%PDF-first-version', filename='guide.pdf')
    jobs.close()
    assert jobs.status(first['job_id'])['status'] == 'completed'
    doc = next(d for d in jobs.documents() if d['name']=='guide.pdf')
    return blocks, calls, doc


def test_replace_reuses_only_unchanged_text_and_removes_old_version(jobs, monkeypatch):
    blocks, calls, document = prepare_document(jobs, monkeypatch)
    doc_id = document['document_id']
    previous = [r for r in load_records(jobs.root/'data/indexes') if isinstance(r, KBChunk) and r.document_id==doc_id]
    old_source_ids = {r.source_id for r in previous}
    old_passage_ids = {p['passage_id'] for p in jobs.state.index.rows if p['source_id'] in old_source_ids}
    old_vector = jobs.state.index.client.retrieve('support_passages', ids=[next(p['passage_id'] for p in jobs.state.index.rows
        if p['text']==long_text('unchanged'))], with_vectors=True)[0].vector
    blocks[:] = [TextBlock(text=long_text('unchanged'), pages=[7]), TextBlock(text=long_text('updated'), pages=[8])]
    calls.clear()
    embedded = []
    def encode(texts):
        embedded.extend(texts)
        return np.tile([1.,0.,0.], (len(texts),1))
    jobs.state.index.embedder.encode = encode
    submitted = jobs.submit('pdf', b'%PDF-second-version', replacement_id=doc_id, filename='guide-updated.pdf')
    jobs.close()
    status = jobs.status(submitted['job_id'])
    assert status['status'] == 'completed'
    assert (status['unchanged_chunks'], status['changed_or_new_chunks'], status['removed_chunks']) == (1,1,2)
    assert status['embedded_unique_texts'] == 1
    assert embedded == calls == [long_text('updated')]
    current = load_records(jobs.root/'data/indexes')
    updated = [r for r in current if isinstance(r, KBChunk) and r.document_id==doc_id]
    assert len(updated)==2 and all(r.source_id not in old_source_ids for r in updated)
    assert next(r for r in updated if r.text==long_text('unchanged')).pages==[7]
    assert any(r.source_id=='KB-1' for r in current) and any(r.source_id=='T-1' for r in current)
    assert not old_passage_ids & {p['passage_id'] for p in jobs.state.index.rows}
    point = next(p for p in jobs.state.index.rows if p['text']==long_text('unchanged'))
    new_vector = jobs.state.index.client.retrieve('support_passages', ids=[point['passage_id']], with_vectors=True)[0].vector
    assert np.allclose(old_vector, new_vector)
    assert next(d for d in jobs.documents() if d['document_id']==doc_id)['name']=='guide-updated.pdf'


def test_identical_replacement_is_noop_and_failed_write_preserves_current(jobs, monkeypatch):
    blocks, calls, document = prepare_document(jobs, monkeypatch)
    calls.clear()
    old_index = jobs.state.index
    old_active = json.loads((jobs.root/'data/indexes/search/active.json').read_text())
    submitted = jobs.submit('pdf', b'%PDF-first-version', replacement_id=document['document_id'])
    jobs.close()
    assert jobs.status(submitted['job_id'])['no_changes']
    assert not calls and jobs.state.index is old_index
    assert len(load_records(jobs.root/'data/indexes'))==5
    blocks[:] = [TextBlock(text=long_text('failedchange'), pages=[1])]
    def fail(*args):
        raise RuntimeError('Cannot publish vector store')
    monkeypatch.setattr(additions, 'write_vectors', fail)
    failed = jobs.submit('pdf', b'%PDF-failing-version', replacement_id=document['document_id'])
    jobs.close()
    assert jobs.status(failed['job_id'])['status']=='failed'
    assert jobs.state.index is old_index
    assert json.loads((jobs.root/'data/indexes/search/active.json').read_text())==old_active
    assert any(r.text==long_text('obsolete') for r in load_records(jobs.root/'data/indexes') if isinstance(r,KBChunk))


def test_replace_original_prepared_pdf_and_document_api(jobs, monkeypatch):
    from fastapi.testclient import TestClient
    from telecom_support.api import app
    monkeypatch.setattr(app.state, 'jobs', jobs, raising=False)
    client = TestClient(app)
    assert client.get('/ingestion/documents').json()[0]['document_id']=='doc'
    assert client.post('/ingestion/pdfs?replacement_id=absent', content=b'%PDF-', headers={'Content-Type':'application/pdf'}).status_code==404
    monkeypatch.setattr(additions, 'extract_pdf', lambda path:([TextBlock(text='Updated cable safety checks.', pages=[2])], {'pages':2}))
    monkeypatch.setattr(additions, 'classify_chunk', lambda *args:MetadataResult(title='Cable safety', categories=[], products=[]))
    response = client.post('/ingestion/pdfs?replacement_id=doc&filename=manual.pdf', content=b'%PDF-original-replaced', headers={'Content-Type':'application/pdf'})
    assert response.status_code==202
    jobs.close()
    assert jobs.status(response.json()['job_id'])['status']=='completed'
    records = load_records(jobs.root/'data/indexes')
    assert len(records)==2 and all(r.source_id!='KB-1' for r in records)
    assert next(r for r in records if isinstance(r, KBChunk)).document_id=='doc'
