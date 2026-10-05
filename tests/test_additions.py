"""Offline publication/rollback tests with real SQLite and local Qdrant."""
import json
import threading
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from telecom_support.ingestion import additions
from telecom_support.ingestion.additions import AdditionJobs, ResolvedConversation, atomic_json
from telecom_support.ingestion.schemas import TextBlock
from telecom_support.ingestion.metadata import MetadataResult
from telecom_support.indexing.storage import write_database
from telecom_support.indexing.records import make_passages, load_records
from telecom_support.ingestion.chunking import TOKENIZER_MODEL
from telecom_support.taxonomy import TAXONOMY, activate_taxonomy, staged_taxonomy, Category
from telecom_support.ingestion.pdf_labels import PDFLabels, MissingLabelSuggestion, expanded_taxonomy
from telecom_support.indexing.vectors import COLLECTION
from test_indexing import WordTokenizer, sources, save_sources
from test_resolution_flow import labels


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    original_taxonomy = copy.deepcopy(TAXONOMY)
    base = tmp_path/'data/indexes'
    path = base/'search/builds'/('a'*32)
    path.mkdir(parents=True)
    ticket, kb = sources()
    save_sources(base, ticket, kb)
    embedder = SimpleNamespace(tokenizer=WordTokenizer(), model=SimpleNamespace(max_seq_length=256),
        encode=lambda texts: np.tile([1., 0., 0.], (len(texts), 1)))
    write_database(path/'evidence.sqlite3', [ticket, kb], make_passages([ticket, kb], embedder.tokenizer), {})
    atomic_json(path/'manifest.json', dict(model=TOKENIZER_MODEL, collection=COLLECTION))
    atomic_json(base/'search/active.json', dict(build_id='a'*32))
    old = SimpleNamespace(database=path/'evidence.sqlite3', path=path, embedder=embedder, close=lambda: None)
    state = SimpleNamespace(lock=threading.Lock(), index=old,
        pipeline=SimpleNamespace(index=old, llm=SimpleNamespace(model='fake', client=None), examples=[]))
    monkeypatch.setattr(additions, 'classify', lambda *args: labels())
    service = AdditionJobs(tmp_path, state, delay=0)
    yield service
    service.close()
    state.index.close()
    activate_taxonomy(original_taxonomy)


def conversation():
    return ResolvedConversation(customer_complaint='Slow broadband', conversation='Customer reports slow throughput after cable damage.',
        resolution_steps='Replace damaged cable after confirming fault.', resolution_summary='Verified working connection.')


def test_ticket_publication_and_cli_rebuild_retains_addition(jobs):
    submitted = jobs.submit('ticket', conversation())
    jobs.close()
    status = jobs.status(submitted['job_id'])
    assert status['status'] == 'completed'
    assert status['added_sources'] == 1 and status['added_passages'] == 1
    assert jobs.state.index is jobs.state.pipeline.index
    assert status['ticket_id'] == 'T-0002'
    assert any(r['source_id'] == status['ticket_id'] for r in jobs.state.index.rows)
    records = load_records(jobs.root/'data/indexes')
    added = [r for r in records if r.source_id == status['ticket_id']]
    assert len(added) == 1 and added[0].resolution_steps == conversation().resolution_steps
    assert not jobs.state.lock.locked()


def test_failed_vector_build_preserves_old_search(jobs, monkeypatch):
    old = jobs.state.index
    def fail(*args):
        raise RuntimeError('Vector build failed')
    monkeypatch.setattr(additions, 'write_vectors', fail)
    submitted = jobs.submit('ticket', conversation())
    jobs.close()
    assert jobs.status(submitted['job_id'])['status'] == 'failed'
    assert jobs.state.index is old
    assert json.loads((jobs.root/'data/indexes/search/active.json').read_text())['build_id'] == 'a'*32
    assert len(load_records(jobs.root/'data/indexes')) == 2
    assert not jobs.state.lock.locked()


def test_pdf_reuses_chunking_and_metadata(jobs, monkeypatch):
    monkeypatch.setattr(additions, 'extract_pdf', lambda path: ([TextBlock(text='Check damaged broadband cables.', pages=[1])], {'pages':1}))
    monkeypatch.setattr(additions, 'classify_chunk', lambda *args: MetadataResult(title='Cable checks',
        categories=['Router / modem hardware'], products=['Fiber Broadband & Gateway']))
    submitted = jobs.submit('pdf', b'%PDF-1.4 fake offline fixture')
    jobs.close()
    status = jobs.status(submitted['job_id'])
    assert status['status'] == 'completed'
    assert status['chunks_completed'] == status['chunks_total'] == 1
    added = [r for r in load_records(jobs.root/'data/indexes') if r.source_file.endswith('upload.pdf')]
    assert len(added) == 1 and added[0].title == 'Cable checks' and added[0].pages == [1]


def test_busy_blank_and_unknown_job(jobs):
    with pytest.raises(ValueError):
        ResolvedConversation(**{**conversation().model_dump(), 'conversation':'  '})
    jobs.state.lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='running'):
            jobs.submit('ticket', conversation())
    finally:
        jobs.state.lock.release()
    with pytest.raises(FileNotFoundError):
        jobs.status('../invalid')


def test_upload_validation_and_job_endpoint(jobs, monkeypatch):
    from fastapi.testclient import TestClient
    from telecom_support.api import app
    monkeypatch.setattr(app.state, 'jobs', jobs, raising=False)
    client = TestClient(app)
    assert client.post('/ingestion/pdfs', content=b'not pdf', headers={'Content-Type':'application/pdf'}).status_code == 400
    assert client.post('/ingestion/pdfs', content=b'%PDF-', headers={'Content-Type':'text/plain'}).status_code == 415
    assert client.post('/ingestion/pdfs', content=b'%PDF-'+b'a'*(10*1024*1024), headers={'Content-Type':'application/pdf'}).status_code == 413
    assert client.get('/ingestion/jobs/'+'b'*32).status_code == 404
    response = client.post('/ingestion/conversations', json=conversation().model_dump())
    assert response.status_code == 202
    jobs.close()
    status = client.get('/ingestion/jobs/'+response.json()['job_id'])
    assert status.json()['status'] == 'completed'


def test_generated_labels_reused_and_ids_increment(jobs, monkeypatch):
    def no_call(*args):
        raise AssertionError('Existing classification must be reused')
    monkeypatch.setattr(additions, 'classify', no_call)
    full_response = {'classification':labels().model_dump(mode='json'), 'resolution':{
        'summary':'Cable fault', 'steps':[{'instruction':'Check cable', 'citations':['S1']}],
        'missing_information':['Confirm cable condition'], 'escalation':'Hardware team'},
        'sources':[{'citation_id':'S1', 'source':{'source_id':'KB-1'}}]}
    payload = ResolvedConversation(**conversation().model_dump(exclude={'classification','reviewed','generated_response'}),
        classification=labels(), reviewed=True, generated_response=full_response)
    first = jobs.submit('ticket', payload)
    jobs.close()
    second = jobs.submit('ticket', payload.model_copy(update={'conversation':'Customer: another problem. Agent: reviewed cable checks.'}))
    jobs.close()
    assert jobs.status(first['job_id'])['ticket_id'] == 'T-0002'
    assert jobs.status(second['job_id'])['ticket_id'] == 'T-0003'
    generated = [r for r in load_records(jobs.root/'data/indexes') if r.source_type == 'ticket' and r.origin == 'agent_reviewed_generated']
    assert len(generated) == 2
    assert all(r.generated_response == full_response for r in generated)
    with pytest.raises(ValueError, match='Review'):
        ResolvedConversation(**payload.model_dump(exclude={'reviewed'}), reviewed=False)


def test_ticket_ids_reserve_evaluation_ids(jobs):
    path = jobs.root/'data/indexes/evaluation_tickets.jsonl'
    path.write_text(json.dumps({'ticket_id':'T-0200'})+'\n')
    assert additions.next_ticket_id(jobs.state.index.database, path) == 'T-0201'


def pdf_fixture(monkeypatch):
    monkeypatch.setattr(additions, 'extract_pdf', lambda path: ([TextBlock(text='IPTV picture freezes. Check the TV receiver connection.', pages=[1])], {'pages':1}))
    def metadata(*args):
        return MetadataResult(title='IPTV receiver checks', categories=['IPTV playback issues'], products=['IPTV & TV Receiver'])
    monkeypatch.setattr(additions, 'classify_chunk', metadata)


def new_pdf_labels():
    return PDFLabels(category='IPTV playback issues', category_description='TV service freezes or cannot play channels.',
        product='IPTV & TV Receiver', product_description='Telecom IPTV and television receivers.')


def test_new_pdf_labels_activate_schema_and_persist(jobs, monkeypatch):
    pdf_fixture(monkeypatch)
    old_version = TAXONOMY['version']
    submitted = jobs.submit('pdf', b'%PDF-test new IPTV', new_pdf_labels())
    jobs.close()
    assert jobs.status(submitted['job_id'])['status'] == 'completed'
    assert 'IPTV playback issues' in TAXONOMY['categories']
    assert TAXONOMY['version'] != old_version
    assert labels().model_json_schema()['$defs']['Category']['enum'][-1] == 'IPTV playback issues'
    from telecom_support.classification.schemas import ComplaintLabels
    assert ComplaintLabels(categories=['IPTV playback issues'], products=['IPTV & TV Receiver'], severity='Low', sentiment='Neutral')
    active = json.loads((jobs.root/'data/indexes/search/active.json').read_text())
    assert active['taxonomy'] == json.loads((jobs.root/'config/taxonomy.json').read_text()) == TAXONOMY
    added = [r for r in load_records(jobs.root/'data/indexes') if r.source_file.endswith('upload.pdf')]
    assert added[0].categories[0].value == 'IPTV playback issues'


def test_new_labels_rollback_on_vector_failure(jobs, monkeypatch):
    pdf_fixture(monkeypatch)
    before = copy.deepcopy(TAXONOMY)
    def fail(*args):
        raise RuntimeError('Vector write failed')
    monkeypatch.setattr(additions, 'write_vectors', fail)
    submitted = jobs.submit('pdf', b'%PDF-test failing IPTV', new_pdf_labels())
    jobs.close()
    assert jobs.status(submitted['job_id'])['status'] == 'failed'
    assert TAXONOMY == before
    assert not (jobs.root/'config/taxonomy.json').exists()
    with pytest.raises(ValueError):
        Category.validate('IPTV playback issues')


def test_missing_label_waits_for_confirmation(jobs, monkeypatch):
    monkeypatch.setattr(additions, 'extract_pdf', lambda path: ([TextBlock(text='TV playback uses broadband gateway.', pages=[1])], {'pages':1}))
    monkeypatch.setattr(additions, 'suggest_missing', lambda *args: MissingLabelSuggestion(category=None,
        product='Fiber Broadband & Gateway', reason='The excerpt explicitly names the broadband gateway.'))
    monkeypatch.setattr(additions, 'classify_chunk', lambda *args: MetadataResult(title='TV playback checks',
        categories=['IPTV playback issues'], products=['Fiber Broadband & Gateway']))
    before = copy.deepcopy(TAXONOMY)
    submitted = jobs.submit('pdf', b'%PDF-single-label', PDFLabels(category='IPTV playback issues',
        category_description='TV playback failures.'))
    jobs.close()
    assert jobs.status(submitted['job_id'])['status'] == 'awaiting_confirmation'
    assert TAXONOMY == before and not jobs.state.lock.locked()
    jobs.confirm_labels(submitted['job_id'])
    jobs.close()
    assert jobs.status(submitted['job_id'])['status'] == 'completed'
    assert 'IPTV playback issues' in TAXONOMY['categories']
    assert TAXONOMY['products'] == before['products']


def test_registration_rejects_missing_descriptions_and_no_supported_guess(jobs, monkeypatch):
    with pytest.raises(ValueError, match='description'):
        PDFLabels(category='Unknown new label')
    with pytest.raises(ValueError, match='at least one'):
        PDFLabels()
    before = copy.deepcopy(TAXONOMY)
    options = PDFLabels(category='slow SPEED', product='Fiber Broadband & Gateway')
    assert expanded_taxonomy(options, before) == before
    monkeypatch.setattr(additions, 'extract_pdf', lambda path: ([TextBlock(text='Telecom television service.', pages=[1])], {'pages':1}))
    monkeypatch.setattr(additions, 'suggest_missing', lambda *args: MissingLabelSuggestion(category=None, product=None, reason='No match'))
    submitted = jobs.submit('pdf', b'%PDF-no-match', PDFLabels(category='IPTV playback issues', category_description='TV playback failures.'))
    jobs.close()
    assert jobs.status(submitted['job_id'])['status'] == 'failed'
    assert 'provide both labels' in jobs.status(submitted['job_id'])['error']
    assert TAXONOMY == before


def test_product_only_suggests_existing_category(jobs, monkeypatch):
    monkeypatch.setattr(additions, 'extract_pdf', lambda path: ([TextBlock(text='Satellite broadband throughput checks.', pages=[1])], {'pages':1}))
    monkeypatch.setattr(additions, 'suggest_missing', lambda *args: MissingLabelSuggestion(
        category='Slow speed', product=None, reason='The excerpt describes throughput issues.'))
    monkeypatch.setattr(additions, 'classify_chunk', lambda *args: MetadataResult(title='Satellite speed checks',
        categories=['Slow speed'], products=['Satellite Broadband']))
    submitted = jobs.submit('pdf', b'%PDF-product-only', PDFLabels(product='Satellite Broadband',
        product_description='Satellite broadband service.'))
    jobs.close()
    assert jobs.status(submitted['job_id'])['status'] == 'awaiting_confirmation'
    assert jobs.status(submitted['job_id'])['proposed_labels']['category'] == 'Slow speed'
    jobs.confirm_labels(submitted['job_id'])
    jobs.close()
    assert jobs.status(submitted['job_id'])['status'] == 'completed'
    assert 'Satellite Broadband' in TAXONOMY['products']


def test_staged_registry_does_not_leak_and_cache_recovers(jobs, monkeypatch):
    pdf_fixture(monkeypatch)
    proposed = expanded_taxonomy(new_pdf_labels(), copy.deepcopy(TAXONOMY))
    with staged_taxonomy(proposed):
        assert Category.validate('IPTV playback issues')
    with pytest.raises(ValueError):
        Category.validate('IPTV playback issues')
    original_vectors = additions.write_vectors
    def fail(*args):
        raise RuntimeError('Vector write failed')
    monkeypatch.setattr(additions, 'write_vectors', fail)
    failed = jobs.submit('pdf', b'%PDF-cached-new', new_pdf_labels())
    jobs.close()
    assert jobs.status(failed['job_id'])['status'] == 'failed'
    monkeypatch.setattr(additions, 'write_vectors', original_vectors)
    def no_call(*args):
        raise AssertionError('Cached metadata should be reused')
    monkeypatch.setattr(additions, 'classify_chunk', no_call)
    resumed = jobs.submit('pdf', b'%PDF-cached-new', new_pdf_labels())
    jobs.close()
    assert jobs.status(resumed['job_id'])['status'] == 'completed'
