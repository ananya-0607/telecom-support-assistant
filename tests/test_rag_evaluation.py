"""RAG scoring bookkeeping and evidence tests, without external API requests."""
import importlib.util
import json
from pathlib import Path
import pytest
from telecom_support.evaluation.rag import (
    ReferenceCase, answer_text, evidence_contexts, citation_validity, report, percentile)


class WordTokenizer:
    def encode(self, text, add_special_tokens=True, **kwargs):
        return list(range(len(text.split()) + (2 if add_special_tokens else 0)))

    def __call__(self, text, **kwargs):
        import re
        return {'offset_mapping': [m.span() for m in re.finditer(r'\S+', text)]}

    def num_special_tokens_to_add(self, pair=False):
        return 2


def output():
    return {'classification': {'categories': ['Slow speed'], 'products': ['Fiber Broadband & Gateway'],
            'severity': 'Medium', 'sentiment': 'Neutral'},
        'resolution': {'status': 'suggested_resolution', 'summary': 'Slow connection',
            'steps': [{'instruction': 'Measure wired throughput.', 'citations': ['S1']}],
            'missing_information': [], 'escalation': ''},
        'sources': [{'citation_id': 'S1', 'passage_text': 'Customer reports slow throughput.',
            'source': {'source_type': 'ticket', 'ticket_id': 'train-1',
                'customer_complaint': 'Slow internet', 'resolution_steps': 'Measure wired throughput.',
                'resolution_summary': 'Traffic contention was investigated.'}}]}


def test_judge_sees_resolution_evidence_not_just_ticket_conversation():
    result = output()
    context = json.loads(evidence_contexts(result)[0])
    assert context['resolution_steps'] == 'Measure wired throughput.'
    assert context['resolution_summary'] == 'Traffic contention was investigated.'
    assert answer_text(result) == 'Slow connection\nMeasure wired throughput.'
    assert citation_validity(result) == 1
    result['resolution']['steps'][0]['citations'] = ['S1', 'not-retrieved']
    assert citation_validity(result) == .5


def test_no_steps_has_no_citation_score_not_perfect_score():
    result = output()
    result['resolution']['steps'] = []
    assert citation_validity(result) is None


def test_report_exposes_missing_scores_failures_and_draft_references():
    cases = [ReferenceCase(ticket_id='test-1', complaint='Slow', reference_resolution='Measure', ticket_version='v')]
    attempts = [{'ticket_id': 'test-1', 'status': 'error', 'error': '429', 'latency_seconds': 1},
        {'ticket_id': 'test-1', 'status': 'success', 'result': output(), 'latency_seconds': 2}]
    metrics = [{'ticket_id': 'test-1', 'metric': 'Faithfulness', 'score': .8, 'status': 'success'}]
    summary = report(cases, attempts, metrics)
    assert summary['Request success rate'] == .5
    assert summary['RAG quality']['Faithfulness'] == {'average_score': .8, 'scored_cases': 1}
    assert summary['RAG quality']['Context recall']['average_score'] is None
    assert 'draft' in summary['Reference status']
    assert percentile([1, 2, 10], .95) == 10


def load_script():
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_rag.py'
    spec = importlib.util.spec_from_file_location('rag_evaluation_script', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_rejects_changed_or_nonheldout_complaint():
    from types import SimpleNamespace
    module = load_script()
    reference = ReferenceCase(ticket_id='test-1', complaint='Slow', reference_resolution='Measure', ticket_version='v')
    with pytest.raises(ValueError, match='held-out'):
        module.validate_references([reference], [])
    ticket = SimpleNamespace(ticket_id='test-1', split='train', customer_complaint='Slow')
    with pytest.raises(ValueError, match='held-out'):
        module.validate_references([reference], [ticket])


def test_citation_judge_receives_only_cited_sources():
    import asyncio
    module = load_script()
    class Judge:
        def call(self, name, schema, messages):
            payload = json.loads(messages[-1]['content'])
            assert len(payload['cited_evidence']) == 1
            assert 'uncited policy' not in str(payload)
            return schema(verdict='supported', reason='Equivalent instruction.')
    result = output()
    result['sources'].append({'citation_id': 'S2', 'passage_text': 'uncited policy',
        'source': {'source_type': 'kb', 'source_file': 'manual.pdf', 'pages': [1]}})
    score, details = asyncio.run(module.score_citations(Judge(), result))
    assert score == 1 and len(details) == 1


def test_legacy_folder_gets_readable_name_without_losing_predictions(tmp_path):
    from telecom_support.evaluation.classification import open_run, fingerprint
    manifest = {'model': 'openai/gpt-oss-20b'}
    legacy = tmp_path / fingerprint(manifest)[:16]
    legacy.mkdir()
    (legacy / 'manifest.json').write_text(json.dumps(manifest))
    (legacy / 'predictions.jsonl').write_text('saved')
    folder = open_run(tmp_path, manifest)
    assert folder.name.startswith('rag-evaluation-openai-gpt-oss-20b-')
    assert (folder / 'predictions.jsonl').read_text() == 'saved'


def test_all_standard_metrics_execute_with_simulated_groq(monkeypatch):
    import asyncio
    import numpy as np
    from types import SimpleNamespace
    monkeypatch.setenv('RAGAS_DO_NOT_TRACK', 'true')
    pytest.importorskip('ragas')
    from ragas.dataset_schema import SingleTurnSample
    from telecom_support.evaluation.ragas_adapter import GroqRagasJudge, RequestPacer, create_metrics
    class Completions:
        replies = []
        def create(self, **kwargs):
            assert kwargs['response_format'] == {'type': 'json_object'}
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content=json.dumps(self.replies.pop(0))))])
    completions = Completions()
    llm = SimpleNamespace(model='fake', client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    judge = GroqRagasJudge(llm, RequestPacer(0))
    embedder = SimpleNamespace(encode=lambda texts: np.array([[1., 0.] for _ in texts]),
        tokenizer=WordTokenizer(), model=SimpleNamespace(max_seq_length=256))
    metrics = create_metrics(judge, embedder)
    sample = SingleTurnSample(user_input='How to diagnose slow speed?', response='Measure wired throughput.',
        reference='Measure wired throughput.', retrieved_contexts=['Measure wired throughput.'])
    responses = {
        'Context precision': [{'reason': 'Relevant', 'verdict': 1}],
        'Context recall': [{'classifications': [{'statement': 'Measure wired throughput.', 'reason': 'Present', 'attributed': 1}]}],
        'Faithfulness': [{'statements': ['Measure wired throughput.']},
            {'statements': [{'statement': 'Measure wired throughput.', 'reason': 'Supported', 'verdict': 1}]}],
        'Answer relevance': [{'question': 'How to diagnose slow speed?', 'noncommittal': 0}] * 3,
        'Answer correctness': [{'statements': ['Measure wired throughput.']}] * 2 +
            [{'TP': [{'statement': 'Measure wired throughput.', 'reason': 'Same meaning'}], 'FP': [], 'FN': []}]
    }
    for name, metric in metrics.items():
        completions.replies = responses[name].copy()
        assert asyncio.run(metric.single_turn_ascore(sample, timeout=30)) == pytest.approx(1)
        assert not completions.replies


def test_long_evaluation_text_is_fully_covered_weighted_and_normalized(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    monkeypatch.setenv('RAGAS_DO_NOT_TRACK', 'true')
    pytest.importorskip('ragas')
    from telecom_support.evaluation.ragas_adapter import LocalEmbeddings
    calls = []
    def encode(texts):
        assert all(len(WordTokenizer().encode(t)) <= 5 for t in texts)
        calls.append(texts)
        return np.array([[1., 0.] if t.startswith('first') else [0., 1.] for t in texts])
    embedder = SimpleNamespace(encode=encode, tokenizer=WordTokenizer(), model=SimpleNamespace(max_seq_length=5))
    adapter = LocalEmbeddings(embedder)
    result = adapter.embed_query('first second third fourth')
    assert calls == [['first second third', 'fourth']]
    assert result == pytest.approx(np.array([3., 1.]) / np.sqrt(10))
    assert adapter.embed_query('first short') == [1., 0.]
    with pytest.raises(ValueError, match='contain text'):
        adapter.embed_query(' ')


def test_answers_resume_and_report_do_not_repeat_successful_calls(tmp_path, monkeypatch):
    import shutil
    import sqlite3
    import sys
    monkeypatch.setenv('RAGAS_DO_NOT_TRACK', 'true')
    pytest.importorskip('ragas')
    module = load_script()
    project = module.ROOT
    shutil.copytree(project / 'src', tmp_path / 'src')
    data = tmp_path / 'data/indexes'
    data.mkdir(parents=True)
    reference_dir = tmp_path / 'data/evaluation'
    reference_dir.mkdir()
    from telecom_support.evaluation.rag import read_rows, save_rows
    from telecom_support.evaluation.classification import fingerprint
    from telecom_support.ingestion.schemas import TicketRecord
    from telecom_support.taxonomy import TAXONOMY
    def make_ticket(tid, split, complaint):
        return TicketRecord(source_id=tid, source_file='tickets.xlsx', source_version='v',
            taxonomy_version=TAXONOMY['version'], search_text=complaint, ticket_id=tid,
            created_at='2026-10-01', category='Slow speed', product='Fiber Broadband & Gateway',
            severity='Medium', sentiment='Neutral', customer_complaint=complaint,
            conversation=complaint, resolution_steps='Measure wired throughput.',
            resolution_summary='Check local traffic.', split=split)
    ticket = make_ticket('test-1', 'test', 'My internet is slow while uploading.')
    train = make_ticket('train-1', 'train', 'Downloads slow down during a backup.')
    save_rows(data / 'evaluation_tickets.jsonl', [ticket.model_dump(mode='json')])
    save_rows(data / 'historical_tickets.jsonl', [train.model_dump(mode='json')])
    (data / 'classification_examples.json').write_text(json.dumps({'taxonomy_version': TAXONOMY['version'],
        'examples': [{'ticket_id': train.ticket_id, 'complaint': train.customer_complaint,
            'labels': output()['classification']}]}))
    reference = ReferenceCase(ticket_id=ticket.ticket_id, complaint=ticket.customer_complaint,
        reference_resolution=ticket.resolution_steps, ticket_version=fingerprint(ticket.model_dump(mode='json')))
    save_rows(reference_dir / 'rag_reference_cases.jsonl', [reference.model_dump()])
    (data / 'search').mkdir()
    (data / 'search/active.json').write_text(json.dumps({'build_id': 'a' * 32}))
    database = tmp_path / 'evidence.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE sources (record_json TEXT, source_type TEXT)')
        for row in read_rows(data / 'historical_tickets.jsonl'):
            connection.execute('INSERT INTO sources VALUES (?, ?)', (json.dumps(row), 'ticket'))
    class Index:
        def __init__(self, root):
            self.database = database
        def close(self):
            pass
    calls = []
    class Pipeline:
        def __init__(self, *args):
            pass
        def resolve(self, complaint):
            calls.append(complaint)
            return output()
    class LLM:
        model = 'simulated'
        def __init__(self, *args):
            pass
        def close(self):
            pass
    import telecom_support.retrieval.index
    import telecom_support.pipeline
    monkeypatch.setattr(telecom_support.retrieval.index, 'SearchIndex', Index)
    monkeypatch.setattr(telecom_support.pipeline, 'ResolutionPipeline', Pipeline)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'StructuredLLM', LLM)
    monkeypatch.setenv('GROQ_MODEL', 'simulated')
    monkeypatch.setattr(sys, 'argv', ['evaluate_rag.py', '--stage', 'answers', '--allow-draft', '--delay', '0'])
    assert module.main() == 0
    assert module.main() == 0
    assert len(calls) == 1
    monkeypatch.setattr(sys, 'argv', ['evaluate_rag.py', '--stage', 'report'])
    assert module.main() == 0
    assert len(calls) == 1
    # Index exclusion is enforced independently of prompt-example exclusion.
    with sqlite3.connect(database) as connection:
        connection.execute('INSERT INTO sources VALUES (?, ?)', (ticket.model_dump_json(), 'ticket'))
    with pytest.raises(ValueError, match='leakage'):
        module.verify_index_separation(database, [ticket])
