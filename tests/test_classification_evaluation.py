"""Metrics, leakage checks, and resumable evaluation without Groq calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from telecom_support.evaluation.classification import (
    build_report, check_examples, fingerprint, open_run, load_attempts)

def labels(category='Slow speed',severity='Medium'):
    return {'categories':[category], 'products':['Fiber Broadband & Gateway'],
            'severity':severity,'sentiment':'Neutral'}

def case(tid,category='Slow speed'):
    return {'ticket_id':tid,'complaint':f'Complaint {tid}','expected':labels(category)}

def success(c,prediction=None):
    return {'ticket_id':c['ticket_id'],'status':'success',
            'prediction':prediction or c['expected'],'latency_seconds':0.5}

def test_perfect_predictions_and_empty_report():
    cases=[case('a'),case('b','Billing dispute')]
    report=build_report(cases,[success(c) for c in cases])
    overall=report['quality_on_valid_predictions']['overall']
    assert overall['all_fields_accuracy']==1
    assert set(overall['field_accuracy'].values())=={1}
    assert overall['category_macro_f1']['value']==1
    assert 'unknown_detection' not in report
    empty=build_report(cases,[])
    assert empty['quality_on_valid_predictions']['overall']['all_fields_accuracy'] is None
    assert empty['successful_and_all_fields_correct_fraction_of_selected']==0

def test_extra_missing_labels_and_field_error():
    c=case('a')
    predicted=labels(severity='High')
    predicted['categories']=['Slow speed','Billing dispute']
    report=build_report([c],[success(c,predicted)])
    overall=report['quality_on_valid_predictions']['overall']
    assert overall['field_accuracy']['categories']==0
    assert overall['field_accuracy']['products']==1
    assert overall['field_accuracy']['severity']==0
    assert overall['category_macro_f1']['value']==0.5
    assert report['incorrect_cases'][0]['incorrect_fields']==['categories','severity']
    predicted['categories']=['Billing dispute']
    assert build_report([c],[success(c,predicted)])['quality_on_valid_predictions']['overall']['category_macro_f1']['value']==0

def test_category_errors_and_failure_retry():
    cases=[case('a'),case('b','Billing dispute'),case('c','Billing dispute')]
    attempts=[{'ticket_id':'a','status':'error','error':'rate limit','latency_seconds':1},
              success(cases[0],labels('Billing dispute')),success(cases[1]),
              success(cases[2],labels('Slow speed'))]
    report=build_report(cases,attempts)
    assert report['quality_on_valid_predictions']['overall']['field_accuracy']['categories']==pytest.approx(1/3)
    assert report['request_health']['failures']==1
    assert report['request_health']['successes']==3
    assert report['completed_tickets']==3
    assert report['quality_on_valid_predictions']['overall']['valid_predictions']==3

def test_incompatible_settings_choose_new_run(tmp_path):
    first=open_run(tmp_path,{'model':'m','prompt':'p','data':'d'})
    assert first==open_run(tmp_path,{'data':'d','prompt':'p','model':'m'})
    assert first!=open_run(tmp_path,{'model':'m','prompt':'changed','data':'d'})
    assert first!=open_run(tmp_path,{'model':'other','prompt':'p','data':'d'})
    (first/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest mismatch'):
        open_run(tmp_path,{'model':'m','prompt':'p','data':'d'})

def test_corrupt_saved_predictions_fail(tmp_path):
    path=tmp_path/'predictions.jsonl'
    path.write_text(json.dumps(success(case('a')))+'\n')
    assert len(load_attempts(path,[case('a')]))==1
    path.write_text('{broken\n')
    with pytest.raises(ValueError,match='line 1'):
        load_attempts(path,[case('a')])

def test_examples_reject_heldout_ticket():
    train=SimpleNamespace(ticket_id='train',split='train',customer_complaint='Training complaint')
    test=SimpleNamespace(ticket_id='test',customer_complaint='Evaluation complaint')
    with pytest.raises(ValueError,match='training tickets'):
        check_examples([{'ticket_id':'test','complaint':'Evaluation complaint','labels':labels()}],[train],[test])
    test.customer_complaint='  training   COMPLAINT '
    with pytest.raises(ValueError,match='overlap'):
        check_examples([], [train],[test])

def test_script_failed_sample_resume_and_full_report(tmp_path,monkeypatch):
    from telecom_support.ingestion.schemas import TicketRecord
    from telecom_support.taxonomy import TAXONOMY
    from telecom_support.classification.schemas import ComplaintLabels
    import sys
    script=Path(__file__).resolve().parents[1]/'scripts/evaluate_classification.py'
    spec=importlib.util.spec_from_file_location('evaluate_script',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder=tmp_path/'data/indexes'
    folder.mkdir(parents=True)
    def record(tid,split,category):
        return TicketRecord(source_id=tid,source_file='tickets.xlsx',source_version='v1',taxonomy_version='2',
            search_text='Complaint '+tid,ticket_id=tid,created_at='2026-01-01',category=category,
            product='Fiber Broadband & Gateway',severity='Medium',sentiment='Neutral',
            customer_complaint='Complaint '+tid,conversation='Conversation',resolution_steps='Fix',
            resolution_summary='Done',split=split)
    training=[record('train','train','Slow speed')]
    evaluation=[record('a','test','Slow speed'),record('b','test','Billing dispute'),record('c','test','Slow speed')]
    for name,rows in [('historical_tickets.jsonl',training),('evaluation_tickets.jsonl',evaluation)]:
        (folder/name).write_text(''.join(r.model_dump_json()+'\n' for r in rows))
    (folder/'classification_examples.json').write_text(json.dumps({'taxonomy_version':TAXONOMY['version'],
        'examples':[{'ticket_id':'train','complaint':'Complaint train','labels':labels()}]}))
    # Identity includes production classifier code without needing it in the fake project.
    destination=tmp_path/'src/telecom_support/classification'
    destination.mkdir(parents=True)
    (destination/'service.py').write_text('fixed classifier')
    (destination.parent/'llm.py').write_text('fixed helper')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    calls=[]
    class FakeLLM:
        def __init__(self,root):pass
        def close(self):pass
    monkeypatch.setattr(module,'StructuredLLM',FakeLLM)
    def fake_classify(llm,complaint,examples):
        assert all(e['ticket_id']=='train' for e in examples)
        calls.append(complaint)
        if complaint.endswith('b') and calls.count(complaint)==1:
            raise module.ServiceError('Simulated rate limit')
        return ComplaintLabels.model_validate(labels('Billing dispute' if complaint.endswith('b') else 'Slow speed'))
    monkeypatch.setattr(module,'classify',fake_classify)
    monkeypatch.setattr(module,'load_dotenv',lambda *args,**kwargs:None)
    monkeypatch.setenv('GROQ_MODEL','fake-model')
    monkeypatch.setattr(sys,'argv',['evaluate','--limit','2','--delay','0'])
    assert module.main()==1
    assert len(calls)==2
    monkeypatch.setattr(sys,'argv',['evaluate','--delay','0'])
    assert module.main()==0
    assert len(calls)==4  # Saved success is skipped; failed ticket is retried.
    runs=list((folder/'evaluations/classification').iterdir())
    assert len(runs)==1
    report=json.loads((runs[0]/'report.json').read_text())
    assert report['completed_tickets']==3
    assert report['request_health']['failures']==1
    assert report['request_health']['successes']==3
    assert report['quality_on_valid_predictions']['overall']['all_fields_accuracy']==1
    monkeypatch.setattr(sys,'argv',['evaluate','--report-only'])
    assert module.main()==0
    assert len(calls)==4
