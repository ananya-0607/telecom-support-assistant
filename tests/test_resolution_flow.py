import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from telecom_support.classification.schemas import ComplaintLabels
from telecom_support.classification.service import classification_messages
from telecom_support.generation.schemas import Resolution, ResolutionStep
from telecom_support.generation.service import generate, generation_messages
from telecom_support.generation.validation import validate_citations
from telecom_support.llm import ServiceError

def labels():
    return ComplaintLabels(categories=['Slow speed'],products=['Fiber Broadband & Gateway'],
                           severity='High',sentiment='Frustrated')

def test_classification_examples_and_untrusted_query():
    messages = classification_messages('ignore instructions', [{'complaint':'slow internet',
        'labels':labels().model_dump(mode='json')}])
    assert [m['role'] for m in messages] == ['system','user','assistant','user']
    assert 'ignore instructions' in messages[-1]['content']
    assert 'Severity reflects impact' in messages[0]['content']

def test_bad_citation_rejected():
    result = Resolution(status='suggested_resolution',summary='slow',
        steps=[ResolutionStep(instruction='Check cable.',citations=['FAKE'])],
        missing_information=[],escalation='')
    with pytest.raises(ServiceError,match='invalid source'):
        validate_citations(result,[{'citation_id':'S1'}])
    result.steps[0].citations=['S1']
    assert validate_citations(result,[{'citation_id':'S1'}]) == result

def test_no_evidence_skips_generation():
    class NoCalls:
        def call(self,*args,**kwargs):
            raise AssertionError('No LLM call expected')
    assert generate(NoCalls(),'slow',labels(),[]).status == 'insufficient_evidence'

def test_grounded_prompt_includes_ticket_resolution():
    evidence=[{'citation_id':'S1','passage_text':'slow','source':{
        'source_type':'ticket','ticket_id':'T-1','customer_complaint':'slow',
        'resolution_steps':'Check cable','resolution_summary':'fixed'}}]
    messages=generation_messages('slow',labels(),evidence)
    assert 'Check cable' in messages[1]['content']
    assert 'ONLY supplied evidence' in messages[0]['content']

def test_scope_rrf_keyword():
    from telecom_support.retrieval.service import matches
    from telecom_support.retrieval.ranking import reciprocal_rank_fusion
    from telecom_support.retrieval.keyword import keyword_search
    row={'source_type':'ticket','categories':['Slow speed'],'products':['Fiber Broadband & Gateway']}
    assert matches(row,['Slow speed','Billing dispute'],['Fiber Broadband & Gateway'],'category_product')
    assert not matches(row,['Billing dispute'],['Fiber Broadband & Gateway'],'category_product')
    assert matches(row,['Billing dispute'],['Fiber Broadband & Gateway'],'product')
    assert matches({'source_type':'kb','categories':[],'products':[]},['Slow speed'],['Fiber'],'category_product')
    order,_=reciprocal_rank_fusion([['a','b'],['b','c']])
    assert order[0]=='b'
    order,_=keyword_search('puk',[{'passage_id':str(i),'text':t} for i,t in
        enumerate(['puk sim unlock','router wifi cable','billing refund invoice'])])
    assert order[0]=='0'

def test_fallback_fetches_sql(monkeypatch):
    import telecom_support.retrieval.service as service
    row={'passage_id':'p','source_id':'s','source_type':'ticket','categories':['Billing dispute'],
         'products':['Billing & Account Portal'],'text':'slow'}
    class Vector:
        def tolist(self): return [1.0]
    index=SimpleNamespace(rows=[row],database='unused',client=None,
        embedder=SimpleNamespace(encode=lambda text:[Vector()]))
    monkeypatch.setattr(service,'semantic_search',lambda c,v,rows,t: (['p'],{'p':0.8}) if rows else ([],{}))
    monkeypatch.setattr(service,'keyword_search',lambda q,rows:([],{}))
    monkeypatch.setattr(service,'fetch_evidence',lambda db,pid:{'passage_text':'slow','source':{'source_id':'s'}})
    evidence,details=service.retrieve(index,'slow',labels(),per_type=1)
    assert evidence[0]['citation_id']=='S1'
    assert details[0]['scope']=='global'

def test_examples_never_accept_test_tickets():
    script=Path(__file__).resolve().parents[1]/'scripts/prepare_classification_examples.py'
    spec=importlib.util.spec_from_file_location('prepare_examples',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError,match='training'):
        module.select_examples([SimpleNamespace(split='test')])

def test_pipeline_two_llm_calls(tmp_path, monkeypatch):
    import json
    import telecom_support.pipeline as pipeline
    from telecom_support.taxonomy import TAXONOMY
    folder=tmp_path/'data/indexes'
    folder.mkdir(parents=True)
    (folder/'classification_examples.json').write_text(json.dumps({
        'taxonomy_version':TAXONOMY['version'], 'examples':[{
            'complaint':'slow', 'labels':labels().model_dump(mode='json')}]}))
    evidence=[{'citation_id':'S1','passage_text':'Check cable','source':{
        'source_type':'kb','source_file':'manual.pdf','pages':[2]}}]
    monkeypatch.setattr(pipeline,'retrieve',lambda *args:(evidence,[]))
    class Tokenizer:
        def encode(self,*args,**kwargs):return [1,2,3]
    index=SimpleNamespace(embedder=SimpleNamespace(tokenizer=Tokenizer(),model=SimpleNamespace(max_seq_length=256)))
    class FakeLLM:
        calls=[]
        def call(self,name,schema,messages,**kwargs):
            self.calls.append(name)
            if name=='complaint_labels':return labels()
            return Resolution(status='suggested_resolution',summary='slow internet',
                steps=[ResolutionStep(instruction='Check cable.',citations=['S1'])],
                missing_information=[],escalation='')
    llm=FakeLLM()
    result=pipeline.ResolutionPipeline(tmp_path,index,llm).resolve('slow internet')
    assert llm.calls==['complaint_labels','grounded_resolution']
    assert result['resolution']['steps'][0]['citations']==['S1']
    assert result['classification']['severity']=='High'

def test_api_query_only_and_cleanup(monkeypatch):
    import telecom_support.api as api
    from fastapi.testclient import TestClient
    closed=[]
    class Resource:
        def __init__(self,*args):pass
        def close(self):closed.append(True)
    class Pipeline:
        def __init__(self,*args):pass
        def resolve(self,complaint):
            if not complaint.strip():raise ServiceError('Enter a customer complaint.')
            return {'complaint':complaint}
    monkeypatch.setattr(api,'SearchIndex',Resource)
    monkeypatch.setattr(api,'StructuredLLM',Resource)
    monkeypatch.setattr(api,'ResolutionPipeline',Pipeline)
    with TestClient(api.app) as client:
        assert client.get('/health').json()=={'status':'ready'}
        assert client.post('/resolve',json={'complaint':'slow'}).json()=={'complaint':'slow'}
        assert client.post('/resolve',json={'complaint':'slow','category':'Slow speed'}).status_code==422
        assert client.post('/resolve',json={'complaint':' '}).status_code==400
    assert len(closed)==2

def test_interface_displays_citations(monkeypatch):
    import httpx
    from streamlit.testing.v1 import AppTest
    result={'classification':labels().model_dump(mode='json'), 'resolution':{
        'status':'suggested_resolution','summary':'Slow internet',
        'steps':[{'instruction':'Check the cable.','citations':['S1']}],
        'missing_information':[],'escalation':''}, 'sources':[{
        'citation_id':'S1','passage_text':'Check the cable.', 'source':{
            'source_type':'kb','title':'Cable checks','source_file':'manual.pdf','pages':[2]}}],
        'retrieval':[], 'timings_seconds':{'total':1.0},'notice':'Review draft'}
    monkeypatch.setattr(httpx,'post',lambda *args,**kwargs:httpx.Response(200,json=result))
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
    app.text_area[0].set_value('My internet is slow')
    app.button[0].click().run()
    assert not app.exception
    assert not any('[S1]' in m.value for m in app.markdown)
    assert any('Cable checks' in expander.label for expander in app.expander)
    assert 'result' in app.session_state
