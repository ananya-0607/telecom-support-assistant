import json
from telecom_support.monitoring import Monitoring


def test_metrics_denominators_reset_and_safe_logs(tmp_path):
    path = tmp_path/'logs/requests.jsonl'
    monitor = Monitoring(path)
    assert monitor.snapshot()['average_response_seconds'] is None
    monitor.record('one', 200, 2, retrieval=0.1, no_evidence=True)
    monitor.record('two', 400, 4, error_type='ServiceError')
    values = monitor.snapshot()
    assert values['resolution_requests'] == 2
    assert values['successful_requests'] == values['failed_requests'] == 1
    assert values['requests_without_evidence'] == 1
    assert values['average_response_seconds'] == 3
    assert values['average_retrieval_seconds'] == 0.1
    assert values['retrieval_timing_samples'] == 1
    monitor.close()
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert entries[1]['error_type'] == 'ServiceError'
    assert all('complaint' not in item and 'exception' not in item for item in entries)
    restarted = Monitoring(path)
    assert restarted.snapshot()['resolution_requests'] == 0
    restarted.close()


def test_endpoint_tracks_success_validation_busy_and_errors(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import telecom_support.api as api
    from telecom_support.llm import ServiceError
    class Resource:
        def __init__(self, *args): pass
        def close(self): pass
    class Pipeline:
        def __init__(self, *args): pass
        def resolve(self, complaint):
            if complaint == 'service': raise ServiceError('PRIVATE KEY AND COMPLAINT')
            if complaint == 'internal': raise RuntimeError('PRIVATE KEY AND COMPLAINT')
            return {'resolution': {'status': 'insufficient_evidence'},
                    'timings_seconds': {'retrieval': 0.2}}
    monkeypatch.setattr(api, 'ROOT', tmp_path)
    monkeypatch.setattr(api, 'SearchIndex', Resource)
    monkeypatch.setattr(api, 'StructuredLLM', Resource)
    monkeypatch.setattr(api, 'ResolutionPipeline', Pipeline)
    with TestClient(api.app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/metrics').json()['resolution_requests'] == 0
        response = client.post('/resolve', json={'complaint': 'PRIVATE CUSTOMER TEXT'})
        assert response.status_code == 200
        assert response.headers['X-Request-ID']
        assert client.post('/resolve', json={}).status_code == 422
        assert client.post('/resolve', json={'complaint': 'service'}).status_code == 400
        assert client.post('/resolve', json={'complaint': 'internal'}).status_code == 500
        api.app.state.lock.acquire()
        try:
            assert client.post('/resolve', json={'complaint': 'busy'}).status_code == 503
        finally:
            api.app.state.lock.release()
        metrics = client.get('/metrics').json()
        assert metrics['resolution_requests'] == 5
        assert metrics['successful_requests'] == 1
        assert metrics['failed_requests'] == 4
        assert metrics['requests_without_evidence'] == 1
        assert metrics['retrieval_timing_samples'] == 1
        assert metrics['status_codes'] == {'200':1, '422':1, '400':1, '500':1, '503':1}
    text = (tmp_path/'logs/resolution_requests.jsonl').read_text()
    assert 'PRIVATE' not in text
