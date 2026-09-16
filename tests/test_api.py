import time
from fastapi.testclient import TestClient
from app.main import app
from app.demo import SAMPLE_RESUME, SAMPLE_JOB

HEADERS = {'X-Resume-Studio': '1'}


def test_missing_key_is_explicit(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'openai')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with TestClient(app) as client:
        assert client.get('/api/config').json()['live_enabled'] is False
        response = client.post('/api/generate', headers=HEADERS, json={
            'resume_text': SAMPLE_RESUME, 'job_text': SAMPLE_JOB, 'confirmed': True})
        assert response.status_code == 503
        assert 'OPENAI_API_KEY' in response.json()['detail']


def test_origin_boundary_and_confirmation():
    with TestClient(app) as client:
        assert client.post('/api/demo').status_code == 403
        assert client.post('/api/demo', headers={**HEADERS, 'Origin': 'https://evil.example'}).status_code == 403
        assert client.post('/api/generate', headers=HEADERS, json={
            'resume_text': SAMPLE_RESUME, 'job_text': SAMPLE_JOB, 'confirmed': False}).status_code == 422


def test_upload_and_demo_download_lifecycle():
    with TestClient(app) as client:
        response = client.post('/api/extract', headers=HEADERS,
                               files={'file': ('resume.txt', SAMPLE_RESUME.encode(), 'text/plain')})
        assert response.status_code == 200
        assert response.json()['text'].startswith('Alex Morgan')
        response = client.post('/api/demo', headers=HEADERS)
        assert response.status_code == 202
        key = response.json()['id']
        assert client.get(f'/api/jobs/{key}/download').status_code == 409
        deadline = time.time() + 10
        while time.time() < deadline:
            state = client.get(f'/api/jobs/{key}').json()
            if state['status'] in ('complete', 'needs_confirmation', 'error'):
                break
            time.sleep(.1)
        assert state['status'] == 'needs_confirmation', state
        assert state['demo'] is True
        assert state['result']['usage']['calls'] == 0
        assert client.get(f'/api/jobs/{key}/download').status_code == 409
        assert '_pending' not in state
        assert state['result']['report']['score'] == 100
        assert client.post(f'/api/jobs/{key}/confirm', headers=HEADERS, json={
            'accepted_terms': ['unknown'], 'reviewed_all': True}).status_code == 422
        assert client.post(f'/api/jobs/{key}/confirm', headers=HEADERS, json={
            'accepted_terms': [], 'reviewed_all': False}).status_code == 422
        assert client.post(f'/api/jobs/{key}/confirm', headers=HEADERS, json={
            'accepted_terms': [], 'accepted_claim_ids': ['nope'], 'reviewed_all': True}).status_code == 422
        final = client.post(f'/api/jobs/{key}/confirm', headers=HEADERS, json={
            'accepted_terms': ['Docker'], 'accepted_claim_ids': ['j1p1'], 'reviewed_all': True})
        assert final.status_code == 200 and final.json()['status'] == 'complete'
        assert 0 < final.json()['result']['report']['score'] < 100
        assert final.json()['result']['unverified_terms'] == []
        assert client.post(f'/api/jobs/{key}/confirm', headers=HEADERS, json={
            'accepted_terms': ['Docker'], 'reviewed_all': True}).status_code == 409
        doc = client.get(f'/api/jobs/{key}/download')
        assert doc.status_code == 200 and doc.content.startswith(b'PK')
        assert 'example-resume.docx' in doc.headers['content-disposition']
        assert client.delete(f'/api/jobs/{key}', headers=HEADERS).status_code == 200
        assert client.get(f'/api/jobs/{key}').status_code == 404


def test_cancel_removes_active_run():
    with TestClient(app) as client:
        key = client.post('/api/demo', headers=HEADERS).json()['id']
        assert client.delete(f'/api/jobs/{key}', headers=HEADERS).status_code == 200
        assert client.get(f'/api/jobs/{key}').status_code == 404


def test_body_limit_before_file_parser():
    with TestClient(app) as client:
        response = client.post('/api/extract', headers=HEADERS, content=b'x'*(6*1024*1024+1))
        assert response.status_code == 413
