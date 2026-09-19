from fastapi.testclient import TestClient
from app.main import app, JOBS
from app.agents import PROVIDERS
from app.demo import DemoProvider, SAMPLE_RESUME, SAMPLE_JOB
from app.models import GenerateRequest
from pathlib import Path

HEADERS = {'X-Resume-Studio': '1'}
PAYLOAD = {'resume_text': SAMPLE_RESUME, 'job_text': SAMPLE_JOB, 'confirmed': True}


def test_config_does_not_expose_keys(monkeypatch):
    monkeypatch.setenv('CLAUDE_API_KEY', 'secret-test-key')
    with TestClient(app) as c:
        response = c.get('/api/config')
        assert 'secret-test-key' not in response.text
        providers = response.json()['providers']
        assert {p['id'] for p in providers} == {'openai', 'claude', 'openrouter'}
        assert next(p for p in providers if p['id'] == 'claude')['configured']


def test_request_model_selection_is_isolated(monkeypatch):
    instances = []
    class FakeClaude(DemoProvider):
        def __init__(self):
            super().__init__()
            self.model = self.writer_model = self.reviewer_model = 'default'
            instances.append(self)
    monkeypatch.setitem(PROVIDERS, 'claude', FakeClaude)
    monkeypatch.setenv('CLAUDE_API_KEY', 'test')
    monkeypatch.setenv('AI_PROVIDER', 'openai')
    with TestClient(app) as c:
        a = c.post('/api/generate', headers=HEADERS, json={**PAYLOAD, 'provider': 'claude', 'model': 'custom/model:version'})
        b = c.post('/api/generate', headers=HEADERS, json={**PAYLOAD, 'provider': 'claude', 'model': 'second-model'})
        assert a.status_code == b.status_code == 202
        assert instances[0].model == instances[0].writer_model == instances[0].reviewer_model == 'custom/model:version'
        assert instances[1].model == 'second-model'
        assert JOBS[a.json()['id']]['provider'] == 'claude'
        assert JOBS[a.json()['id']]['model'] == 'custom/model:version'
        assert c.get('/api/config').json()['provider'] == 'openai'


def test_selected_key_required_and_invalid_model_rejected(monkeypatch):
    monkeypatch.delenv('CLAUDE_API_KEY', raising=False)
    with TestClient(app) as c:
        assert c.post('/api/generate', headers=HEADERS, json={**PAYLOAD, 'provider': 'claude'}).status_code == 503
        for fields in [{'provider': 'unknown'}, {'model': 'https://evil.example?q=x'}, {'model': ''}]:
            assert c.post('/api/generate', headers=HEADERS, json={**PAYLOAD, **fields}).status_code == 422


def test_separate_reviewer_model(monkeypatch):
    instances = []
    class FakeRouter(DemoProvider):
        def __init__(self):
            super().__init__(); self.model = self.writer_model = self.reviewer_model = 'default'; instances.append(self)
    monkeypatch.setitem(PROVIDERS, 'openrouter', FakeRouter)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test')
    with TestClient(app) as c:
        response = c.post('/api/generate', headers=HEADERS, json={**PAYLOAD,
            'provider': 'openrouter', 'model': 'openai/gpt-5-nano',
            'writer_model': 'openai/gpt-4.1-mini',
            'reviewer_model': 'openai/gpt-4.1-mini'})
        assert response.status_code == 202
        assert instances[0].model == 'openai/gpt-5-nano'
        assert instances[0].writer_model == 'openai/gpt-4.1-mini'
        assert instances[0].reviewer_model == 'openai/gpt-4.1-mini'
        assert JOBS[response.json()['id']]['writer_model'] == 'openai/gpt-4.1-mini'
        assert JOBS[response.json()['id']]['reviewer_model'] == 'openai/gpt-4.1-mini'


def test_user_demo_uploads():
    root = Path(__file__).resolve().parents[1] / 'demo_files'
    with TestClient(app) as c:
        text = {}
        resume_file = next(p.name for p in root.iterdir() if p.suffix == '.docx')
        for key, filename in [('resume_text', resume_file), ('job_text', 'job description.txt')]:
            response = c.post('/api/extract', headers=HEADERS, files={'file': (filename, (root / filename).read_bytes())})
            assert response.status_code == 200, response.text
            text[key] = response.json()['text']
        request = GenerateRequest(**text, confirmed=True)
        assert 'Walmart' in request.job_text
        assert len(request.resume_text) > 1000


def test_luna_selection_updates_unpinned_repair_model(monkeypatch):
    instances = []
    class FakeOpenAI(DemoProvider):
        def __init__(self):
            super().__init__()
            self.model = self.writer_model = self.reviewer_model = self.cheap_model = 'old-model'
            instances.append(self)
    monkeypatch.setitem(PROVIDERS, 'openai', FakeOpenAI)
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    monkeypatch.delenv('CHEAP_MODEL', raising=False)
    with TestClient(app) as c:
        response = c.post('/api/generate', headers=HEADERS, json={**PAYLOAD,
            'provider': 'openai', 'model': 'gpt-5.6-luna'})
        assert response.status_code == 202
        assert instances[0].cheap_model == 'gpt-5.6-luna'
        monkeypatch.setenv('CHEAP_MODEL', 'pinned-model')
        response = c.post('/api/generate', headers=HEADERS, json={**PAYLOAD,
            'provider': 'openai', 'reviewer_model': 'custom-reviewer'})
        assert response.status_code == 202
        assert instances[1].reviewer_model == 'custom-reviewer'
        assert instances[1].cheap_model == 'old-model'
