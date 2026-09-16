import asyncio
import json
from types import SimpleNamespace
import pytest
from app.agents import OpenRouterProvider, provider_settings
from app.demo import DemoProvider, SAMPLE_JOB, sample_draft
from app.models import Resume


def test_provider_selection(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'openrouter')
    assert provider_settings() == ('openrouter', 'OPENROUTER_API_KEY')
    monkeypatch.setenv('AI_PROVIDER', 'invalid')
    with pytest.raises(ValueError):
        provider_settings()


def test_openrouter_schema_usage_and_model_isolation(monkeypatch):
    captured = {}
    plan = asyncio.run(DemoProvider(delay=0).profile(SAMPLE_JOB))
    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=55), choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=plan.model_dump_json(), refusal=None))])
    monkeypatch.setattr('app.agents.AsyncOpenAI', lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setenv('OPENROUTER_WRITER_MODEL', '')
    monkeypatch.setenv('OPENROUTER_MODEL', 'example/model')
    monkeypatch.setenv('WRITER_MODEL', 'openai-only')
    monkeypatch.setenv('MAX_RUN_TOKENS', '60000')
    provider = OpenRouterProvider()
    assert provider.writer_model == 'example/model'
    assert asyncio.run(provider.profile(SAMPLE_JOB)) == plan
    assert captured['model'] == 'example/model'
    assert captured['extra_body']['provider']['require_parameters']
    assert captured['extra_body']['plugins'] == [{'id': 'response-healing'}]
    assert captured['response_format']['json_schema']['strict']
    assert captured['max_tokens'] == 5000
    assert provider.tokens == 55


def test_openrouter_invalid_response_fails(monkeypatch):
    async def create(**kwargs):
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content='{}', refusal=None))])
    monkeypatch.setattr('app.agents.AsyncOpenAI', lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setenv('MAX_RUN_TOKENS', '60000')
    with pytest.raises(ValueError, match='required schema'):
        asyncio.run(OpenRouterProvider().profile(SAMPLE_JOB))


def test_openrouter_allows_large_structured_resume_output(monkeypatch):
    captured = {}
    resume = sample_draft()
    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(
                content=resume.model_dump_json(), refusal=None))])
    monkeypatch.setattr('app.agents.AsyncOpenAI', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setenv('MAX_RUN_TOKENS', '60000')
    provider = OpenRouterProvider()
    result = asyncio.run(provider._call('prompt', {}, Resume, 'openai/gpt-4.1-mini'))
    assert result == resume
    assert captured['max_tokens'] == 16000


@pytest.mark.parametrize('model,effort', [
    ('openai/gpt-5-nano', 'minimal'),
    ('xiaomi/mimo-v2.5', 'none'),
])
def test_openrouter_controls_reasoning_for_structured_models(monkeypatch, model, effort):
    captured = {}
    plan = asyncio.run(DemoProvider(delay=0).profile(SAMPLE_JOB))
    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(
                content=plan.model_dump_json(), refusal=None))])
    monkeypatch.setattr('app.agents.AsyncOpenAI', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setenv('MAX_RUN_TOKENS', '60000')
    provider = OpenRouterProvider()
    asyncio.run(provider._call('prompt', {'input': 'test'}, type(plan), model))
    assert captured['extra_body']['reasoning'] == {'effort': effort, 'exclude': True}


@pytest.mark.parametrize('finish_reason,message', [
    ('length', 'output limit'), ('content_filter', 'content_filter')])
def test_openrouter_reports_finish_reason(monkeypatch, finish_reason, message):
    async def create(**kwargs):
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason=finish_reason, message=SimpleNamespace(
                content='partial', refusal=None))])
    monkeypatch.setattr('app.agents.AsyncOpenAI', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setenv('MAX_RUN_TOKENS', '60000')
    with pytest.raises(ValueError, match=message):
        asyncio.run(OpenRouterProvider().profile(SAMPLE_JOB))
