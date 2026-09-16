import asyncio
from types import SimpleNamespace
import pytest
from app.agents import ClaudeProvider, BudgetExceeded, provider_settings
from app.demo import DemoProvider, SAMPLE_JOB


def test_claude_selection(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'claude')
    assert provider_settings() == ('claude', 'CLAUDE_API_KEY')


@pytest.mark.parametrize('stop_reason', ['end_turn', 'max_tokens', 'refusal'])
def test_claude_structured_request_and_usage(monkeypatch, stop_reason):
    captured = {}; plan = asyncio.run(DemoProvider(delay=0).profile(SAMPLE_JOB))
    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stop_reason=stop_reason,
            content=[SimpleNamespace(type='text', text=plan.model_dump_json())],
            usage=SimpleNamespace(input_tokens=40, output_tokens=15))
    monkeypatch.setattr('app.agents.AsyncAnthropic', lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setenv('CLAUDE_MODEL', 'claude-example'); monkeypatch.setenv('CLAUDE_WRITER_MODEL', '')
    monkeypatch.setenv('WRITER_MODEL', 'openai-only'); monkeypatch.setenv('MAX_RUN_TOKENS', '60000')
    provider = ClaudeProvider(); assert provider.writer_model == 'claude-example'
    if stop_reason == 'end_turn': assert asyncio.run(provider.profile(SAMPLE_JOB)) == plan
    else:
        with pytest.raises(ValueError, match='stopped before completing'): asyncio.run(provider.profile(SAMPLE_JOB))
    assert captured['model'] == 'claude-example'
    assert captured['output_config']['format']['type'] == 'json_schema'
    assert captured['system'] and captured['messages'][0]['role'] == 'user'
    assert provider.tokens == 55


def test_claude_budget_prevents_dispatch(monkeypatch):
    monkeypatch.setattr('app.agents.AsyncAnthropic', lambda **kwargs: SimpleNamespace())
    monkeypatch.setenv('MAX_RUN_TOKENS', '1'); provider = ClaudeProvider()
    with pytest.raises(BudgetExceeded): asyncio.run(provider.profile(SAMPLE_JOB))
    assert provider.calls == 0


def test_claude_caches_shared_evidence_block(monkeypatch):
    captured = {}
    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stop_reason='end_turn', usage=None,
            content=[SimpleNamespace(type='text', text='{"claim_checks":[],"proposal_checks":[],"eligibility":[],"questions":[]}')])
    monkeypatch.setattr('app.agents.AsyncAnthropic', lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=create)))
    provider = ClaudeProvider()
    asyncio.run(provider.audit({'E1': 'line'}, [], []))
    assert captured['system'][1]['cache_control'] == {'type': 'ephemeral'}
    assert 'candidate_evidence' in captured['system'][1]['text']
