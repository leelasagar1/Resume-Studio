"""Offline request-contract checks; no API credentials or network required."""
import asyncio
from types import SimpleNamespace as NS

import pytest

from app.agents import OpenAIProvider, BudgetExceeded, price_for
from app.models import Audit, JobProfile
from app.demo import DemoProvider, SAMPLE_JOB


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr('app.agents.AsyncOpenAI', lambda **kw: NS(responses=NS()))
    for key in ('OPENAI_MODEL', 'WRITER_MODEL', 'REVIEWER_MODEL', 'CHEAP_MODEL'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('MAX_RUN_TOKENS', '250000')
    return OpenAIProvider()


def response(parsed=None, reason=None):
    return NS(output_text=(parsed.model_dump_json() if hasattr(parsed, 'model_dump_json') else parsed or ''), status='incomplete' if reason else 'completed',
              incomplete_details=NS(reason=reason),
              usage=NS(input_tokens=100, output_tokens=50, total_tokens=150))


@pytest.mark.parametrize('schema,effort', [(JobProfile, 'low'), (Audit, 'medium')])
def test_luna_request_contract(provider, schema, effort):
    calls = []
    parsed = (asyncio.run(DemoProvider(delay=0).profile(SAMPLE_JOB)) if schema is JobProfile
              else Audit(claim_checks=[], proposal_checks=[], eligibility=[], questions=[]))
    async def parse(**kw):
        calls.append(kw)
        return response(parsed)
    provider.client.responses.create = parse
    assert provider.model == provider.writer_model == provider.reviewer_model == provider.cheap_model == 'gpt-5.6-luna'
    assert asyncio.run(provider._call('prompt', {}, schema, provider.model)) == parsed
    assert calls[0]['reasoning'] == {'effort': effort}
    assert calls[0]['max_output_tokens'] > provider._output_cap(schema)
    assert calls[0]['text']['format']['strict'] is True
    assert calls[0]['text']['format']['name'] == schema.__name__
    assert calls[0]['store'] is False
    assert 'temperature' not in calls[0]
    assert provider.tokens == 150


def test_legacy_model_keeps_request_contract(provider):
    captured = {}
    async def parse(**kw):
        captured.update(kw)
        return response(JobProfile(title='Engineer', keywords=[], eligibility=[], responsibilities=[]))
    provider.client.responses.create = parse
    asyncio.run(provider._call('prompt', {}, JobProfile, 'gpt-4.1-mini'))
    assert 'reasoning' not in captured
    assert captured['max_output_tokens'] == 5000


def test_truncation_retries_with_more_room_and_counts_both_calls(provider):
    calls = []
    async def parse(**kw):
        calls.append(kw)
        return response(reason='max_output_tokens') if len(calls) == 1 else response(JobProfile(title='Engineer', keywords=[], eligibility=[], responsibilities=[]))
    provider.client.responses.create = parse
    assert asyncio.run(provider._call('prompt', {}, JobProfile, provider.model)) .title == 'Engineer'
    assert calls[1]['max_output_tokens'] == calls[0]['max_output_tokens'] * 2
    assert provider.tokens == 300


def test_refusal_is_not_retried(provider):
    async def parse(**kw):
        return response()
    provider.client.responses.create = parse
    with pytest.raises(ValueError, match='refused'):
        asyncio.run(provider._call('prompt', {}, JobProfile, provider.model))
    assert provider.calls == 1


def test_budget_accounts_for_reasoning_allowance(provider):
    provider.limit = 8000
    with pytest.raises(BudgetExceeded):
        asyncio.run(provider._call('prompt', {}, JobProfile, provider.model))
    assert provider.calls == 0


def test_luna_pricing_and_unknown_family_members():
    assert price_for('gpt-5.6-luna') == (0.20, 1.20)
    assert price_for('openai/gpt-5.6-luna') == (0.20, 1.20)
    assert price_for('gpt-4.1-mini-2025-04-14') == (0.40, 1.60)
    assert price_for('gpt-5.6-unknown') is None
    assert price_for('gpt-5-unknown') is None


def test_invalid_json_usage_is_counted(provider):
    async def create(**kw):
        return response('{invalid')
    provider.client.responses.create = create
    with pytest.raises(ValueError, match='invalid JSON'):
        asyncio.run(provider._call('prompt', {}, JobProfile, provider.model))
    assert provider.tokens == 300
    assert provider.calls == 2
