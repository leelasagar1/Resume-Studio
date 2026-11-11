"""The keyword review that precedes writing, the per-posting cache and the run log."""
import asyncio
import json

from fastapi.testclient import TestClient

from app.demo import SAMPLE_JOB, SAMPLE_RESUME, DemoProvider, sample_profile
from app.models import ConfirmedSkill, GenerateRequest
from app.review import analyze, apply_decisions, data_dir, job_profile
from app.scoring import ground_profile
from app.skeleton import bullet_evidence_ids, clean_markdown, make_evidence, parse_resume
from app.workflow import run_workflow

HEADERS = {'X-Resume-Studio': '1'}


async def silent(*_):
    pass


def request(**kwargs):
    return GenerateRequest(resume_text=SAMPLE_RESUME, job_text=SAMPLE_JOB, confirmed=True, **kwargs)


def grounded():
    return ground_profile(sample_profile(), SAMPLE_JOB)


class Cacheable(DemoProvider):
    cacheable = True
    model = 'test-model'

    def __init__(self):
        super().__init__(delay=0)
        self.profile_calls = 0

    async def profile(self, job_text):
        self.profile_calls += 1
        return await super().profile(job_text)


def test_review_lists_keywords_against_the_resume_and_the_roles_to_place_them_in():
    result = analyze(grounded(), SAMPLE_RESUME)
    status = {k['term']: k['in_resume'] for k in result['keywords']}
    assert status['SQL'] and status['Tableau'] and not status['Docker'] and not status['Airflow']
    assert 'stakeholders' not in status, 'soft skills are not scored, so they are not reviewed'
    assert [r['heading'] for r in result['roles']] == ['Data Analyst | Harbor Analytics', 'Data Analyst Intern | Northstar Labs']


def test_a_confirmed_skill_with_a_role_becomes_a_source_bullet_of_that_role():
    role = analyze(grounded(), SAMPLE_RESUME)['roles'][0]['entry_id']
    decisions = request(confirmed_skills=[
        ConfirmedSkill(term='Airflow', entry_id=role, detail='Scheduled the nightly report refresh'),
        ConfirmedSkill(term='Docker'), ConfirmedSkill(term='Kubernetes')])
    profile, text, declined, summary = apply_decisions(grounded(), SAMPLE_RESUME, decisions)
    evidence = make_evidence(clean_markdown(text))
    skeleton = parse_resume(evidence, frozenset(bullet_evidence_ids(text)))
    first = skeleton.sections[0].entries[0]
    # The line names the skill even though the candidate's sentence did not.
    assert first.bullets[-1].text == 'Scheduled the nightly report refresh (Airflow).'
    assert len(skeleton.sections[0].entries[1].bullets) == 1, 'the other role is untouched'
    assert 'Additional skills: Docker' in evidence.values(), 'no role given: listed under Skills'
    assert summary == {'removed': 0, 'declined': 0, 'confirmed_with_detail': 1, 'confirmed_skills': 1}
    assert not any('Kubernetes' in line for line in evidence.values()), 'only the job’s own keywords can be confirmed'


def test_removed_keywords_leave_the_score_and_declined_ones_are_never_offered():
    result = asyncio.run(run_workflow(request(removed_keywords=['Airflow'], declined_keywords=['Docker']),
                                      DemoProvider(delay=0), silent))
    terms = {k['term'] for k in result['report']['keywords']}
    assert 'Airflow' not in terms and 'Docker' in terms
    text = json.dumps(result['resume'])
    assert 'Docker' not in text, 'a declined skill appears nowhere: not in Skills, not in a proposed bullet'
    assert 'Docker' not in result.get('unverified_terms', [])
    assert all('Docker' not in p['text'] for p in result.get('proposals', []))
    assert result['review_decisions'] == {'removed': 1, 'declined': 1, 'confirmed_with_detail': 0, 'confirmed_skills': 0}


def test_confirmed_skills_count_as_evidence_and_need_no_second_confirmation():
    result = asyncio.run(run_workflow(request(confirmed_skills=[ConfirmedSkill(term='Airflow'), ConfirmedSkill(term='Docker')]),
                                      DemoProvider(delay=0), silent))
    assert not {'Airflow', 'Docker'} & set(result.get('unverified_terms', []))
    # "Before" stays the resume as submitted.
    plain = asyncio.run(run_workflow(request(), DemoProvider(delay=0), silent))
    assert result['before']['score'] == plain['before']['score']


def test_the_same_posting_is_analysed_once():
    provider = Cacheable()
    first, cached_first = asyncio.run(job_profile(provider, SAMPLE_JOB))
    second, cached_second = asyncio.run(job_profile(provider, '  ' + SAMPLE_JOB.replace('\n', '\n\n')))
    assert (cached_first, cached_second) == (False, True) and provider.profile_calls == 1
    assert [k.term for k in first.keywords] == [k.term for k in second.keywords]
    # A fresh process reads it from disk; a different posting is analysed again.
    from app import review
    review._MEMORY.clear()
    assert asyncio.run(job_profile(Cacheable(), SAMPLE_JOB))[1] is True
    other = SAMPLE_JOB + 'SQL and Python are essential.'
    assert asyncio.run(job_profile(provider, other))[1] is False and provider.profile_calls == 2
    # Providers that are not cacheable (the fictional demo) never touch the cache.
    assert asyncio.run(job_profile(DemoProvider(delay=0), SAMPLE_JOB))[1] is False


def test_each_run_reports_its_steps_and_duration():
    result = asyncio.run(run_workflow(request(), DemoProvider(delay=0), silent))
    assert {t['step'] for t in result['timings']} >= {'profile', 'write', 'audit'}
    assert result['seconds'] >= 0 and result['profile_cached'] is False


def test_analyze_endpoint_and_run_log(monkeypatch):
    from app import main
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setitem(main.PROVIDERS, 'openai', Cacheable)
    for name in ('provider_name', 'key_name'):
        monkeypatch.setattr(Cacheable, name, 'OpenAI' if name == 'provider_name' else 'OPENAI_API_KEY', raising=False)
    body = {'resume_text': SAMPLE_RESUME, 'job_text': SAMPLE_JOB, 'provider': 'openai', 'model': 'test-model'}
    with TestClient(main.app) as client:
        response = client.post('/api/analyze', headers=HEADERS, json=body)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data['cached'] is False and any(k['term'] == 'Docker' and not k['in_resume'] for k in data['keywords'])
        assert client.post('/api/analyze', headers=HEADERS, json=body).json()['cached'] is True
        started = client.post('/api/generate', headers=HEADERS, json={**body, 'confirmed': True, 'declined_keywords': ['Docker']})
        assert started.status_code == 202, started.text
        for _ in range(200):
            job = client.get(f"/api/jobs/{started.json()['id']}", headers=HEADERS).json()
            if job['status'] in ('complete', 'needs_confirmation', 'error'):
                break
        assert job['status'] in ('complete', 'needs_confirmation'), job
        assert job['result']['profile_cached'] is True, 'the generate step reuses the reviewed analysis'
    records = [json.loads(line) for line in (data_dir() / 'runs.jsonl').read_text().splitlines()]
    record = records[-1]
    assert record['event'] == 'generated' and record['review']['declined'] == 1 and record['steps']
    assert 'Alex Morgan' not in json.dumps(records) and 'Harbor' not in json.dumps(records), 'no resume content in the log'


def test_a_short_resume_gets_no_style_verdict_and_does_not_crash(monkeypatch):
    from app.workflow import source_style_report
    short = ('Alex Morgan\nalex@example.com | 555-010-2000\nData Analyst | Harbor Analytics\n2022 - Present\n'
             'Built SQL reports for the weekly operations review meetings each Monday.\n')
    assert source_style_report(make_evidence(short)) is None
    # A run whose resume is too short to judge completes without a verdict.
    monkeypatch.setattr('app.workflow.source_style_report', lambda evidence: None)
    result = asyncio.run(run_workflow(request(), DemoProvider(delay=0), silent))
    assert result['source_style'] is None and result['report']['score'] > 0
    monkeypatch.undo()
    # Five similar sentences are not a habit; the sample resume is not flagged.
    assert source_style_report(make_evidence(SAMPLE_RESUME))['reads_machine_written'] is False
