import asyncio
import io
import json
from types import SimpleNamespace
import pytest
from docx import Document
from app.agents import BudgetExceeded, OpenAIProvider
from app.demo import DemoProvider, SAMPLE_JOB, SAMPLE_RESUME, rewrite_of, sample_draft, sample_profile
from app.documents import extract_text, render_docx, MAX_FILE
from app.models import Audit, Claim, ClaimCheck, EligibilityCheck, EligibilityRule, GenerateRequest, JobProfile, Keyword, ProposalCheck, ProposedBullet, Proposals, Repair, Resume, SkillGroup, SkillItem, resume_plain_text
from app.scoring import ground_profile, inflected_occurrences, keyword_in_evidence, normalize, score_resume
from app.workflow import (MAX_PROPOSED_PER_ROLE, add_proposed_bullets, bullet_gaps, claims_to_audit, make_evidence, normalize_skills, run_workflow,
                          strip_unsupported, unverified_terms, validate_draft)


async def silent(*_):
    pass


def request(**kwargs):
    return GenerateRequest(resume_text=SAMPLE_RESUME, job_text=SAMPLE_JOB, confirmed=True, **kwargs)


def keyword(id, term, aliases=(), priority='required', kind='tool'):
    return Keyword(id=id, term=term, aliases=list(aliases), kind=kind, priority=priority)


# ------------------------------------------------------------------ scoring

def test_keyword_matching_is_exact_case_insensitive_and_alias_aware():
    profile = JobProfile(title='Data Scientist', keywords=[
        keyword('K1', 'scikit-learn', ['sklearn']), keyword('K2', 'Machine Learning', ['ML']),
        keyword('K3', 'SQL'), keyword('K4', 'R'), keyword('K5', 'C++')], eligibility=[], responsibilities=[])
    text = 'Built ML models with sklearn and PostgreSQL; wrote C++ and R scripts.'
    rows = {r['term']: r for r in score_resume(profile, text)['keywords']}
    assert rows['scikit-learn']['matched'] and rows['Machine Learning']['matched']
    assert rows['C++']['matched'] and rows['R']['matched']
    assert not rows['SQL']['matched'], 'PostgreSQL must not count as SQL'


def test_score_components_and_title():
    profile = JobProfile(title='Senior Data Scientist', keywords=[
        keyword('K1', 'Python'), keyword('K2', 'SQL'), keyword('K3', 'Docker', priority='preferred')],
        eligibility=[], responsibilities=[])
    full = 'Data Scientist\nname@example.com 555-123-4567\nExperience Python SQL Docker Skills Education'
    result = score_resume(profile, full)
    assert result['components']['required_keywords']['earned'] == 70
    assert result['components']['preferred_keywords']['earned'] == 15
    assert result['components']['job_title']['earned'] == 5  # role noun without seniority
    assert result['components']['format']['earned'] == 5
    assert result['score'] == 95
    assert score_resume(profile, 'Senior Data Scientist Python SQL Docker')['components']['job_title']['earned'] == 10
    no_preferred = JobProfile(title='X', keywords=[keyword('K1', 'Python')], eligibility=[], responsibilities=[])
    assert score_resume(no_preferred, 'python')['components']['required_keywords']['available'] == 85


def test_ground_profile_drops_keywords_absent_from_job_and_renumbers():
    profile = JobProfile(title='Data Analyst', keywords=[
        keyword('K9', 'SQL', ['or', 'Structured Query Language']), keyword('K3', 'Kubernetes'), keyword('K4', 'sql'), keyword('K5', 'Python')],
        eligibility=[EligibilityRule(id='x', text='3 years', kind='years')], responsibilities=[])
    grounded = ground_profile(profile, SAMPLE_JOB)
    assert [k.term for k in grounded.keywords] == ['SQL', 'Python']
    assert [k.id for k in grounded.keywords] == ['K1', 'K2']
    assert grounded.keywords[0].aliases == ['Structured Query Language'], 'stopword alias must be dropped'
    assert grounded.eligibility[0].id == 'L1'
    with pytest.raises(ValueError, match='grounded'):
        ground_profile(JobProfile(title='x', keywords=[keyword('K1', 'Fortran')], eligibility=[], responsibilities=[]), SAMPLE_JOB)


def test_degree_fields_become_credentials_and_leave_the_score():
    job = ('Senior Data Scientist. Proficiency in Python. Strong foundation in statistics and operations research. '
           "Bachelor's degree in Statistics, Economics, Computer Science or related field and 3 years' experience. "
           "Master's degree in Computer Science or Econometrics, Successful completion of assessments in Python, Spark, Scala, or R, "
           "Using open source frameworks (for example, scikit learn, tensorflow, torch), We value inclusive candidates")
    profile = JobProfile(title='Senior Data Scientist', keywords=[
        keyword('K1', 'Python'), keyword('K2', 'Computer Science', kind='domain'), keyword('K3', 'Econometrics', kind='domain', priority='preferred'),
        keyword('K4', 'statistics', kind='domain'), keyword('K5', "Bachelor's degree", kind='domain'),
        keyword('K6', 'scikit learn', ['scikit-learn'], priority='preferred'), keyword('K7', 'Scala', priority='preferred')], eligibility=[], responsibilities=[])
    grounded = ground_profile(profile, job)
    kinds = {k.term: k.kind for k in grounded.keywords}
    assert kinds['Computer Science'] == 'credential' and kinds['Econometrics'] == 'credential' and kinds["Bachelor's degree"] == 'credential'
    assert kinds['statistics'] == 'domain', 'appears outside a degree sentence too'
    assert kinds['scikit learn'] == 'tool' and kinds['Scala'] == 'tool', 'tools later in the same run-on sentence are not degree fields'
    vague = ground_profile(JobProfile(title='x', keywords=[keyword('K1', 'deploy scalable models', kind='method'), keyword('K2', 'business domains', kind='domain'),
                                                          keyword('K3', 'optimization models', kind='method'), keyword('K4', 'SQL')],
                                      eligibility=[], responsibilities=[]),
                           'deploy scalable models across business domains using optimization models and SQL')
    assert {k.term: k.kind for k in vague.keywords} == {'deploy scalable models': 'soft', 'business domains': 'soft',
                                                        'optimization models': 'method', 'SQL': 'tool'}
    result = score_resume(grounded, 'Python statistics scikit-learn Scala')
    assert {r['term'] for r in result['keywords']} == {'Python', 'statistics', 'scikit learn', 'Scala'}, 'degree fields leave the score'
    assert result['score'] >= 85


def test_evidenced_skills_use_the_candidates_spelling_and_labels_merge():
    evidence = make_evidence('Skills: Python, Scikit-learn, TensorFlow, PyTorch, Kafka')
    profile = JobProfile(title='x', keywords=[keyword('K1', 'torch', ['PyTorch']), keyword('K2', 'scikit learn', ['scikit-learn', 'sklearn']),
                                              keyword('K3', 'tensorflow')], eligibility=[], responsibilities=[])
    draft = sample_draft(1)
    draft.sections[1].skill_groups = [
        SkillGroup(label='Required Methods and Tools', items=[SkillItem(text='torch', evidence_ids=[]), SkillItem(text='scikit learn', evidence_ids=[])]),
        SkillGroup(label='Preferred Methods and Tools', items=[SkillItem(text='tensorflow', evidence_ids=[]), SkillItem(text='kafka', evidence_ids=[]),
                                                              SkillItem(text='optimization models', evidence_ids=[])])]
    normalize_skills(draft, evidence, profile)
    groups = draft.sections[1].skill_groups
    flat = [i.text for g in groups for i in g.items]
    assert flat == ['PyTorch', 'Scikit-learn', 'TensorFlow'], 'Kafka is an orphan and "optimization models" is not a job keyword here'
    assert [g.label for g in groups] == ['Additional Skills']


def test_tidy_layout_fixes_headings_labels_and_dates():
    from app.workflow import tidy_layout
    draft = sample_draft(1)
    draft.sections[0].heading = 'WORK EXPERIENCE'
    draft.sections[1].skill_groups[1].label = 'Required Methods and Soft Skills'
    entry = draft.sections[0].entries[0]
    entry.heading.text = 'George Mason University, MS in Data Analytics Engineering January 2023 - December 2024'; entry.detail = None
    tidy_layout(draft)
    assert draft.sections[0].heading == 'Work Experience'
    assert draft.sections[1].skill_groups[1].label == 'Methods & Domains'
    assert entry.heading.text == 'George Mason University, MS in Data Analytics Engineering' and entry.detail.text == 'January 2023 - December 2024'


def test_inflected_matching_is_for_evidence_only():
    assert inflected_occurrences('validation', 'trained and validated a model') == 1
    assert inflected_occurrences('feature engineering', 'built a feature store to engineer signals') == 0
    assert inflected_occurrences('scalability', 'designed scalable models') == 1
    assert inflected_occurrences('data', 'database technologies') == 0
    kw = keyword('K1', 'Validation')
    assert keyword_in_evidence(kw, normalize('Trained and validated a supervised model'))
    assert not score_resume(JobProfile(title='x', keywords=[kw], eligibility=[], responsibilities=[]),
                            'Trained and validated a supervised model')['keywords'][0]['matched'], 'ATS score stays exact'


# --------------------------------------------------------------- validation

def test_skill_items_are_evidenced_deterministically():
    evidence = make_evidence(SAMPLE_RESUME)
    draft = sample_draft(1)
    draft.sections[1].skill_groups[0].items[0].evidence_ids = ['E1']  # wrong citation for SQL
    draft.sections[1].skill_groups[1].items[1].evidence_ids = ['E11']  # forged citation for Docker
    normalize_skills(draft, evidence, sample_profile())
    items = {i.text: i.evidence_ids for g in draft.sections[1].skill_groups for i in g.items}
    assert items['SQL'] and all(eid in evidence and 'SQL' in evidence[eid] for eid in items['SQL'])
    assert items['Docker'] == []
    assert set(unverified_terms(draft)) == {'Dashboards', 'Data quality', 'Docker', 'Airflow'}
    from app.models import Section, SkillGroup
    draft.sections.insert(0, Section(heading='PROFESSIONAL SUMMARY', entries=[], skill_groups=[]))
    assert [s.heading for s in normalize_skills(draft, evidence, sample_profile()).sections] == ['Experience', 'Skills', 'Education']


def test_normalize_removes_credential_items_splits_skills_and_dates():
    evidence = make_evidence(SAMPLE_RESUME)
    draft = sample_draft(1)
    draft.sections[1].skill_groups[1].items += [
        SkillItem(text="Master's degree in Machine Learning", evidence_ids=[]),
        SkillItem(text='assessments in Python, Spark, Scala, or R', evidence_ids=[]),
        SkillItem(text='Successful completion of one or more assessments in Python', evidence_ids=[]),
        SkillItem(text='analytics related field', evidence_ids=[])]
    # skills nested inside the experience section, dates glued to the heading
    draft.sections[0].skill_groups = [SkillGroup(label='Extra', items=[SkillItem(text='Excel', evidence_ids=[]), SkillItem(text='Tableau', evidence_ids=[])])]
    draft.sections[0].entries[0].heading.text = 'Data Analyst | Harbor Analytics 2022 - Present | Boston'
    draft.sections[0].entries[0].detail = None
    normalize_skills(draft, evidence, sample_profile())
    items = [i.text for s in draft.sections for g in s.skill_groups for i in g.items]
    assert "Master's degree in Machine Learning" not in items and not any('assessments' in i for i in items)
    assert 'analytics related field' not in items
    assert 'Tableau' in items and 'Excel' not in items and draft.sections[0].skill_groups == []
    assert [s.heading for s in draft.sections] == ['Experience', 'Skills', 'Education']
    entry = draft.sections[0].entries[0]
    assert entry.heading.text == 'Data Analyst | Harbor Analytics' and entry.detail.text == '2022 - Present | Boston'
    assert entry.detail.evidence_ids == entry.heading.evidence_ids
    assert validate_draft(draft, evidence, sample_profile()) == []


def test_compound_skill_items_need_every_part_evidenced():
    evidence = make_evidence('Skills: PySpark, FAISS, Pinecone\nBuilt vector databases for search.')
    profile = JobProfile(title='x', keywords=[keyword('K1', 'Spark', ['PySpark']), keyword('K2', 'vector databases')], eligibility=[], responsibilities=[])
    draft = sample_draft(1)
    draft.sections[1].skill_groups[0].items = [
        SkillItem(text='Spark (PySpark)', evidence_ids=[]), SkillItem(text='Vector Databases (FAISS, Pinecone)', evidence_ids=[]),
        SkillItem(text='Vector Databases (FAISS, Milvus)', evidence_ids=['E1'])]
    normalize_skills(draft, evidence, profile)
    items = {i.text: i.evidence_ids for i in draft.sections[1].skill_groups[0].items}
    assert items['Spark (PySpark)'] == ['E1']
    assert set(items['Vector Databases (FAISS, Pinecone)']) == {'E1', 'E2'}
    assert items['Vector Databases (FAISS, Milvus)'] == []


def test_validate_draft_rejects_novel_numbers_and_unevidenced_keywords():
    evidence = make_evidence(SAMPLE_RESUME)
    profile = sample_profile()
    draft = sample_draft(1)
    assert validate_draft(draft, evidence, profile) == []
    draft.sections[0].entries[0].bullets[0].text = 'Built 12 SQL reports with Docker for weekly reviews used by regional operations managers.'
    issues = validate_draft(draft, evidence, profile)
    assert any('number 12' in i for i in issues)
    assert any('"Docker" is not in the candidate evidence' in i for i in issues)
    draft = sample_draft(1)
    draft.sections[0].entries[0].bullets[0].text = 'Built SQL reports and dashboard-style weekly operations reviews used by regional operations managers.'
    assert any('"dashboards" is not in the candidate evidence' in i for i in validate_draft(draft, evidence, profile))
    inflected = JobProfile(title='x', keywords=[keyword('K1', 'Validation')], eligibility=[], responsibilities=[])
    draft = sample_draft(1)
    draft.sections[0].entries[1].bullets[0].text = 'Performed validation of customer datasets using SQL before they reached the reporting warehouse.'
    assert validate_draft(draft, evidence, inflected) == [], 'evidence says "validated"; inflection is honest'
    draft = sample_draft(1)
    draft.sections[0].entries[0].bullets[0].evidence_ids = ['E99']
    assert any('invalid evidence' in i for i in validate_draft(draft, evidence, profile))
    draft = sample_draft(1)
    draft.sections[0].entries[0].bullets[0].text = '- Built SQL reports for weekly operations reviews used by regional operations managers.'
    assert any('bullet marker' in i for i in validate_draft(draft, evidence, profile))


def test_dropped_employer_or_degree_is_caught_locally():
    evidence = make_evidence(SAMPLE_RESUME)
    profile = sample_profile()
    draft = sample_draft(1)
    del draft.sections[0].entries[1]  # Northstar Labs 2021 - 2022 gone
    issues = validate_draft(draft, evidence, profile)
    assert any('2021 - 2022' in i and 'missing' in i for i in issues)
    draft = sample_draft(1)
    draft.sections[0].entries[0].detail = None  # Harbor dates gone
    assert any('2022 - Present' in i for i in validate_draft(draft, evidence, profile))
    draft = sample_draft(1)
    draft.sections[0].entries[1].detail.text = '2021 – 2022'
    assert validate_draft(draft, evidence, profile) == [], 'dash style differences are fine'
    draft = sample_draft(1)
    draft.sections[0].entries[1].detail.text = '2021 – 2023'
    assert any('2021 - 2022' in i for i in validate_draft(draft, evidence, profile)), 'changed dates are caught'
    draft = sample_draft(1)
    draft.sections[0].heading = 'id'
    assert any('real section heading' in i for i in validate_draft(draft, evidence, profile))
    from app.workflow import date_ranges
    assert date_ranges('Con Edison, Senior AI/ML Engineer\tJanuary 2025 – Present | NYC') == ['jan 2025 present']
    assert date_ranges('GMR Institute, BTech\t\t  June 2016 – May 2020') == date_ranges('06/2016 - may 2020') == ['jun 2016 may 2020']
    assert date_ranges('Databricks Certified — June 2025') == []
    from app.workflow import entry_keys
    assert entry_keys('Databricks Certified Data Engineer Associate — June 2025') == ['single jun 2025']
    assert entry_keys('AWS Certified (March 2024)') == ['single mar 2024']
    assert entry_keys('Led migration completed in March 2024 across teams and finished the rollout, reducing incidents and costs by a lot for everyone involved') == []
    cert_evidence = make_evidence(SAMPLE_RESUME + 'Certifications\nTableau Desktop Specialist — June 2025\n')
    assert any('Tableau Desktop Specialist' in i for i in validate_draft(sample_draft(1), cert_evidence, profile))


def test_only_non_verbatim_claims_are_audited():
    evidence = make_evidence(SAMPLE_RESUME)
    ids = {c.id for c in claims_to_audit(sample_draft(1), evidence)}
    assert 'name' not in ids and 'j1b1' not in ids and 'j1p1' not in ids
    assert 's1' in ids


def test_proposed_bullets_may_use_missing_keywords_but_no_numbers_and_only_as_bullets():
    evidence = make_evidence(SAMPLE_RESUME)
    profile = sample_profile()
    draft = sample_draft(1)
    assert validate_draft(draft, evidence, profile) == []
    draft.sections[0].entries[0].bullets[-1].text = 'Containerized 3 reporting jobs with Docker.'
    assert any('must not contain numbers' in i for i in validate_draft(draft, evidence, profile))
    draft = sample_draft(1)
    draft.summary[0].proposed = True
    assert any('only experience bullets may be proposed' in i for i in validate_draft(draft, evidence, profile))


def test_unsuitable_proposals_are_dropped_before_confirmation():
    provider = ScriptedProvider(audits=[{'j1p1': False}])
    result = asyncio.run(run_workflow(request(max_revisions=0), provider, silent))
    assert [p['id'] for p in result['proposals']] == ['e2p2'] and result['report']['factual_pass']
    assert 'j1p1' not in [b.id for b in Resume.model_validate(result['resume']).sections[0].entries[0].bullets]
    assert result['report']['proposed_count'] == 1 and result['report']['rejected_proposals'] == 1


def test_propose_step_targets_required_keywords_absent_from_experience():
    evidence = make_evidence(SAMPLE_RESUME)
    profile = sample_profile()
    draft = sample_draft(1)
    assert bullet_gaps(draft, profile) == ['dashboards', 'data quality']  # Docker is preferred; stakeholders is soft

    class Sloppy(ScriptedProvider):
        async def propose(self, evidence, profile, resume, gaps, roles=None):
            return Proposals(bullets=[
                ProposedBullet(entry_id='j1', covers=['dashboards'], context_evidence_ids=['E3'],
                               text='Built 5 Tableau dashboards for operations.', fit_reason='Routine work for this team.'),          # number: dropped
                ProposedBullet(entry_id='nope', covers=['dashboards'], context_evidence_ids=['E3'],
                               text='Built dashboards.', fit_reason='Routine work for this team.'),                                   # unknown entry: dropped
                ProposedBullet(entry_id='j2', covers=['data quality'], context_evidence_ids=['E99'],
                               text='Introduced data quality checks on incoming customer datasets before analysis.', fit_reason='Routine work for this team.'),
                ProposedBullet(entry_id='j2', covers=['dashboards'], context_evidence_ids=['E8'],
                               text='Maintained weekly operations dashboards for the intern reporting rotation across two regions.', fit_reason='Routine work for this team.'),
                ProposedBullet(entry_id='j2', covers=['dashboards'], context_evidence_ids=['E8'],
                               text='Refreshed dashboards each Monday for the whole operations team before reviews.', fit_reason='Routine work for this team.'),
                ProposedBullet(entry_id='j2', covers=['dashboards'], context_evidence_ids=['E8'],
                               text='Published the operations dashboard pack to the shared drive before each weekly review.',
                               fit_reason='Routine work for this team.')])  # fourth for j2: over the per-role cap
    draft, added, _ = asyncio.run(add_proposed_bullets(Sloppy(), evidence, profile, draft, bullet_gaps(draft, profile)))
    assert added == MAX_PROPOSED_PER_ROLE and bullet_gaps(draft, profile) == []
    assert sum(b.proposed for b in draft.sections[0].entries[1].bullets) == MAX_PROPOSED_PER_ROLE
    assert sum(b.proposed for b in draft.sections[0].entries[0].bullets) == 1, 'j1 already had one writer proposal'
    sloppy_draft = draft
    full = sample_draft(1)
    for n in range(2):
        full.sections[0].entries[0].bullets.append(
            Claim(id=f'j1p{n + 2}', text=f'Automated dashboard refresh job number {n} with Docker for the operations team.',
                  evidence_ids=['E3'], proposed=True))
    full, added, _ = asyncio.run(add_proposed_bullets(ScriptedProvider(), evidence, profile, full, ['dashboards']))
    assert added == 0, 'three proposed bullets already: per-role cap reached'

    class Misplaced(ScriptedProvider):
        async def propose(self, evidence, profile, resume, gaps, roles=None):
            return Proposals(bullets=[ProposedBullet(entry_id='ed1', covers=gaps, context_evidence_ids=['E13'],
                                                     text='Applied dashboards practices during coursework projects for the analytics program.', fit_reason='Routine work for this team.')])
    draft = sample_draft(1)
    draft, added, _ = asyncio.run(add_proposed_bullets(Misplaced(), evidence, profile, draft, ['dashboards']))
    assert added == 0 and draft.sections[2].entries[0].bullets == [], 'never under a degree or certification'

    class Borrower(ScriptedProvider):
        async def propose(self, evidence, profile, resume, gaps, roles=None):
            return Proposals(bullets=[ProposedBullet(entry_id='j1', covers=gaps, context_evidence_ids=['E3'],
                                                     text='Cleaned dashboards data with Northstar Labs tooling and SQL validation for Lakeside University reviews.', fit_reason='Routine work for this team.')])
    draft = sample_draft(1)
    draft.sections[0].entries[0].bullets = [b for b in draft.sections[0].entries[0].bullets if not b.proposed]
    draft, added, _ = asyncio.run(add_proposed_bullets(Borrower(), evidence, profile, draft, ['dashboards']))
    assert added == 0, 'names Northstar Labs and Lakeside University, which belong to other entries'

    j2 = sloppy_draft.sections[0].entries[1].bullets
    assert [b.id for b in j2] == ['j2b1', 'j2p1', 'j2p2', 'j2p3'] and all(b.proposed for b in j2[1:])
    assert j2[1].evidence_ids == ['E8'], 'invalid context ids fall back to the entry heading evidence'
    assert validate_draft(draft, evidence, profile) == []


def test_strip_unsupported_removes_bullets_but_not_headings():
    draft = sample_draft(1)
    stripped = strip_unsupported(draft, {'j1b2', 's1'})
    assert [b.id for b in stripped.sections[0].entries[0].bullets] == ['j1b1', 'j1b3', 'j1p1']
    assert [c.id for c in stripped.summary] == ['s2', 's3']
    assert strip_unsupported(draft, {'j1'}) is None


# ----------------------------------------------------------------- workflow

class ScriptedProvider(DemoProvider):
    """Demo provider whose audit verdicts and drafts can be scripted per call."""
    def __init__(self, drafts=None, audits=None):
        super().__init__(delay=0)
        self.drafts = list(drafts or [])
        self.audits = list(audits or [])
        self.write_calls = []
        self.audit_calls = 0

    async def write(self, evidence, profile, keyword_status, skeleton, previous=None, feedback=None):
        self.write_calls.append({'status': keyword_status, 'feedback': feedback, 'skeleton': skeleton})
        return rewrite_of(self.drafts.pop(0) if self.drafts else sample_draft(1), skeleton)

    async def propose(self, evidence, profile, resume, gaps, roles=None):
        self.propose_calls = getattr(self, 'propose_calls', 0) + 1
        first = resume.sections[0].entries[0].heading.id
        return Proposals(bullets=[ProposedBullet(entry_id=first, covers=gaps, context_evidence_ids=['E3'],
                                                 text='Applied ' + ' and '.join(gaps) + ' practices in weekly reporting work.', fit_reason='Routine work for this team.')])

    async def audit(self, evidence, claims_to_audit, eligibility_rules, proposed=()):
        self.audit_calls += 1
        verdict = self.audits.pop(0) if self.audits else {}
        return Audit(claim_checks=[ClaimCheck(claim_id=c.id, supported=verdict.get(c.id, True), reason='scripted')
                                   for c in claims_to_audit],
                     proposal_checks=[ProposalCheck(claim_id=c.id, practical=verdict.get(c.id, True), relevant=True, period_consistent=True, reason='scripted')
                                      for c in proposed],
                     eligibility=[EligibilityCheck(rule_id=r.id, status='unknown', reason='scripted') for r in eligibility_rules],
                     questions=[])


def test_proposals_cover_preferred_skills_and_see_each_role_stack():
    from app.workflow import role_profiles, skill_gaps
    evidence = make_evidence(SAMPLE_RESUME)
    profile = sample_profile()
    draft = sample_draft(1)
    draft.sections[0].entries[0].bullets = [b for b in draft.sections[0].entries[0].bullets if not b.proposed]
    # required (dashboards, data quality) first, then preferred (Docker, Airflow)
    assert skill_gaps(draft, profile) == ['dashboards', 'data quality', 'Docker', 'Airflow']
    seen = {}

    class Recorder(ScriptedProvider):
        async def propose(self, evidence, profile, resume, gaps, roles=None):
            seen['gaps'], seen['roles'] = gaps, roles
            return Proposals(bullets=[])
    asyncio.run(add_proposed_bullets(Recorder(), evidence, profile, draft, skill_gaps(draft, profile)))
    assert 'Airflow' in seen['gaps'], 'preferred skills get a bullet too'
    harbor = next(r for r in seen['roles'] if r['company_and_title'].startswith('Data Analyst | Harbor'))
    assert harbor['period'] == '2022 - Present' and harbor['years'].startswith('2022-')
    assert 'SQL' in harbor['technologies_already_shown'] and 'Python' in harbor['technologies_already_shown']
    assert 'Built' not in harbor['technologies_already_shown'], 'verbs are not technologies'
    assert [r['entry_id'] for r in seen['roles']] == ['j1', 'j2'], 'only roles with real bullets host proposals'


def test_anachronistic_and_unreviewed_proposals_are_rejected():
    from app.workflow import anachronistic, role_period
    evidence = make_evidence(SAMPLE_RESUME)
    profile = sample_profile()
    draft = sample_draft(1)
    intern = draft.sections[0].entries[1]          # Northstar Labs, 2021 - 2022
    assert role_period(intern) == (2021, 2022)
    assert anachronistic('Built dashboards with LangChain and GPT-4 for the reporting team.', intern) == ['gpt 4']
    assert anachronistic('Built dashboards with SQL and Airflow for the reporting team.', intern) == []
    # A tool released in the role's final year is allowed; only later ones are not.
    assert anachronistic('Used LangChain to summarize customer datasets for the team.', intern) == []

    class TooNew(ScriptedProvider):
        async def propose(self, evidence, profile, resume, gaps, roles=None):
            return Proposals(bullets=[ProposedBullet(entry_id='j2', covers=['dashboards'], context_evidence_ids=['E8'],
                                                     text='Built GPT-4 powered dashboards for the intern reporting rotation each week.',
                                                     fit_reason='Reporting work.')])
    draft, added, reasons = asyncio.run(add_proposed_bullets(TooNew(), evidence, profile, draft, ['dashboards']))
    assert added == 0 and reasons == {}, 'GPT-4 did not exist during a 2021-2022 role'


def test_proposal_in_the_wrong_company_and_filler_proposals_are_removed():
    from app.workflow import company_words, prune_pointless_proposals
    evidence = make_evidence(SAMPLE_RESUME)
    profile = sample_profile()
    assert company_words(sample_draft(1).sections[0].entries[1]) == {'Northstar', 'Labs'}

    class WrongCompany(ScriptedProvider):
        async def propose(self, evidence, profile, resume, gaps, roles=None):
            return Proposals(bullets=[ProposedBullet(entry_id='j1', covers=['dashboards'], context_evidence_ids=['E3'],
                                                     text='Maintained weekly operations dashboards for the regional review meetings.',
                                                     fit_reason='Northstar Labs owns the reporting rotation, so dashboards are routine there.')])
    draft = sample_draft(1)
    draft, added, _ = asyncio.run(add_proposed_bullets(WrongCompany(), evidence, profile, draft, ['dashboards']))
    assert added == 0, 'the rationale names another employer, so the placement is wrong'

    draft = sample_draft(1)
    entry = draft.sections[0].entries[0]
    entry.bullets.append(Claim(id='j1p9', text='Collaborated with cross-functional stakeholders to align reporting priorities each quarter.',
                               evidence_ids=['E3'], proposed=True))
    draft, removed = prune_pointless_proposals(draft, evidence, profile)
    ids = [b.id for b in draft.sections[0].entries[0].bullets]
    assert removed == 1 and 'j1p9' not in ids, 'an unconfirmed bullet that adds no missing skill is filler'
    assert 'j1p1' in ids, 'the Docker proposal still earns its place'


@pytest.mark.parametrize('verdict', [
    {'practical': False, 'relevant': True, 'period_consistent': True},
    {'practical': True, 'relevant': False, 'period_consistent': True},
    {'practical': True, 'relevant': True, 'period_consistent': False},
])
def test_reviewer_rejects_a_proposal_that_fails_any_single_check(verdict):
    class Picky(ScriptedProvider):
        async def audit(self, evidence, claims_to_audit, eligibility_rules, proposed=()):
            self.audit_calls += 1
            return Audit(claim_checks=[ClaimCheck(claim_id=c.id, supported=True, reason='ok') for c in claims_to_audit],
                         proposal_checks=[ProposalCheck(claim_id=c.id, reason='judged', **verdict) for c in proposed],
                         eligibility=[], questions=[])
    result = asyncio.run(run_workflow(request(max_revisions=0), Picky(), silent))
    assert result['proposals'] == [] and result['report']['proposed_count'] == 0
    assert 'Applied' not in resume_plain_text(Resume.model_validate(result['resume']))


class NoProposals(ScriptedProvider):
    """For tests about draft selection, stopping and budget, where added
    bullets would change the score under test."""
    async def propose(self, evidence, profile, resume, gaps, roles=None):
        return Proposals(bullets=[])


def test_workflow_reaches_target_and_asks_to_confirm_unevidenced_skills():
    provider = ScriptedProvider()
    result = asyncio.run(run_workflow(request(max_revisions=2), provider, silent))
    assert result['report']['score'] == 100 and result['report']['factual_pass']
    assert result['before']['score'] < 90
    assert set(result['unverified_terms']) == {'Dashboards', 'Data quality', 'Docker', 'Airflow'}
    assert result['floor_score'] < result['report']['score']
    assert result['docx'] is None and '_pending' in result
    assert [p['id'] for p in result['proposals']] == ['j1p1', 'e2p2'] and result['proposals'][0]['terms'] == ['Docker']
    # Preferred skills are covered too, not just required ones: Airflow rides along.
    assert set(result['proposals'][1]['terms']) == {'dashboards', 'data quality', 'Airflow'}
    assert result['proposals'][1]['fit_reason'] and result['proposals'][1]['review_note'] == 'scripted'
    assert result['proposals'][0]['role_context'].startswith('Data Analyst | Harbor Analytics')
    assert result['history'][0]['bullet_gaps'] == [] and result['history'][0]['proposed'] == 2
    assert 'Target ATS score reached' in result['stop_reason']
    assert len(provider.write_calls) == 1 and provider.audit_calls == 1
    assert provider.write_calls[0]['status']['missing'], 'first write must receive the missing keyword list'


def test_writer_receives_missing_keywords_and_factual_issues_on_revision():
    weak = sample_draft(0)
    weak.sections[1].skill_groups = [weak.sections[1].skill_groups[0]]  # no unevidenced additions
    provider = NoProposals(drafts=[weak, sample_draft(1)], audits=[{'s1': False}, {}])
    result = asyncio.run(run_workflow(request(max_revisions=1), provider, silent))
    feedback = provider.write_calls[1]['feedback']
    assert {m['term'] for m in feedback['missing_keywords']} >= {'Airflow'}
    assert feedback['factual_issues'] and feedback['factual_issues'][0].startswith('s1:')
    assert result['selected_version'] == 1 and result['report']['score'] == 100


def test_unsupported_statements_are_stripped_when_revisions_run_out():
    draft = sample_draft(1)
    draft.sections[0].entries[0].bullets[1].text = 'Automated recurring Python reports for the analytics team, cutting preparation time by 30%.'
    provider = ScriptedProvider(drafts=[draft], audits=[{'j1b2': False}])
    result = asyncio.run(run_workflow(request(max_revisions=0), provider, silent))
    ids = [b.id for b in Resume.model_validate(result['resume']).sections[0].entries[0].bullets]
    assert 'j1b2' not in ids
    assert result['report']['factual_pass'] and 'removed automatically' in result['stop_reason']


def test_clean_later_draft_beats_stripped_earlier_draft_on_tie():
    draft = sample_draft(1)
    draft.sections[0].entries[0].bullets[1].text = 'Automated recurring Python reports for the analytics team, cutting preparation time by 30%.'
    provider = ScriptedProvider(drafts=[draft, sample_draft(1)], audits=[{'j1b2': False}, {}])
    result = asyncio.run(run_workflow(request(max_revisions=1, target_score=100), provider, silent))
    assert result['selected_version'] == 1 and 'removed automatically' not in result['stop_reason']
    assert len(Resume.model_validate(result['resume']).sections[0].entries[0].bullets) == 5


def test_source_skills_always_survive_and_credential_terms_are_dropped():
    from app.skeleton import parse_resume, skeleton_payload, assemble, split_items
    assert split_items('Azure (ML, Databricks, ADLS), AWS (SageMaker, S3), PySpark; Kafka') == ['Azure (ML, Databricks, ADLS)', 'AWS (SageMaker, S3)', 'PySpark', 'Kafka']
    evidence = make_evidence(SAMPLE_RESUME)
    skeleton = parse_resume(evidence)
    draft = sample_draft(1)
    draft.sections[1].skill_groups = [SkillGroup(label='Programming', items=[SkillItem(text='Python', evidence_ids=[]),
                                                                                SkillItem(text='Information Technology', evidence_ids=[])])]
    resume = assemble(skeleton, rewrite_of(draft, skeleton_payload(skeleton)), evidence)
    items = {i.text for s in resume.sections for g in s.skill_groups for i in g.items}
    assert {'SQL', 'Excel', 'Tableau', 'Python'} <= items, 'dropped source skills are merged back'
    normalize_skills(resume, evidence, sample_profile())
    items = {i.text for s in resume.sections for g in s.skill_groups for i in g.items}
    assert {'SQL', 'Tableau', 'Python'} <= items and 'Excel' not in items, 'then curated: Excel is not in the job and not in a bullet'
    resume.sections[1].skill_groups.append(SkillGroup(label='Extra', items=[SkillItem(text='python', evidence_ids=[]), SkillItem(text='SQL', evidence_ids=['E11'])]))
    normalize_skills(resume, evidence, sample_profile())
    flat = [i.text for s in resume.sections for g in s.skill_groups for i in g.items]
    assert flat.count('Python') == 1 and flat.count('SQL') == 1 and 'python' not in flat, 'one home per skill'
    resume.sections[1].skill_groups.append(SkillGroup(label='Extra', items=[SkillItem(text='torch', evidence_ids=[]), SkillItem(text='PyTorch', evidence_ids=['E11'])]))
    torch_profile = sample_profile(); torch_profile.keywords.append(keyword('K8', 'torch', ['PyTorch'], priority='preferred'))
    normalize_skills(resume, make_evidence(SAMPLE_RESUME.replace('Excel, Tableau', 'Excel, Tableau, PyTorch')), torch_profile)
    flat = [i.text for s in resume.sections for g in s.skill_groups for i in g.items]
    assert 'PyTorch' in flat and 'torch' not in flat, 'unverified fragment of an evidenced skill is dropped'
    long_source = make_evidence(SAMPLE_RESUME.replace('Skills: SQL, Python, Excel, Tableau', 'Skills: NLP (Named Entity Recognition, Text Classification, Information Extraction), SQL'))
    resume = assemble(parse_resume(long_source), rewrite_of(sample_draft(1), skeleton_payload(parse_resume(long_source))), long_source)
    nlp_profile = sample_profile(); nlp_profile.keywords.append(keyword('K8', 'NLP', ['natural language processing']))
    normalize_skills(resume, long_source, nlp_profile)
    assert 'NLP (Named Entity Recognition, Text Classification, Information Extraction)' in [i.text for s in resume.sections for g in s.skill_groups for i in g.items]
    profile = sample_profile()
    profile.keywords.append(keyword('K9', 'Information Technology', kind='credential'))
    profile.keywords.append(keyword('K10', 'Excel', kind='credential'))  # misclassified, but the candidate lists Excel
    normalize_skills(resume, evidence, profile)
    items = {i.text for s in resume.sections for g in s.skill_groups for i in g.items}
    assert 'Information Technology' not in items and 'SQL' in items, 'evidenced relevant skills are never dropped'
    draft = sample_draft(1)
    draft.summary[0].text = 'Data analyst.'
    assert any('summary too short' in i for i in validate_draft(draft, evidence, profile))


def test_skills_curated_like_an_hr_reviewer():
    from app.workflow import curate_skills, MAX_UNVERIFIED_SKILLS
    evidence = make_evidence(SAMPLE_RESUME.replace('Skills: SQL, Python, Excel, Tableau', 'Skills: SQL, Python, Excel, Tableau, Kafka, Docker'))
    profile = sample_profile()
    profile.keywords += [keyword(f'K{n}', f'tool{n}', kind='tool', priority='preferred') for n in range(20, 40)]
    draft = sample_draft(1)
    draft.sections[1].skill_groups = [
        SkillGroup(label='Core', items=[SkillItem(text='SQL', evidence_ids=['E11']), SkillItem(text='Kafka', evidence_ids=['E11']),
                                        SkillItem(text='Docker', evidence_ids=['E11']), SkillItem(text='stakeholders', evidence_ids=['E7']),
                                        SkillItem(text='validate models', evidence_ids=[]), SkillItem(text='Dashboards', evidence_ids=[]),
                                        SkillItem(text='advanced dashboards', evidence_ids=[])]),
        SkillGroup(label='More', items=[SkillItem(text=f'tool{n}', evidence_ids=[]) for n in range(20, 40)])]
    curate_skills(draft, evidence, profile)
    flat = [i.text for g in draft.sections[1].skill_groups for i in g.items]
    assert 'SQL' in flat and 'Docker' in flat, 'job keywords stay'
    assert 'Kafka' not in flat, 'orphan: not in the job, not demonstrated in a bullet'
    assert 'stakeholders' not in flat, 'soft skills are not skill items'
    assert 'validate models' not in flat, 'writer-invented unverified phrase'
    assert 'Dashboards' in flat and 'advanced dashboards' not in flat, 'near duplicates collapse to the shorter job term'
    assert sum(1 for g in draft.sections[1].skill_groups for i in g.items if not i.evidence_ids) <= MAX_UNVERIFIED_SKILLS
    # compound redundancy and soft-skill detection
    evidence = make_evidence(SAMPLE_RESUME.replace('Skills: SQL, Python, Excel, Tableau', 'Skills: AWS SageMaker, AWS (SageMaker, S3, Glue), Vector Databases (FAISS), Vector Databases (FAISS, Pinecone)'))
    profile = JobProfile(title='x', keywords=[keyword('K1', 'AWS'), keyword('K2', 'vector databases'), keyword('K3', 'decision-making', kind='method')],
                         eligibility=[], responsibilities=[])
    profile = ground_profile(profile, 'AWS, vector databases and decision-making required')
    assert {k.term: k.kind for k in profile.keywords}['decision-making'] == 'soft'
    draft = sample_draft(1)
    draft.sections[1].skill_groups = [SkillGroup(label='Cloud', items=[
        SkillItem(text='AWS SageMaker', evidence_ids=[]), SkillItem(text='AWS (SageMaker, S3, Glue)', evidence_ids=[]),
        SkillItem(text='Vector Databases (FAISS)', evidence_ids=[]), SkillItem(text='Vector Databases (FAISS, Pinecone)', evidence_ids=[]),
        SkillItem(text='decision-making', evidence_ids=[])])]
    normalize_skills(draft, evidence, profile)
    assert [i.text for g in draft.sections[1].skill_groups for i in g.items] == ['AWS (SageMaker, S3, Glue)', 'Vector Databases (FAISS, Pinecone)']
    assert all(len(g.items) <= 8 for g in draft.sections[1].skill_groups)


def test_source_groups_fold_into_similar_writer_groups_and_group_cap():
    from app.skeleton import merge_source_skills, best_group
    writer = [SkillGroup(label='NLP & AI Technologies', items=[SkillItem(text='LangChain', evidence_ids=['E1'])]),
              SkillGroup(label='Data & Cloud Platforms', items=[SkillItem(text='Databricks', evidence_ids=['E1'])])]
    source = [SkillGroup(label='NLP & AI', items=[SkillItem(text='RAG', evidence_ids=['E2'])]),
              SkillGroup(label='Cloud & Data', items=[SkillItem(text='Kafka', evidence_ids=['E2'])]),
              SkillGroup(label='MLOps', items=[SkillItem(text='MLflow', evidence_ids=['E2'])])]
    merged = merge_source_skills(writer, source)
    assert [g.label for g in merged] == ['NLP & AI Technologies', 'Data & Cloud Platforms', 'MLOps']
    assert [i.text for i in merged[0].items] == ['LangChain', 'RAG'] and [i.text for i in merged[1].items] == ['Databricks', 'Kafka']
    assert best_group('Languages & ML Frameworks', [SkillGroup(label='Machine Learning Methods', items=[])]).label == 'Machine Learning Methods'
    from app.workflow import curate_skills, MAX_SKILL_GROUPS
    evidence = make_evidence(SAMPLE_RESUME)
    profile = sample_profile()
    profile.keywords += [keyword(f'K{n}', f'tool{n}', kind='tool') for n in range(20, 34)]
    draft = sample_draft(1)
    draft.sections[1].skill_groups = [SkillGroup(label=f'Group {n}', items=[SkillItem(text=f'tool{n}', evidence_ids=[]), SkillItem(text=f'tool{n + 7}', evidence_ids=[])])
                                      for n in range(20, 27)]
    curate_skills(draft, evidence, profile)
    assert len(draft.sections[1].skill_groups) <= MAX_SKILL_GROUPS
    from app.workflow import summary_issues
    draft = sample_draft(1)
    assert summary_issues(draft, evidence, sample_profile()) == []
    draft.summary = [draft.summary[0].model_copy(update={'text': ' '.join(['word'] * 45)})]
    issues = summary_issues(draft, evidence, sample_profile())
    assert any('single sentence' in i for i in issues) and any('must name the role' in i for i in issues) and any('job keyword' in i for i in issues)
    draft = sample_draft(1)
    draft.summary[0].text = 'Leveraged SQL and Python at Harbor Analytics as a Data Analyst delivering weekly operations reviews.'
    issues = summary_issues(draft, evidence, sample_profile())
    assert any('remove "leveraged"' in i for i in issues) and any('opens with an identity sentence' in i for i in issues)


def test_skeleton_headings_and_dates_are_verbatim_and_never_audited():
    """The writer cannot rename a company or change a date: Python copies every
    entry heading and detail from the source and the model only supplies bullets."""
    class Vandal(ScriptedProvider):
        async def write(self, evidence, profile, keyword_status, skeleton, previous=None, feedback=None):
            draft = sample_draft(1)
            draft.sections[0].entries[0].heading.text = 'Lead Data Analyst | Harbor Analytics'
            draft.sections[0].entries[0].detail.text = '2019 - Present'
            del draft.sections[0].entries[1]  # tries to drop Northstar
            self.write_calls.append({'status': keyword_status, 'feedback': feedback, 'skeleton': skeleton})
            return rewrite_of(draft, skeleton)
    result = asyncio.run(run_workflow(request(max_revisions=0), Vandal(), silent))
    resume = Resume.model_validate(result['resume'])
    entries = resume.sections[0].entries
    assert [e.heading.text for e in entries] == ['Data Analyst | Harbor Analytics', 'Data Analyst Intern | Northstar Labs']
    assert [e.detail.text for e in entries] == ['2022 - Present', '2021 - 2022']
    assert entries[1].bullets and entries[1].bullets[0].text.startswith('Cleaned and validated'), 'skipped entry keeps source bullets'
    assert resume.name.text == 'Alex Morgan' and resume.contact[0].text.startswith('alex.morgan@')
    audited = {c.id for c in claims_to_audit(resume, make_evidence(SAMPLE_RESUME))}
    assert not any(e.heading.id in audited or (e.detail and e.detail.id in audited) for e in entries)


def test_local_failure_uses_small_repair_call_not_a_rewrite():
    bad = sample_draft(1)
    bad.sections[0].entries[0].bullets[0].text = '- Built SQL reports for weekly operations reviews used by regional operations managers.'
    provider = ScriptedProvider(drafts=[bad])
    result = asyncio.run(run_workflow(request(max_revisions=1), provider, silent))
    assert len(provider.write_calls) == 1 and provider.audit_calls == 1
    assert result['history'][0]['audited'] is True and result['selected_version'] == 0
    text = Resume.model_validate(result['resume']).sections[0].entries[0].bullets[0].text
    assert text.startswith('Built')


def test_bullet_standards_and_role_floor():
    from app.workflow import bullet_standard_issues, source_bullet_counts, trim_bullets, MAX_BULLETS
    from app.models import Claim
    evidence = make_evidence(SAMPLE_RESUME)
    profile = sample_profile()
    assert source_bullet_counts(evidence) == {'2022 present': 3, '2021 2022': 1}
    short = Claim(id='x', text='Built SQL reports.', evidence_ids=['E5'], proposed=False)
    weak = Claim(id='y', text='Responsible for building SQL reports for the weekly operations review meetings each week.', evidence_ids=['E5'], proposed=False)
    cliche = Claim(id='z', text='Delivered results-driven SQL reporting for weekly operations reviews across regional teams.', evidence_ids=['E5'], proposed=False)
    assert any('too short' in i for i in bullet_standard_issues(short))
    assert any('action verb' in i for i in bullet_standard_issues(weak))
    assert any('cliche' in i for i in bullet_standard_issues(cliche))
    present = Claim(id='w', text='Design optimization models with advanced algorithms to enhance operational planning outcomes.', evidence_ids=['E5'], proposed=True)
    assert any('not past tense' in i for i in bullet_standard_issues(present))
    assert bullet_standard_issues(Claim(id='v', text='Led the migration of reporting jobs to Python, cutting preparation time for the analytics team.', evidence_ids=['E5'], proposed=False)) == []
    draft = sample_draft(1)
    del draft.sections[0].entries[0].bullets[:2]  # Harbor drops to 2 of its 3 source bullets
    assert any('keep at least 3 evidence-backed bullets' in i for i in validate_draft(draft, evidence, profile))
    from app.workflow import backfill_bullets
    draft, added = backfill_bullets(draft, evidence, profile)
    ids = [b.id for b in draft.sections[0].entries[0].bullets]
    assert added == 2 and ids == ['j1b3', 'j1p1', 'j1s1', 'j1s2']
    assert draft.sections[0].entries[0].bullets[2].evidence_ids == ['E6'], 'the Python/30% line is the most relevant source line'
    assert validate_draft(draft, evidence, profile) == []
    # Trim: 9 bullets -> MAX_BULLETS, least relevant evidence bullets go first, proposed stay, unique keywords stay.
    draft = sample_draft(1)
    entry = draft.sections[0].entries[0]
    for n in range(5):
        entry.bullets.append(Claim(id=f'f{n}', text=f'Organized team lunch number {n} and booked the meeting rooms for the quarterly offsite.',
                                   evidence_ids=['E3'], proposed=False))
    draft, removed = trim_bullets(draft, profile)
    ids = [b.id for b in entry.bullets]
    assert len(ids) == MAX_BULLETS and removed == 2
    assert 'j1p1' in ids and 'j1b1' in ids and 'j1b2' in ids and 'j1b3' in ids
    assert sum(i.startswith('f') for i in ids) == 3
    # never trims evidence bullets below the floor even when over the cap
    entry.bullets += [Claim(id='q1', text='Coordinated the quarterly volunteer day and ordered supplies for the analytics floor.', evidence_ids=['E3'], proposed=False)]
    for n in range(3):
        entry.bullets.append(Claim(id=f'p{n}', text=f'Proposed responsibility number {n} covering an unevidenced job keyword for confirmation.', evidence_ids=['E3'], proposed=True))
    draft, removed = trim_bullets(draft, profile)
    assert removed == 2 and sum(not b.proposed for b in entry.bullets) == 5


def test_failed_repair_falls_back_to_full_rewrite():
    class NoRepair(ScriptedProvider):
        async def repair(self, evidence, profile, claims_to_fix, issues):
            return Repair(claims=claims_to_fix)
    bad = sample_draft(1)
    bad.sections[0].entries[0].bullets[0].text = 'Built 12 SQL reports for weekly operations reviews.'
    provider = NoRepair(drafts=[bad, sample_draft(1)])
    result = asyncio.run(run_workflow(request(max_revisions=1), provider, silent))
    assert len(provider.write_calls) == 2 and provider.audit_calls == 1
    assert provider.write_calls[1]['feedback']['local_issues']
    assert provider.write_calls[0]['status']['missing'][0]['placement'].startswith('ONLY')
    assert result['history'][0]['audited'] is False and result['selected_version'] == 1


def test_best_version_is_kept_when_a_revision_regresses():
    worse = sample_draft(1)
    worse.sections[1].skill_groups = [worse.sections[1].skill_groups[0]]
    worse.sections[0].entries[0].bullets = [b for b in worse.sections[0].entries[0].bullets if not b.proposed]
    worse.summary[0].text = 'Data analyst.'
    first = sample_draft(0)
    first.sections[1].skill_groups[1].items.pop()  # Airflow missing: below the 100 target
    provider = NoProposals(drafts=[first, worse])
    result = asyncio.run(run_workflow(request(max_revisions=1, target_score=100), provider, silent))
    assert result['selected_version'] == 0
    assert result['history'][1]['score'] < result['history'][0]['score']


def test_stops_when_a_revision_barely_moves_the_score():
    first = sample_draft(0)
    first.sections[1].skill_groups[1].items.pop()  # Airflow missing: 92.5, below the 100 target
    provider = NoProposals(drafts=[first, first.model_copy(deep=True), sample_draft(1)])
    result = asyncio.run(run_workflow(request(max_revisions=2, target_score=100), provider, silent))
    assert len(result['history']) == 2 and 'would not raise the score' in result['stop_reason']


def test_budget_keeps_only_audited_version():
    class Budgeted(NoProposals):
        async def write(self, *args, **kwargs):
            if self.write_calls:
                raise BudgetExceeded('budget')
            return await super().write(*args, **kwargs)
    first = sample_draft(0)
    first.sections[1].skill_groups[1].items.pop()
    provider = Budgeted(drafts=[first])
    result = asyncio.run(run_workflow(request(max_revisions=2, target_score=100), provider, silent))
    assert 'token budget' in result['stop_reason'] and result['selected_version'] == 0


def test_missed_audit_claims_are_retried_then_treated_conservatively():
    class Forgetful(ScriptedProvider):
        async def audit(self, evidence, claims_to_audit, eligibility_rules, proposed=()):
            self.audit_calls += 1
            return Audit(claim_checks=[ClaimCheck(claim_id=c.id, supported=True, reason='ok')
                                       for c in claims_to_audit if c.id != 's1'],
                         proposal_checks=[ProposalCheck(claim_id=c.id, practical=True, relevant=True, period_consistent=True, reason='ok') for c in proposed],
                         eligibility=[], questions=[])
    provider = Forgetful(drafts=[sample_draft(1)])
    result = asyncio.run(run_workflow(request(max_revisions=0), provider, silent))
    assert provider.audit_calls == 2
    assert [c.id for c in Resume.model_validate(result['resume']).summary] == ['s2', 's3'], 'only the unaudited s1 is stripped'


# ---------------------------------------------------------------- documents

def test_docx_roundtrip_and_upload_extraction():
    draft = sample_draft(1)
    with pytest.raises(ValueError, match='Confirm or remove'):
        render_docx(draft)
    from app.confirmation import without_terms
    from app.workflow import drop_claims
    final = drop_claims(without_terms(draft, set(unverified_terms(draft))), {'j1p1'})
    data, checks = render_docx(final)
    text = '\n'.join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    assert 'Alex Morgan' in text and 'Analytics: SQL, Python, Tableau, Excel' in text and 'Docker' not in text
    assert checks['text_round_trip']
    assert extract_text('resume.docx', data).startswith('Alex Morgan')
    assert resume_plain_text(final).count('SQL') >= 2


@pytest.mark.parametrize('name,data', [('r.exe', b'x' * 100), ('r.pdf', b'notpdf'), ('big.txt', b'x' * (MAX_FILE + 1)), ('s.txt', b'hi')])
def test_bad_uploads(name, data):
    with pytest.raises(ValueError):
        extract_text(name, data)


# ------------------------------------------------------------------ adapter

def test_openai_adapter_contract_and_usage(monkeypatch):
    captured = {}
    profile = sample_profile()

    async def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_parsed=profile, usage=SimpleNamespace(total_tokens=321))
    monkeypatch.setattr('app.agents.AsyncOpenAI', lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    monkeypatch.setenv('MAX_RUN_TOKENS', '60000')
    provider = OpenAIProvider()
    assert asyncio.run(provider.profile(SAMPLE_JOB)) == profile
    assert captured['store'] is False and captured['text_format'] is JobProfile
    assert provider.tokens == 321 and provider.calls == 1


def test_writer_runs_one_small_call_per_role_and_revises_only_flagged_roles(monkeypatch):
    from app.models import EntryRewrite, HeaderRewrite, Claim
    from app.skeleton import parse_resume, skeleton_payload, assemble
    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        schema = kwargs['text_format']
        body = json.loads(kwargs['input'][1]['content'])
        if schema is EntryRewrite:
            eid = body['role']['entry_id']
            parsed = EntryRewrite(bullets=[Claim(id=f'{eid}b1', text='Built SQL reports for weekly operations reviews used by regional operations managers.',
                                                 evidence_ids=[body['role']['source_bullets'][0]['eid']], proposed=False)])
        else:
            parsed = HeaderRewrite(headline='Data Analyst', summary=[Claim(id='s1', text='Data analyst with SQL reporting experience across two employers and weekly operations reviews for regional managers and stakeholders.', evidence_ids=['E3'], proposed=False)],
                                   skill_groups=[SkillGroup(label='Core', items=[SkillItem(text='SQL', evidence_ids=['E11'])])])
        return SimpleNamespace(output_parsed=parsed, usage=SimpleNamespace(input_tokens=100, output_tokens=50))
    monkeypatch.setattr('app.agents.AsyncOpenAI', lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    monkeypatch.setenv('WRITER_MODEL', 'gpt-4.1-mini')
    provider = OpenAIProvider()
    evidence = make_evidence(SAMPLE_RESUME)
    skeleton = parse_resume(evidence)
    status = {'matched': [], 'missing': []}
    rewrite = asyncio.run(provider.write(evidence, sample_profile(), status, skeleton_payload(skeleton)))
    assert len(calls) == 3, 'two roles with bullets + one header call'
    assert {c['text_format'] for c in calls} == {EntryRewrite, HeaderRewrite}
    entry_call = next(c for c in calls if c['text_format'] is EntryRewrite)
    assert 'E12' not in entry_call['input'][0]['content'], 'role calls get only their own evidence slice (plus header lines)'
    assert provider.usage()['estimated_cost_usd'] > 0 and provider.usage()['input_tokens'] == 300
    draft = assemble(skeleton, rewrite, evidence)
    calls.clear()
    feedback = {'factual_issues': ['e2b1: overstated'], 'local_issues': [], 'missing_keywords': []}
    rewrite2 = asyncio.run(provider.write(evidence, sample_profile(), status, skeleton_payload(skeleton), draft, feedback))
    assert len(calls) == 1 and calls[0]['text_format'] is EntryRewrite and json.loads(calls[0]['input'][1]['content'])['role']['entry_id'] == 'e2'
    assert [e.entry_id for e in rewrite2.entries] == ['e2', 'e4'] and rewrite2.headline == 'Data Analyst'


def test_audit_runs_in_parallel_chunks_with_rules_once(monkeypatch):
    calls = []

    async def parse(**kwargs):
        calls.append(kwargs['input'][1]['content'])
        return SimpleNamespace(output_parsed=Audit(claim_checks=[], proposal_checks=[], eligibility=[], questions=['q']),
                               usage=SimpleNamespace(total_tokens=1))
    monkeypatch.setattr('app.agents.AsyncOpenAI', lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    provider = OpenAIProvider()
    claims = [SimpleNamespace(model_dump=lambda i=i: {'id': f'c{i}'}) for i in range(45)]
    merged = asyncio.run(provider.audit({'E1': 'x'}, claims, [EligibilityRule(id='L1', text='r', kind='years')]))
    assert len(calls) == 2 and '"eligibility_rules":[{' in calls[0] and '"eligibility_rules":[]' in calls[1]
    assert '"proposed_claims":[]' in calls[1]
    assert merged.questions == ['q', 'q']


def test_budget_prevents_provider_dispatch(monkeypatch):
    monkeypatch.setattr('app.agents.AsyncOpenAI', lambda **kwargs: SimpleNamespace())
    monkeypatch.setenv('MAX_RUN_TOKENS', '1')
    provider = OpenAIProvider()
    with pytest.raises(BudgetExceeded):
        asyncio.run(provider.profile(SAMPLE_JOB))
    assert provider.calls == 0
