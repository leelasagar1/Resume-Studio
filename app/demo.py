"""Fixed fictional fixture, explicitly labeled in the API and interface. No AI calls."""
import asyncio
from .models import Audit, ClaimCheck, EntryBullets, JobProfile, ProposedBullet, Proposals, Repair, Resume, Rewrite

SAMPLE_RESUME = '''Alex Morgan
alex.morgan@example.com | 555-010-2000 | Boston, MA
Data Analyst | Harbor Analytics
2022 - Present
Built SQL reports for weekly operations reviews used by regional operations managers.
Automated recurring reports with Python, reducing preparation time by 30% for the analytics team.
Presented monthly findings to operations stakeholders and documented follow-up actions for each review.
Data Analyst Intern | Northstar Labs
2021 - 2022
Cleaned and validated customer datasets using SQL before they reached the reporting warehouse.
Skills: SQL, Python, Excel, Tableau
Education
BS in Information Systems | Lakeside University | 2021
'''
SAMPLE_JOB = '''Data Analyst
We require experience using SQL to analyze operational data.
You will automate recurring reports with Python and build dashboards in Tableau.
You will present findings to stakeholders and document data quality checks.
Docker and Airflow experience is preferred.
Join our operations analytics team and help make reporting clear and useful.
'''


def sample_profile():
    return JobProfile.model_validate({'title': 'Data Analyst', 'keywords': [
        {'id': 'K1', 'term': 'SQL', 'aliases': [], 'kind': 'tool', 'priority': 'required'},
        {'id': 'K2', 'term': 'Python', 'aliases': [], 'kind': 'tool', 'priority': 'required'},
        {'id': 'K3', 'term': 'Tableau', 'aliases': [], 'kind': 'tool', 'priority': 'required'},
        {'id': 'K4', 'term': 'dashboards', 'aliases': ['dashboard'], 'kind': 'skill', 'priority': 'required'},
        {'id': 'K5', 'term': 'data quality', 'aliases': [], 'kind': 'method', 'priority': 'required'},
        {'id': 'K6', 'term': 'stakeholders', 'aliases': ['stakeholder'], 'kind': 'soft', 'priority': 'required'},
        {'id': 'K7', 'term': 'Docker', 'aliases': [], 'kind': 'tool', 'priority': 'preferred'},
        {'id': 'K8', 'term': 'Airflow', 'aliases': [], 'kind': 'tool', 'priority': 'preferred'},
    ], 'eligibility': [], 'responsibilities': ['automate recurring reports', 'present findings']})


def sample_draft(version=0):
    def c(id, text, *e, proposed=False):
        return {'id': id, 'text': text, 'evidence_ids': list(e), 'proposed': proposed}
    bullets = [c('j1b1', 'Built SQL reports for weekly operations reviews used by regional operations managers.', 'E5'),
               c('j1b3', 'Presented monthly findings to operations stakeholders and documented follow-up actions for each review.', 'E7')]
    if version:
        bullets.insert(1, c('j1b2', 'Automated recurring reports with Python, reducing preparation time by 30% for the analytics team.', 'E6'))
    bullets.append(c('j1p1', 'Containerized recurring Python reporting jobs with Docker so scheduled runs used one consistent environment.',
                     'E3', 'E6', proposed=True))
    missing = [{'text': 'Dashboards', 'evidence_ids': []}, {'text': 'Data quality', 'evidence_ids': []},
               {'text': 'Docker', 'evidence_ids': []}, {'text': 'Airflow', 'evidence_ids': []}]
    return Resume.model_validate({
        'name': c('name', 'Alex Morgan', 'E1'), 'headline': 'Data Analyst',
        'contact': [c('c1', 'alex.morgan@example.com | 555-010-2000 | Boston, MA', 'E2')],
        'summary': [c('s1', 'Data Analyst with SQL reporting and Python automation experience at Harbor Analytics and Northstar Labs, specializing in operations reporting.', 'E3', 'E5', 'E6', 'E8'),
                    c('s2', 'Automated recurring reports with Python, reducing preparation time by 30%, and built the SQL reports behind weekly operations reviews.', 'E5', 'E6'),
                    c('s3', 'Brings SQL, Python and Tableau skills with a record of presenting monthly findings to operations stakeholders.', 'E7', 'E11')],
        'sections': [
            {'heading': 'Experience', 'entries': [
                {'heading': c('j1', 'Data Analyst | Harbor Analytics', 'E3'),
                 'detail': c('j1d', '2022 - Present', 'E4'), 'bullets': bullets},
                {'heading': c('j2', 'Data Analyst Intern | Northstar Labs', 'E8'),
                 'detail': c('j2d', '2021 - 2022', 'E9'),
                 'bullets': [c('j2b1', 'Cleaned and validated customer datasets using SQL before they reached the reporting warehouse.', 'E10')]}
            ], 'skill_groups': []},
            {'heading': 'Skills', 'entries': [], 'skill_groups': [
                {'label': 'Analytics', 'items': [{'text': 'SQL', 'evidence_ids': ['E11']}, {'text': 'Python', 'evidence_ids': ['E11']},
                                                 {'text': 'Tableau', 'evidence_ids': ['E11']}, {'text': 'Excel', 'evidence_ids': ['E11']}]},
                {'label': 'To confirm', 'items': missing}]},
            {'heading': 'Education', 'entries': [{'heading': c('ed1', 'BS in Information Systems | Lakeside University | 2021', 'E13'),
                                                  'detail': None, 'bullets': []}], 'skill_groups': []}
        ]})


def rewrite_of(resume, skeleton):
    """Express a full sample Resume as the Rewrite a writer would return,
    mapping its bullets onto the skeleton's entry ids by heading text."""
    by_heading = {e['heading']: e['entry_id'] for s in skeleton['sections'] for e in s['entries']}
    entries = []
    for section in resume.sections:
        for entry in section.entries:
            entry_id = by_heading.get(entry.heading.text)
            if entry_id and entry.bullets:
                entries.append(EntryBullets(entry_id=entry_id, bullets=entry.bullets))
    return Rewrite(headline=resume.headline, summary=resume.summary, entries=entries,
                   skill_groups=[g for s in resume.sections for g in s.skill_groups])


class DemoProvider:
    provider_name = 'Demo'
    key_name = ''
    calls = 0
    tokens = 0

    def __init__(self, delay=.25):
        self.delay = delay

    async def profile(self, job_text):
        await asyncio.sleep(self.delay)
        return sample_profile()

    async def write(self, evidence, profile, keyword_status, skeleton, previous=None, feedback=None):
        await asyncio.sleep(self.delay)
        return rewrite_of(sample_draft(1 if previous else 0), skeleton)

    async def repair(self, evidence, profile, claims_to_fix, issues):
        await asyncio.sleep(self.delay)
        return Repair(claims=[c.model_copy(update={'text': c.text.lstrip('-–—•* ')}) for c in claims_to_fix])

    async def propose(self, evidence, profile, resume, gaps):
        await asyncio.sleep(self.delay)
        return Proposals(bullets=[ProposedBullet(entry_id=resume.sections[0].entries[0].heading.id, covers=gaps, context_evidence_ids=['E3', 'E5'],
            text='Documented data quality checks for weekly Tableau dashboards so operations reviewers could trust each report.')])

    async def audit(self, evidence, claims_to_audit, eligibility_rules, proposed=()):
        await asyncio.sleep(self.delay)
        return Audit(claim_checks=[ClaimCheck(claim_id=c.id, supported=True, reason='Matches the fictional source fixture.')
                                   for c in claims_to_audit],
                     proposal_checks=[ClaimCheck(claim_id=c.id, supported=True, reason='Plausible for the role; not evidence.')
                                      for c in proposed],
                     eligibility=[], questions=['Have you used Docker or Airflow? Keep them only if you can discuss them.'])

    async def close(self):
        pass
