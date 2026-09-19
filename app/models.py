from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:/-]*$'


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class GenerateRequest(StrictModel):
    resume_text: str = Field(min_length=80, max_length=16000)
    job_text: str = Field(min_length=80, max_length=14000)
    additional_facts: str = Field(default='', max_length=4000)
    confirmed: Literal[True]
    max_revisions: int = Field(default=2, ge=0, le=3)
    target_score: int = Field(default=90, ge=50, le=100)
    provider: Literal['openai', 'openrouter', 'claude'] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=160, pattern=MODEL_PATTERN)
    writer_model: str | None = Field(default=None, min_length=1, max_length=160, pattern=MODEL_PATTERN)
    reviewer_model: str | None = Field(default=None, min_length=1, max_length=160, pattern=MODEL_PATTERN)


# ---------------------------------------------------------------- job profile
# Every structured-output model below lists every field as required. Strict
# JSON-schema modes (OpenAI, OpenRouter, Anthropic) reject optional fields.

class Keyword(StrictModel):
    id: str
    term: str
    aliases: list[str]
    kind: Literal['skill', 'tool', 'method', 'domain', 'credential', 'soft']
    priority: Literal['required', 'preferred']


class EligibilityRule(StrictModel):
    id: str
    text: str
    kind: Literal['degree', 'years', 'location', 'authorization', 'license', 'clearance', 'other']


class JobProfile(StrictModel):
    title: str
    keywords: list[Keyword] = Field(max_length=80)
    eligibility: list[EligibilityRule] = Field(max_length=10)
    responsibilities: list[str] = Field(max_length=15)


# --------------------------------------------------------------------- resume

class Claim(StrictModel):
    """proposed=True marks a bullet the writer added to cover job keywords the
    evidence does not contain. evidence_ids then cite the role context only.
    Such bullets are disclosed as Content to Confirm and removed unless the
    candidate confirms them."""
    id: str
    text: str
    evidence_ids: list[str]
    proposed: bool


class Entry(StrictModel):
    heading: Claim
    detail: Claim | None
    bullets: list[Claim]


class SkillItem(StrictModel):
    """A skill keyword. Empty evidence_ids marks a job keyword the candidate
    has not evidenced; it is shown for confirmation and removable."""
    text: str
    evidence_ids: list[str]


class SkillGroup(StrictModel):
    label: str
    items: list[SkillItem]


class Section(StrictModel):
    heading: str = Field(min_length=1, max_length=80)
    entries: list[Entry]
    skill_groups: list[SkillGroup]


class Resume(StrictModel):
    name: Claim
    headline: str
    contact: list[Claim]
    summary: list[Claim]
    sections: list[Section]


class EntryBullets(StrictModel):
    entry_id: str
    bullets: list[Claim]


class EntryRewrite(StrictModel):
    """One role's bullets, written by a small focused call."""
    bullets: list[Claim]


class HeaderRewrite(StrictModel):
    """Headline, summary and skill groups, written by one small call."""
    headline: str
    summary: list[Claim]
    skill_groups: list[SkillGroup]


class Rewrite(StrictModel):
    """What the writer returns. Name, contact, entry headings and dates are
    never part of it; Python copies those from the source verbatim."""
    headline: str
    summary: list[Claim]
    entries: list[EntryBullets]
    skill_groups: list[SkillGroup]


class Repair(StrictModel):
    claims: list[Claim]


class ProposedBullet(StrictModel):
    entry_id: str
    text: str
    covers: list[str]
    context_evidence_ids: list[str]
    # Why this work is practical for that company, role, seniority and period.
    # Shown to the candidate so they can judge it before confirming.
    fit_reason: str


class Proposals(StrictModel):
    bullets: list[ProposedBullet]


# ---------------------------------------------------------------------- audit

class ClaimCheck(StrictModel):
    claim_id: str
    supported: bool
    reason: str


class EligibilityCheck(StrictModel):
    rule_id: str
    status: Literal['met', 'unmet', 'unknown']
    reason: str


class ProposalCheck(StrictModel):
    """A proposed bullet is kept only when all three verdicts are true. None of
    them is evidence that the candidate did the work."""
    claim_id: str
    practical: bool          # realistic work for that company, team and seniority
    relevant: bool           # matters for the target job
    period_consistent: bool  # the tools and practices existed during that employment
    reason: str


class Audit(StrictModel):
    claim_checks: list[ClaimCheck]
    proposal_checks: list[ProposalCheck]
    eligibility: list[EligibilityCheck]
    questions: list[str]


class ConfirmationRequest(StrictModel):
    accepted_terms: list[str] = Field(default_factory=list, max_length=120)
    accepted_claim_ids: list[str] = Field(default_factory=list, max_length=40)
    reviewed_all: Literal[True]


def claims(resume: Resume) -> list[Claim]:
    result = [resume.name, *resume.contact, *resume.summary]
    for section in resume.sections:
        for entry in section.entries:
            result.append(entry.heading)
            if entry.detail:
                result.append(entry.detail)
            result.extend(entry.bullets)
    return result


def proposed_claims(resume: Resume) -> list[Claim]:
    return [c for c in claims(resume) if c.proposed]


def skill_items(resume: Resume) -> list[SkillItem]:
    return [item for section in resume.sections for group in section.skill_groups for item in group.items]


def resume_plain_text(resume: Resume) -> str:
    """Text as an ATS parser would see it: what the DOCX renders."""
    lines = [resume.name.text, resume.headline, ' | '.join(c.text for c in resume.contact)]
    lines.extend(c.text for c in resume.summary)
    for section in resume.sections:
        lines.append(section.heading)
        for entry in section.entries:
            lines.append(entry.heading.text)
            if entry.detail:
                lines.append(entry.detail.text)
            lines.extend(b.text for b in entry.bullets)
        for group in section.skill_groups:
            lines.append(group.label + ': ' + ', '.join(i.text for i in group.items))
    return '\n'.join(line for line in lines if line)
