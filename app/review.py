"""The step before writing: what the job asks for, and what the candidate says
about it.

The job's keyword list is extracted once per job description and remembered,
so the same posting always scores against the same list. The candidate can
then drop terms that are not real requirements, say which missing skills they
have (and where), and say which they do not have, before a single bullet is
written.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from .agents import PROFILE_PROMPT
from .models import JobProfile
from .scoring import ground_profile, keyword_hits, keyword_in_evidence, normalize
from .skeleton import bullet_evidence_ids, clean_markdown, make_evidence, parse_resume

_MEMORY: dict[str, dict] = {}


def data_dir() -> Path:
    """Local state that is not resume content: cached job analyses, the run log."""
    return Path(os.getenv('DATA_DIR') or Path(__file__).resolve().parent.parent / 'data')


def cache_key(provider, job_text: str) -> str:
    """Same model, same prompt, same posting (whitespace aside): same analysis."""
    parts = [str(getattr(provider, 'provider_name', '')), str(getattr(provider, 'model', '')), PROFILE_PROMPT,
             ' '.join(job_text.split())]
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


async def job_profile(provider, job_text: str) -> tuple[JobProfile, bool]:
    """The grounded job profile and whether it came from the cache.

    An empty or ungrounded extraction is an occasional model misfire (seen on
    ads that are only a requirements list), so a fresh extraction is retried.
    """
    cacheable = getattr(provider, 'cacheable', False)
    key = cache_key(provider, job_text) if cacheable else ''
    path = data_dir() / 'profiles' / f'{key}.json'
    if cacheable:
        stored = _MEMORY.get(key)
        if stored is None and path.is_file():
            try:
                stored = json.loads(path.read_text())
            except (OSError, ValueError):
                stored = None
        if stored is not None:
            try:
                profile = ground_profile(JobProfile.model_validate(stored), job_text)
                _MEMORY[key] = stored
                return profile, True
            except ValueError:
                pass  # a stale or damaged entry; extract again
    for attempt in range(3):
        try:
            raw = await provider.profile(job_text)
            stored = raw.model_dump()
            profile = ground_profile(raw, job_text)
            break
        except ValueError:
            if attempt == 2:
                raise
    if cacheable:
        _MEMORY[key] = stored
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(stored))
        except OSError:
            pass  # the cache is a convenience; a read-only disk must not fail a run
    return profile, False


def scorable(profile: JobProfile):
    return [k for k in profile.keywords if k.kind not in ('credential', 'soft')]


def analyze(profile: JobProfile, marked_text: str, additional_facts: str = '') -> dict:
    """What the review screen shows: each scored keyword and whether the
    resume already has it, and the roles a confirmed skill can be placed in."""
    evidence = make_evidence(clean_markdown(marked_text) + '\n' + additional_facts)
    skeleton = parse_resume(evidence, frozenset(bullet_evidence_ids(marked_text)))
    normalized = normalize(' '.join(evidence.values()))
    return {
        'title': profile.title,
        'keywords': [{'term': k.term, 'priority': k.priority, 'kind': k.kind,
                      'in_resume': keyword_in_evidence(k, normalized)} for k in scorable(profile)],
        'roles': [{'entry_id': e.id, 'heading': e.heading.text, 'detail': e.detail.text if e.detail else ''}
                  for s in skeleton.sections for e in s.entries if e.bullets],
    }


def _insert_after(marked_text: str, additions: dict[int, list[str]], tail: list[str]) -> str:
    """Insert lines after the given non-blank line numbers (the numbering
    make_evidence uses), then append `tail`."""
    out, number = [], 0
    for line in marked_text.splitlines():
        out.append(line)
        if clean_markdown(line).strip():
            number += 1
            out.extend(additions.get(number, []))
    return '\n'.join(out + tail)


def apply_decisions(profile: JobProfile, marked_text: str, request) -> tuple[JobProfile, str, list, dict]:
    """Apply the candidate's review decisions before anything is written.

    Returns the profile without the removed terms, the resume text with the
    confirmed skills written into it (a bullet under the named role when the
    candidate said what they did there, otherwise a Skills line), the keywords
    the candidate does not have, and counts for the run log.

    Confirmed content enters the resume as the candidate's own source text: it
    is cited and audited like every other line.
    """
    removed = {normalize(t) for t in request.removed_keywords}
    profile.keywords = [k for k in profile.keywords if normalize(k.term) not in removed]
    by_term = {normalize(k.term): k for k in scorable(profile)}
    declined = [by_term[t] for t in dict.fromkeys(normalize(t) for t in request.declined_keywords) if t in by_term]
    declined_terms = {normalize(k.term) for k in declined}

    evidence = make_evidence(clean_markdown(marked_text))
    skeleton = parse_resume(evidence, frozenset(bullet_evidence_ids(marked_text)))
    normalized = normalize(' '.join(evidence.values()))
    last_line = {e.id: int((e.bullets[-1] if e.bullets else e.detail or e.heading).eid[1:])
                 for s in skeleton.sections for e in s.entries if e.bullets}
    bullets: dict[int, list[str]] = {}
    skills, seen = [], set()
    for item in request.confirmed_skills:
        keyword = by_term.get(normalize(item.term))
        if keyword is None or keyword.term in seen or normalize(keyword.term) in declined_terms:
            continue
        seen.add(keyword.term)
        detail = ' '.join(item.detail.split()).lstrip('-–—•* ')
        if detail and len(detail.split()) >= 3 and item.entry_id in last_line:
            # The line must name the skill, or it would not evidence it.
            if not keyword_hits(keyword, normalize(detail)):
                detail = f'{detail.rstrip(".")} ({keyword.term})'
            bullets.setdefault(last_line[item.entry_id], []).append('• ' + detail.rstrip('.') + '.')
        elif not keyword_in_evidence(keyword, normalized):
            skills.append(keyword.term)
    additions, tail = dict(bullets), []
    if skills:
        line = 'Additional skills: ' + ', '.join(skills)
        skill_lines = [l for s in skeleton.sections if s.kind == 'skills' for l in s.lines]
        if skill_lines:
            additions.setdefault(int(skill_lines[-1].eid[1:]), []).append(line)
        else:
            tail.append(line)
    if additions or tail:
        marked_text = _insert_after(marked_text, additions, tail)
    summary = {'removed': len(removed), 'declined': len(declined),
               'confirmed_with_detail': sum(len(v) for v in bullets.values()), 'confirmed_skills': len(skills)}
    return profile, marked_text, declined, summary
