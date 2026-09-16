"""Deterministic ATS-style keyword scoring. No model judgment involved.

Applicant tracking systems rank resumes mostly by exact (case-insensitive)
presence of the job's keywords. This module reproduces that behaviour so the
headline score is reproducible and the writer can be told exactly what is
missing.
"""
import re
from .models import JobProfile, Keyword

REQUIRED_POINTS = 70
PREFERRED_POINTS = 15
TITLE_POINTS = 10
FORMAT_POINTS = 5

_TYPO = str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-', ' ': ' '})
# An alias that is also an ordinary word (operations research -> "OR") would
# match prose everywhere, so such aliases are discarded.
STOPWORDS = {'a', 'an', 'and', 'as', 'at', 'be', 'by', 'do', 'for', 'if', 'in', 'is', 'it', 'no', 'of', 'on',
             'or', 'so', 'the', 'to', 'up', 'us', 'we', 'all', 'any', 'can', 'not', 'our', 'out', 'per', 'you'}


def normalize(text: str) -> str:
    text = text.translate(_TYPO).lower()
    text = re.sub(r'[^a-z0-9+#.]+', ' ', text)
    text = re.sub(r'(?<![a-z0-9])\.|\.(?![a-z0-9])', ' ', text)  # keep node.js, 2.2
    return ' '.join(text.split())


def variants(keyword: Keyword) -> list[str]:
    seen = []
    for raw in [keyword.term, *keyword.aliases]:
        v = normalize(raw)
        if v and v not in seen:
            seen.append(v)
    return seen


def count_occurrences(variant: str, normalized_text: str) -> int:
    if not variant:
        return 0
    pattern = r'(?<![a-z0-9+#])' + re.escape(variant) + r'(?:s|es)?(?![a-z0-9+#])'
    return len(re.findall(pattern, normalized_text))


def keyword_hits(keyword: Keyword, normalized_text: str) -> int:
    return sum(count_occurrences(v, normalized_text) for v in variants(keyword))


def _similar(a: str, b: str) -> bool:
    """Same word allowing inflection: validation/validated, scalable/scalability."""
    if a == b:
        return True
    shortest = min(len(a), len(b))
    if shortest < 4:
        return False
    needed = max(5, shortest - 3)
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    return prefix >= needed


def inflected_occurrences(variant: str, normalized_text: str) -> int:
    """Count phrase occurrences where each word may be an inflection of the
    keyword's word. Used only for evidence grounding, never for ATS scoring,
    which stays exact like a real scanner."""
    words = variant.split()
    tokens = normalized_text.split()
    if not words or len(tokens) < len(words):
        return 0
    return sum(1 for i in range(len(tokens) - len(words) + 1)
               if all(_similar(w, tokens[i + j]) for j, w in enumerate(words)))


def keyword_in_evidence(keyword: Keyword, normalized_evidence: str) -> bool:
    return any(inflected_occurrences(v, normalized_evidence) for v in variants(keyword))


def present_keyword_ids(profile: JobProfile, text: str) -> set[str]:
    normalized = normalize(text)
    return {k.id for k in profile.keywords if keyword_hits(k, normalized)}


def ground_profile(profile: JobProfile, job_text: str) -> JobProfile:
    """Keep only keywords whose term or alias literally appears in the job text.
    Deduplicate by normalized term and renumber deterministically."""
    normalized_job = normalize(job_text)
    kept, seen = [], set()
    for keyword in profile.keywords:
        term = normalize(keyword.term)
        if not term or term in seen:
            continue
        if not any(count_occurrences(v, normalized_job) for v in variants(keyword)):
            continue
        # Aliases are matching helpers only; an alias equal to another keyword's
        # term would double count, so drop such aliases.
        seen.add(term)
        kept.append(keyword)
    terms = {normalize(k.term) for k in kept}
    credential = re.compile(r"\b(degree|bachelor|master|phd|doctorate|diploma|certification|certified|assessments?)\b", re.I)
    # Degree-field lists: the text right after "degree in" / "degree(s) in".
    # Fields are a run of Capitalized phrases separated by commas / or / and:
    # "degree in Statistics, Economics, Computer Science or related field".
    field_list = r"((?:[A-Z][A-Za-z&/-]*(?:\s+[A-Z][A-Za-z&/-]*)*(?:\s*,\s*|\s+or\s+|\s+and\s+)?){1,20})"
    degree_spans = [normalize(m.group(1)) for m in re.finditer(r"degrees?\s+in\s+" + field_list, job_text)]
    outside = normalized_job
    for span in degree_spans:
        outside = outside.replace(span, ' ')
    softish = re.compile(r"\b(decision[- ]making|communicat|collaborat|stakeholder|leadership|teamwork|team player|interpersonal|"
                         r"problem[- ]solving|presentation skills|mentor|cross[- ]functional|self[- ]starter|attention to detail|"
                         r"time management|adaptab|curios|passion|ownership|influenc)", re.I)
    # "deploy scalable models", "business domains", "inclusive digital experiences":
    # responsibility phrasing, not a skill an ATS keys on. Treated like soft skills.
    vague = re.compile(r"^(deploy|develop|design|build|create|ensure|identify|perform|translate|collaborate|communicate|"
                       r"support|drive|deliver|lead|manage|maintain|work|use|apply|guide|uphold|leverage)\b|"
                       r"\b(solutions?|experiences?|approach|domains?|needs|requirements?|standards|outcomes?|"
                       r"excellence|innovation|readiness|strategy|initiatives?|mindset)$", re.I)
    for index, keyword in enumerate(kept, 1):
        keyword.id = f'K{index}'
        if credential.search(keyword.term):
            keyword.kind = 'credential'
            continue
        if softish.search(keyword.term) or (vague.search(keyword.term) and keyword.kind != 'tool'):
            keyword.kind = 'soft'
            continue
        # A term that appears only inside a degree-field list ("Master's degree in
        # Statistics, Computer Science ...") is a degree field, not a skill.
        in_span = any(count_occurrences(v, span) for span in degree_spans for v in variants(keyword))
        if in_span and not any(count_occurrences(v, outside) for v in variants(keyword)):
            keyword.kind = 'credential'
        keyword.aliases = [a for a in keyword.aliases if normalize(a) and normalize(a) not in STOPWORDS
                           and normalize(a) not in terms - {normalize(keyword.term)}]
    profile.keywords = kept
    if not kept:
        raise ValueError('No job keywords could be grounded in the job description text.')
    for index, rule in enumerate(profile.eligibility, 1):
        rule.id = f'L{index}'
    return profile


def title_score(title: str, normalized_text: str) -> tuple[int, str]:
    full = normalize(title)
    if not full:
        return 0, 'no title'
    if count_occurrences(full, normalized_text):
        return TITLE_POINTS, 'exact title present'
    words = full.split()
    # Seniority prefixes are not what the parser keys on; the role noun is.
    core = ' '.join(w for w in words if w not in ('senior', 'sr', 'junior', 'jr', 'lead', 'principal', 'staff', 'ii', 'iii', 'i'))
    if core and core != full and count_occurrences(core, normalized_text):
        return TITLE_POINTS // 2, 'role noun present without seniority level'
    return 0, 'title absent'


def format_score(text: str) -> tuple[int, list[str]]:
    checks = []
    checks.append(('email', bool(re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', text))))
    checks.append(('phone', bool(re.search(r'(?:\+?\d[\d\s().-]{7,}\d)', text))))
    lowered = text.lower()
    checks.append(('experience section', 'experience' in lowered))
    checks.append(('education section', 'education' in lowered))
    checks.append(('skills section', 'skills' in lowered))
    passed = sum(1 for _, ok in checks if ok)
    return round(FORMAT_POINTS * passed / len(checks), 1), [name for name, ok in checks if not ok]


def score_resume(profile: JobProfile, text: str) -> dict:
    normalized = normalize(text)
    rows = []
    for keyword in profile.keywords:
        if keyword.kind in ('credential', 'soft'):
            # Degrees are judged as eligibility rules; soft skills belong in the
            # summary and bullets and are reported separately, not scored.
            continue
        hits = keyword_hits(keyword, normalized)
        rows.append({'id': keyword.id, 'term': keyword.term, 'kind': keyword.kind,
                     'priority': keyword.priority, 'matched': hits > 0, 'count': hits})
    required = [r for r in rows if r['priority'] == 'required']
    preferred = [r for r in rows if r['priority'] == 'preferred']
    required_points, preferred_points = REQUIRED_POINTS, PREFERRED_POINTS
    if not preferred:
        required_points += preferred_points
        preferred_points = 0
    if not required:
        preferred_points += required_points
        required_points = 0
    fraction = lambda rows_: (sum(r['matched'] for r in rows_) / len(rows_)) if rows_ else 0
    title_points, title_reason = title_score(profile.title, normalized)
    format_points, format_missing = format_score(text)
    components = {
        'required_keywords': {'earned': round(required_points * fraction(required), 1), 'available': required_points,
                              'matched': sum(r['matched'] for r in required), 'total': len(required)},
        'preferred_keywords': {'earned': round(preferred_points * fraction(preferred), 1), 'available': preferred_points,
                               'matched': sum(r['matched'] for r in preferred), 'total': len(preferred)},
        'job_title': {'earned': title_points, 'available': TITLE_POINTS, 'reason': title_reason},
        'format': {'earned': format_points, 'available': FORMAT_POINTS, 'missing': format_missing},
    }
    total = round(sum(c['earned'] for c in components.values()), 1)
    soft = [{'term': k.term, 'matched': keyword_hits(k, normalized) > 0} for k in profile.keywords if k.kind == 'soft']
    return {'score': min(100.0, total), 'components': components, 'keywords': rows,
            'missing': [r for r in rows if not r['matched']], 'soft_skills': soft}
