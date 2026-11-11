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


# Words that name no skill on their own. A term made only of these ("ML
# models", "downstream consumption systems", "external advisory professional
# services") is job-ad phrasing, not something a recruiter searches for.
GENERIC_WORDS = {'ml', 'ai', 'data', 'model', 'models', 'pipeline', 'pipelines', 'scaling', 'scalable', 'system',
                 'systems', 'tool', 'tools', 'platform', 'platforms', 'release', 'releases', 'workflow', 'workflows',
                 'framework', 'frameworks', 'solution', 'solutions', 'manipulation', 'consumption', 'downstream',
                 'upstream', 'external', 'internal', 'professional', 'service', 'services', 'advisory', 'frontier',
                 'family', 'families', 'end', 'to', 'and', 'of', 'the', 'for', 'with', 'technical', 'technology',
                 'technologies', 'process', 'processes', 'production', 'performance', 'business', 'operational',
                 'best', 'practices', 'based', 'driven', 'features', 'feature', 'product', 'products', 'results',
                 'insights', 'reliability', 'efficiency', 'impact', 'value', 'teams', 'team', 'other', 'various',
                 'related', 'relevant', 'multiple', 'new', 'existing', 'key', 'core', 'large', 'small', 'high',
                 'library', 'libraries'}
# A trailing container noun hides the searchable term: "DevOps tools" -> DevOps.
TRAILING_CONTAINER = {'tools', 'tool', 'families', 'family', 'frameworks', 'platforms', 'platform', 'services',
                      'technologies', 'stack', 'libraries'}
# A job title is never a section heading of the ad.
NOT_A_TITLE = re.compile(r"\b(requirements?|qualifications?|responsibilities|about|overview|description|benefits|"
                         r"duties|what you|who you|we are|summary|preferred|minimum|basic|skills)\b", re.I)
# "at least 2 of the following: A, B, C", "such as A, B or C", "e.g. A": any one
# item satisfies the requirement, so no single item is required.
OPTION_LIST = re.compile(r"\b(at least (?:\w+|\d+) of|any (?:combination )?of|one or more of|one of the following|"
                         r"some of|such as|e\.g\.?|eg\b|for example|including but not limited to|or similar|"
                         r"or equivalent|or other)", re.I)


def valid_title(title: str) -> str:
    title = ' '.join((title or '').split())
    if not title or len(title.split()) > 8 or NOT_A_TITLE.search(title):
        return ''
    return title


def option_spans(job_text: str) -> list[str]:
    """Normalized text of every "any of / such as / e.g." list in the ad."""
    spans = []
    for line in job_text.splitlines():
        # Sentence ends, but not the dots inside "e.g." / "i.e.".
        for part in re.split(r'(?<=[.;])(?<!e\.g\.)(?<!i\.e\.)\s+', line):
            match = OPTION_LIST.search(part)
            if match:
                spans.append(normalize(part[match.start():]))
    return [span for span in spans if span]


def ground_profile(profile: JobProfile, job_text: str) -> JobProfile:
    """Keep only keywords whose term or alias literally appears in the job text.
    Deduplicate by normalized term and renumber deterministically."""
    normalized_job = normalize(job_text)
    profile.title = valid_title(profile.title)
    for keyword in profile.keywords:
        words = keyword.term.split()
        # Only a named product or practice survives the trim ("DevOps tools" ->
        # DevOps); "assistive technologies" must not become "assistive".
        if (len(words) > 1 and words[-1].lower() in TRAILING_CONTAINER
                and any(c.isupper() for c in ''.join(words[:-1]))):
            words = words[:-1]
            # "Generative AI models and frameworks" -> "Generative AI models", not "... and".
            while len(words) > 1 and words[-1].lower().strip(',') in ('and', 'or', '&', 'and/or', 'with'):
                words = words[:-1]
            keyword.term = ' '.join(words).rstrip(',')
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
                       r"excellence|innovation|readiness|strategy|initiatives?|mindset|expertise|techniques|practices|"
                       r"technologies|capabilities)$|"
                       # "robust models", "rigorous validation", "advanced statistical": a
                       # quality claim wrapped around a skill, not a term a recruiter searches.
                       r"^(robust|rigorous|advanced|strong|solid|proven|excellent|effective|reliable|"
                       r"high[- ]quality|innovative|complex|cutting[- ]edge|modern|sound|best)\b|"
                       # "decision-driven", "data-informed": adjectives, not skills.
                       r"-(driven|informed|focused|oriented|minded)$", re.I)
    # One generic word ("engineering", "product", "reliability") matches almost
    # any resume by accident and names no skill.
    generic = {'engineering', 'product', 'products', 'reliability', 'scalability', 'quality', 'performance',
               'efficiency', 'accuracy', 'impact', 'business', 'technology', 'operations', 'research',
               'analysis', 'execution', 'delivery', 'data', 'systems', 'models', 'insights', 'results',
               'engineer', 'science', 'software', 'tools', 'teams', 'customers', 'growth', 'value',
               'statistical', 'analytical', 'technical', 'quantitative', 'scalable', 'robust', 'advanced'}
    options = option_spans(job_text)
    outside_options = normalized_job
    for span in options:
        outside_options = outside_options.replace(span, ' ')
    for index, keyword in enumerate(kept, 1):
        keyword.id = f'K{index}'
        # Listed only as one alternative among several: preferred, not required.
        if (keyword.priority == 'required'
                and any(count_occurrences(v, span) for span in options for v in variants(keyword))
                and not any(count_occurrences(v, outside_options) for v in variants(keyword))):
            keyword.priority = 'preferred'
        if credential.search(keyword.term):
            keyword.kind = 'credential'
            continue
        if (softish.search(keyword.term) or (vague.search(keyword.term) and keyword.kind != 'tool')
                or normalize(keyword.term) in generic
                or all(w in GENERIC_WORDS for w in normalize(keyword.term).split())):
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
    if not profile.title:
        # No usable title in the ad: its points go to the required keywords
        # rather than capping every resume at 90.
        required_points += TITLE_POINTS if required else 0
        preferred_points += TITLE_POINTS if not required else 0
    fraction = lambda rows_: (sum(r['matched'] for r in rows_) / len(rows_)) if rows_ else 0
    title_points, title_reason = title_score(profile.title, normalized)
    format_points, format_missing = format_score(text)
    components = {
        'required_keywords': {'earned': round(required_points * fraction(required), 1), 'available': required_points,
                              'matched': sum(r['matched'] for r in required), 'total': len(required)},
        'preferred_keywords': {'earned': round(preferred_points * fraction(preferred), 1), 'available': preferred_points,
                               'matched': sum(r['matched'] for r in preferred), 'total': len(preferred)},
        'job_title': {'earned': title_points, 'available': TITLE_POINTS if profile.title else 0, 'reason': title_reason},
        'format': {'earned': format_points, 'available': FORMAT_POINTS, 'missing': format_missing},
    }
    total = round(sum(c['earned'] for c in components.values()), 1)
    soft = [{'term': k.term, 'matched': keyword_hits(k, normalized) > 0} for k in profile.keywords if k.kind == 'soft']
    return {'score': min(100.0, total), 'components': components, 'keywords': rows,
            'missing': [r for r in rows if not r['matched']], 'soft_skills': soft}
