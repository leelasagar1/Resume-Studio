"""Deterministic resume skeleton.

The candidate's resume is parsed once into a fixed structure: name, headline,
contact lines, section order, and every entry (job, degree, certification)
with its heading, dates and source bullets. The model never regenerates this
skeleton; it only supplies bullets, a summary and skill groups, and Python
reassembles the resume around the verbatim skeleton.
"""
import re
from pydantic import BaseModel, Field
from .models import Claim, Entry, Resume, Section, SkillGroup, SkillItem

MONTH = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4}'
DATE_RANGE = re.compile(rf'({MONTH})\s*(?:-|–|—|to)\s*({MONTH}|present|current|now)', re.I)
SINGLE_DATE = re.compile(rf'(?:—|–|-|\||,|\()\s*({MONTH})\s*\)?\s*$', re.I)
SUMMARY_WORDS = ('summary', 'profile', 'objective', 'about')
SKILL_WORDS = ('skill', 'technolog', 'tool', 'competenc', 'expertise', 'stack')
BULLET_WORDS = 8


class SourceLine(BaseModel):
    eid: str
    text: str


class SkeletonEntry(BaseModel):
    id: str
    heading: SourceLine
    detail: SourceLine | None
    bullets: list[SourceLine]


class SkeletonSection(BaseModel):
    heading: str
    kind: str  # 'summary' | 'skills' | 'entries'
    entries: list[SkeletonEntry] = Field(default_factory=list)
    lines: list[SourceLine] = Field(default_factory=list)  # summary or skills lines


class Skeleton(BaseModel):
    name: SourceLine
    headline: SourceLine | None
    contact: list[SourceLine]
    sections: list[SkeletonSection]


LABEL_LINE = re.compile(r'^[A-Za-z][A-Za-z &/-]{1,40}:\s*\S')
LOCATION_LINE = re.compile(r'^[A-Z][A-Za-z.\' -]+,\s*[A-Z][A-Za-z. ]+$')  # "Princeton, NJ", "Hyderabad, India"
TRAILING_YEAR = re.compile(r'^(.*?\S)\s{2,}((?:' + MONTH + r'))\s*$', re.I)


JOB_SECTION = re.compile(r'^\s*#*\s*(job summary|job description|about the (role|job|position)|key responsibilities|responsibilities|'
                         r'what you.ll do|what you.ll bring|requirements( & qualifications)?|qualifications|minimum qualifications|'
                         r'preferred qualifications|technical stack( & skills)?|benefits( & perks)?|perks|compensation|equal opportunity)\s*:?\s*$', re.I)


def strip_pasted_job(resume_text: str, job_text: str) -> str:
    """Drop a job description pasted at the end of the resume: either lines
    that appear verbatim in the given job text, or a job-posting section
    heading (Key Responsibilities, Requirements, Benefits ...) that shows up
    after the resume's own content."""
    job_lines = {' '.join(l.split()).lower() for l in job_text.splitlines() if len(l.split()) >= 3}
    lines = resume_text.splitlines()
    normalized = [' '.join(l.split()).lower() for l in lines]
    cut = None
    for i in range(len(lines)):
        window = [n for n in normalized[i:i + 3] if n]
        if len(window) == 3 and all(n in job_lines for n in window):
            cut = i
            break
        if i > 5 and JOB_SECTION.match(lines[i]):
            cut = i
            break
    if cut is None:
        return resume_text
    while cut > 0 and (not normalized[cut - 1] or normalized[cut - 1].startswith('#')):
        cut -= 1
    # A dangling keyword-soup line right above the pasted posting goes too.
    if cut > 0 and normalized[cut - 1].count(',') >= 5 and not DATE_RANGE.search(normalized[cut - 1]):
        cut -= 1
    return '\n'.join(lines[:cut]).rstrip()


def is_list_line(text: str) -> bool:
    """'AI safety, Generative AI, LLMs, Python, ...': a comma list, not a bullet."""
    return text.count(',') >= 5 and text.count(',') * 4 >= len(text.split())


def clean_markdown(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r'^\s*#{1,6}\s*', '', line)
        stripped = re.sub(r'^\s*[-*•]\s+', '', stripped)
        lines.append(stripped)
    return '\n'.join(lines)


def is_heading(text: str, index: int) -> bool:
    words = text.split()
    if index < 1 or '@' in text or re.search(r'\d', text) or DATE_RANGE.search(text):
        return False
    if LABEL_LINE.match(text) or LOCATION_LINE.match(text):
        return False
    return len(words) <= 3 or (text.isupper() and len(words) <= 6)


def is_contact(text: str) -> bool:
    return bool('@' in text or re.search(r'\d{3}[\s.-]?\d{3}[\s.-]?\d{4}', text)
                or re.search(r'linkedin|github|portfolio|www\.|https?://', text, re.I))


def clean(text: str) -> str:
    return ' '.join(text.split())


def split_dated(text: str):
    """'Company, Title <tab> Jan 2025 – Present | City' -> heading, detail."""
    plain = text.translate(str.maketrans({'–': '-', '—': '-'}))
    match = DATE_RANGE.search(plain)
    if not match or match.start() <= 8:
        return clean(text), None
    return clean(text[:match.start()].rstrip(' ,|-–—\t')), clean(text[match.start():])


def parse_resume(evidence: dict) -> Skeleton:
    items = list(evidence.items())
    name = SourceLine(eid=items[0][0], text=clean(items[0][1]))
    headline, contact = None, []
    sections: list[SkeletonSection] = []
    index = 1
    # Header block: everything before the first section heading.
    while index < len(items) and not is_heading(items[index][1], index):
        eid, text = items[index]
        next_text = items[index + 1][1] if index + 1 < len(items) else ''
        if is_contact(text):
            contact.append(SourceLine(eid=eid, text=clean(text)))
        elif (headline is None and len(text.split()) <= 6 and not DATE_RANGE.search(text)
              and not DATE_RANGE.search(next_text) and not re.match(r'^[A-Za-z ]+:', text)):
            headline = SourceLine(eid=eid, text=clean(text))
        else:
            break  # summary or experience without a heading; treat as body
        index += 1

    current = None
    pending_heading: SourceLine | None = None
    entry_counter = 0

    def new_section(heading):
        lowered = heading.lower()
        kind = 'summary' if any(w in lowered for w in SUMMARY_WORDS) else \
               'skills' if any(w in lowered for w in SKILL_WORDS) else 'entries'
        section = SkeletonSection(heading=clean(heading).title() if heading.isupper() else clean(heading), kind=kind)
        sections.append(section)
        return section

    for position in range(index, len(items)):
        eid, text = items[position]
        last_entry = current.entries[-1] if current is not None and current.kind == 'entries' and current.entries else None
        under_dated = (last_entry is not None and not last_entry.bullets and last_entry.detail is not None
                       and (DATE_RANGE.search(last_entry.detail.text) or SINGLE_DATE.search(last_entry.detail.text)
                            or TRAILING_YEAR.match(last_entry.heading.text + '  ' + last_entry.detail.text)))
        if (under_dated and not text.isupper() and len(text.split()) <= 5 and not LABEL_LINE.match(text)
                and not DATE_RANGE.search(text) and not re.search(r'\d', text)):
            # "George Mason University" right under "Master in Data Analytics Engineering  2024"
            last_entry.detail = SourceLine(eid=last_entry.detail.eid, text=last_entry.detail.text + ' | ' + clean(text))
            continue
        if is_heading(text, position):
            current = new_section(text.strip(' :'))
            pending_heading = None
            continue
        inline = re.match(r'^([A-Za-z &/]{3,40}):\s*(.+)$', text)
        if inline and any(w in inline.group(1).lower() for w in SKILL_WORDS + SUMMARY_WORDS) and \
                (current is None or current.kind != 'skills' or not any(w in inline.group(1).lower() for w in SUMMARY_WORDS)):
            label = inline.group(1).strip()
            if current is None or current.kind != ('skills' if any(w in label.lower() for w in SKILL_WORDS) else 'summary'):
                current = new_section(label)
            current.lines.append(SourceLine(eid=eid, text=clean(text)))
            pending_heading = None
            continue
        plain = text.translate(str.maketrans({'–': '-', '—': '-'}))
        trailing = TRAILING_YEAR.match(plain) if len(text.split()) <= 10 else None
        dated = bool(DATE_RANGE.search(text) or SINGLE_DATE.search(plain) or trailing)
        next_text = items[position + 1][1] if position + 1 < len(items) else ''
        next_dated = bool(DATE_RANGE.search(next_text))
        if current is None:
            prose = len(text.split()) >= BULLET_WORDS and not dated and not next_dated
            current = new_section('Summary' if prose else 'Experience')
        elif current.kind == 'summary' and (dated or next_dated) and not is_contact(text):
            current = new_section('Experience')  # summary prose ended; jobs follow without a heading
        line = SourceLine(eid=eid, text=clean(text))
        if current.kind in ('summary', 'skills'):
            current.lines.append(line)
            continue
        if dated:
            entry_counter += 1
            heading_text, detail_text = split_dated(text)
            if detail_text is None and trailing:
                heading_text, detail_text = clean(trailing.group(1)), clean(trailing.group(2))
            if detail_text is None and pending_heading is not None and len(text.split()) <= 6:
                # "Data Analyst | Harbor Analytics" on one line, "2022 - Present" on the next.
                entry = SkeletonEntry(id=f'e{entry_counter}', heading=pending_heading,
                                      detail=SourceLine(eid=eid, text=clean(text)), bullets=[])
            elif detail_text is None:
                entry = SkeletonEntry(id=f'e{entry_counter}', heading=line, detail=None, bullets=[])
            else:
                entry = SkeletonEntry(id=f'e{entry_counter}', heading=SourceLine(eid=eid, text=heading_text),
                                      detail=SourceLine(eid=eid, text=detail_text), bullets=[])
            current.entries.append(entry)
            pending_heading = None
            continue
        last = current.entries[-1] if current.entries else None
        just_after_heading = last is not None and not last.bullets and last.heading.eid != eid
        if len(text.split()) >= BULLET_WORDS:
            if last is not None and not is_list_line(text):
                last.bullets.append(line)
            # a long line with no entry above it is prose we cannot place; ignored
        elif just_after_heading and (LOCATION_LINE.match(text) or len(text.split()) <= 5) and not LABEL_LINE.match(text):
            # "New York City, NY" or "George Mason University" right under a dated heading
            if last.detail is None:
                last.detail = line
            else:
                last.detail = SourceLine(eid=last.detail.eid, text=last.detail.text + ' | ' + line.text)
        else:
            pending_heading = line
            # A short undated line (project name, award, certification) becomes
            # its own entry unless a dated line follows and claims it.
            entry_counter += 1
            current.entries.append(SkeletonEntry(id=f'e{entry_counter}', heading=line, detail=None, bullets=[]))
    # A pending short line that was immediately followed by its date line was
    # appended as a placeholder entry; drop that placeholder.
    for section in sections:
        kept = []
        for entry in section.entries:
            if kept and kept[-1].heading.eid == entry.heading.eid and kept[-1].detail is None and entry.detail is not None:
                kept[-1] = entry
            else:
                kept.append(entry)
        section.entries = kept
    if not any(s.kind == 'entries' and s.entries for s in sections):
        raise ValueError('Could not find any dated experience or education entries in the resume text.')
    return Skeleton(name=name, headline=headline, contact=contact, sections=sections)


def skeleton_payload(skeleton: Skeleton) -> dict:
    """What the writer sees: fixed lines it must not change, and the source bullets it rewrites."""
    return {
        'name': skeleton.name.text,
        'headline': skeleton.headline.text if skeleton.headline else '',
        'contact': [c.text for c in skeleton.contact],
        'sections': [{
            'heading': s.heading, 'kind': s.kind,
            'lines': [{'eid': l.eid, 'text': l.text} for l in s.lines],
            'entries': [{'entry_id': e.id, 'heading': e.heading.text, 'detail': e.detail.text if e.detail else '',
                         'heading_eid': e.heading.eid, 'detail_eid': e.detail.eid if e.detail else '',
                         'source_bullets': [{'eid': b.eid, 'text': b.text} for b in e.bullets]} for e in s.entries],
        } for s in skeleton.sections],
    }


def split_items(text: str) -> list[str]:
    """Split a skills line on commas and semicolons outside parentheses."""
    items, depth, current = [], 0, ''
    for ch in text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        if ch in ',;' and depth == 0:
            items.append(current.strip())
            current = ''
        else:
            current += ch
    items.append(current.strip())
    return [i.strip(' .') for i in items if i.strip(' .')]


def source_skill_groups(skeleton: Skeleton) -> list[SkillGroup]:
    """The candidate's own skills lines as groups: 'Label: a, b (c, d), e'."""
    groups = []
    for section in skeleton.sections:
        if section.kind != 'skills':
            continue
        for line in section.lines:
            label, _, rest = line.text.partition(':')
            if not rest.strip():
                label, rest = section.heading, line.text
            if re.match(r'^\s*skills?\s*$', label, re.I):
                label = 'Skills'
            groups.append(SkillGroup(label=label.strip()[:60],
                                     items=[SkillItem(text=i, evidence_ids=[line.eid]) for i in split_items(rest)]))
    return groups


LABEL_SYNONYMS = {'ml': 'ml', 'machine': 'ml', 'ai': 'ai', 'nlp': 'nlp', 'languages': 'language', 'frameworks': 'framework',
                  'tools': 'tool', 'platforms': 'platform', 'methods': 'method', 'libraries': 'library', 'technologies': 'technology',
                  'skills': 'skill', 'domains': 'domain', 'deployment': 'mlops', 'devops': 'mlops', 'engineering': 'engineering'}
LABEL_NOISE = {'and', 'or', 'of', 'the', 'other', 'core', 'key', 'additional', 'technology', 'skill', 'expertise', 'knowledge', 'learning'}


def label_words(label: str) -> set[str]:
    words = re.sub(r'[^a-z0-9]+', ' ', label.lower()).split()
    return {LABEL_SYNONYMS.get(w, w) for w in words} - LABEL_NOISE


def best_group(label: str, groups: list[SkillGroup]):
    """The group whose label shares the most words with `label`, if any."""
    wanted = label_words(label)
    scored = [(len(wanted & label_words(g.label)), len(g.items), g) for g in groups]
    scored = [t for t in scored if t[0] > 0]
    return max(scored, key=lambda t: (t[0], -t[1]))[2] if scored else None


def merge_source_skills(groups: list[SkillGroup], source: list[SkillGroup]) -> list[SkillGroup]:
    """Every skill the candidate listed survives: an item the writer dropped is
    appended to the writer's group with the closest label, else the source
    group is added."""
    def key(text):
        return re.sub(r'[^a-z0-9+#]+', ' ', text.lower()).strip()
    present = {key(i.text) for g in groups for i in g.items}
    # also count items contained inside a compound item, e.g. "Azure (ML, Databricks)"
    compound = ' '.join(key(i.text) for g in groups for i in g.items)
    for src in source:
        missing = [i for i in src.items if key(i.text) not in present and key(i.text) not in compound]
        if not missing:
            continue
        target = best_group(src.label, groups)
        if target is None:
            target = SkillGroup(label=src.label, items=[])
            groups.append(target)
        target.items.extend(missing)
        present.update(key(i.text) for i in missing)
    return groups


def assemble(skeleton: Skeleton, rewrite, evidence: dict) -> Resume:
    """Build the Resume from the verbatim skeleton plus the model's bullets,
    summary, headline and skill groups. Unknown entry ids are ignored; an
    entry the model skipped keeps its source bullets."""
    by_entry = {e.entry_id: e.bullets for e in rewrite.entries}
    skill_groups = merge_source_skills([g.model_copy(deep=True) for g in rewrite.skill_groups], source_skill_groups(skeleton))
    used = set()

    def unique(claim_id):
        base = claim_id or 'c'
        candidate, n = base, 1
        while candidate in used:
            n += 1
            candidate = f'{base}_{n}'
        used.add(candidate)
        return candidate

    name = Claim(id=unique('name'), text=skeleton.name.text, evidence_ids=[skeleton.name.eid], proposed=False)
    contact = [Claim(id=unique(f'c{i + 1}'), text=c.text, evidence_ids=[c.eid], proposed=False)
               for i, c in enumerate(skeleton.contact)]
    summary = [Claim(id=unique(c.id), text=c.text, evidence_ids=c.evidence_ids, proposed=False) for c in rewrite.summary]
    sections = []
    skills_placed = False
    for source in skeleton.sections:
        if source.kind == 'summary':
            continue  # rendered from rewrite.summary under the Summary heading
        if source.kind == 'skills':
            if skill_groups and not skills_placed:
                sections.append(Section(heading=source.heading, entries=[], skill_groups=skill_groups))
                skills_placed = True
            continue
        entries = []
        for entry in source.entries:
            heading = Claim(id=unique(entry.id), text=entry.heading.text, evidence_ids=[entry.heading.eid], proposed=False)
            detail = Claim(id=unique(entry.id + 'd'), text=entry.detail.text, evidence_ids=[entry.detail.eid], proposed=False) \
                if entry.detail else None
            bullets = by_entry.get(entry.id)
            if bullets is None:
                bullets = [Claim(id=unique(f'{entry.id}s{i + 1}'), text=b.text.rstrip('.') + '.', evidence_ids=[b.eid], proposed=False)
                           for i, b in enumerate(entry.bullets)]
            else:
                bullets = [Claim(id=unique(b.id), text=b.text, evidence_ids=b.evidence_ids, proposed=b.proposed) for b in bullets]
            entries.append(Entry(heading=heading, detail=detail, bullets=bullets))
        if entries:
            sections.append(Section(heading=source.heading, entries=entries, skill_groups=[]))
    if skill_groups and not skills_placed:
        # Source had no skills section: place skills before Education if present, else last.
        position = next((i for i, s in enumerate(sections) if 'educat' in s.heading.lower()), len(sections))
        sections.insert(position, Section(heading='Technical Skills', entries=[], skill_groups=skill_groups))
    headline = (rewrite.headline or (skeleton.headline.text if skeleton.headline else '')).strip()
    return Resume(name=name, headline=headline, contact=contact, summary=summary, sections=sections)
