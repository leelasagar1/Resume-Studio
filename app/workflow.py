import re
from difflib import SequenceMatcher
from .agents import BudgetExceeded
from .documents import render_docx
from .models import Audit, Claim, ClaimCheck, Resume, claims, proposed_claims, skill_items, resume_plain_text
from .skeleton import assemble, best_group, clean_markdown, parse_resume, skeleton_payload, strip_pasted_job
from .scoring import (count_occurrences, ground_profile, inflected_occurrences, keyword_hits, keyword_in_evidence,
                      normalize, score_resume)

NUMBER = r'\d+(?:[.,]\d+)*%?'


def make_evidence(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {f'E{i + 1}': line for i, line in enumerate(lines)}


def keyword_status(ats, profile, evidence):
    """What the writer sees: matched terms, missing terms, and for each missing
    term whether the candidate's evidence contains it (may be written into
    bullets) or not (Skills-only, unevidenced)."""
    normalized_evidence = normalize(' '.join(evidence.values()))
    by_id = {k.id: k for k in profile.keywords}
    missing = []
    for r in ats['missing']:
        in_evidence = keyword_in_evidence(by_id[r['id']], normalized_evidence)
        missing.append({'id': r['id'], 'term': r['term'], 'priority': r['priority'], 'kind': r['kind'],
                        'in_candidate_evidence': in_evidence,
                        'placement': 'anywhere the evidence supports it' if in_evidence
                        else 'ONLY as a Skills item with empty evidence_ids'})
    return {'matched': [r['term'] for r in ats['keywords'] if r['matched']], 'missing': missing}


def normalize_skills(resume, evidence, profile):
    """Deterministically decide which skill items are evidenced.

    A skill item counts as evidenced when every part of its text ("Spark
    (PySpark)" has the parts "Spark" and "PySpark") literally occurs in the
    candidate's evidence, directly or through a job-keyword alias. Evidence ids
    the writer supplied are replaced by the lines that actually contain the
    parts, so a sloppy citation cannot smuggle an unevidenced keyword through,
    and a forgotten citation does not force the candidate to re-confirm a real
    skill.
    """
    normalized_lines = {eid: normalize(line) for eid, line in evidence.items()}
    by_term = {normalize(k.term): k for k in profile.keywords}
    for keyword in profile.keywords:
        for alias in keyword.aliases:
            by_term.setdefault(normalize(alias), keyword)

    def lines_with(part):
        keyword = by_term.get(part)
        variants = [part] + ([normalize(a) for a in keyword.aliases] + [normalize(keyword.term)] if keyword else [])
        return [eid for eid, line in normalized_lines.items()
                if any(inflected_occurrences(v, line) for v in variants if v)]

    def source_spelling(part):
        """The candidate's own casing for a single-term item (tensorflow -> TensorFlow,
        torch -> PyTorch). Prefers the spelling with capitals (Skills line) over prose."""
        keyword = by_term.get(part)
        found = []
        for variant in [part] + ([normalize(a) for a in keyword.aliases] + [normalize(keyword.term)] if keyword else []):
            pattern = r'(?<![A-Za-z0-9+#])' + r'[\s\-]*'.join(re.escape(w) for w in variant.split()) + r'(?![A-Za-z0-9+#])'
            for line in evidence.values():
                found.extend(m.group() for m in re.finditer(pattern, line, re.I))
            if found:
                break
        return max(found, key=lambda t: sum(ch.isupper() for ch in t)) if found else None

    for item in skill_items(resume):
        item.text = ' '.join(item.text.split())
        parts = [normalize(p) for p in re.split(r'[()/,;&]| and ', item.text)]
        parts = [p for p in parts if p]
        hits = []
        for part in parts:
            found = lines_with(part)
            if not found:
                hits = []
                break
            hits.extend(found)
        item.evidence_ids = list(dict.fromkeys(hits))[:3]
        if item.evidence_ids and len(parts) == 1:
            spelled = source_spelling(parts[0])
            if spelled:
                item.text = spelled
    curate_skills(resume, evidence, profile)
    # Skill groups nested inside an entries section (Work Experience) get their
    # own Skills section right after it, so the heading renders.
    rebuilt = []
    for section in resume.sections:
        if section.entries and section.skill_groups:
            groups, section.skill_groups = section.skill_groups, []
            rebuilt.append(section)
            home = next((s for s in resume.sections if s.skill_groups and not s.entries), None)
            if home is not None:
                home.skill_groups.extend(groups)
            else:
                from .models import Section
                rebuilt.append(Section(heading='Technical Skills', entries=[], skill_groups=groups))
        else:
            rebuilt.append(section)
    resume.sections = [s for s in rebuilt if s.entries or s.skill_groups]
    return tidy_layout(resume)


def tidy_layout(resume):
    """Presentation fixes that need no model: dates on the detail line, Title
    Case section headings, skill labels that do not leak workflow words."""
    for section in resume.sections:
        if section.heading.isupper():
            section.heading = section.heading.title()
        for group in section.skill_groups:
            if re.search(r'\b(required|preferred|missing|job|confirm|unverified|credential)', group.label, re.I):
                group.label = 'Methods & Domains' if any(len(i.text.split()) > 1 for i in group.items) else 'Additional Skills'
        merged = {}
        for group in section.skill_groups:
            key = normalize(group.label)
            if key in merged:
                seen = {normalize(i.text) for i in merged[key].items}
                merged[key].items.extend(i for i in group.items if normalize(i.text) not in seen)
            else:
                merged[key] = group
        section.skill_groups = list(merged.values())
        for entry in section.entries:
            if entry.detail is None:
                match = DATE_RANGE.search(entry.heading.text.translate(str.maketrans({'–': '-', '—': '-'})))
                if match and match.start() > 8:
                    raw = entry.heading.text
                    cut = match.start()
                    head, tail = raw[:cut].rstrip(' ,|-–—\t'), raw[cut:].strip()
                    entry.detail = Claim(id=entry.heading.id + 'd', text=tail, evidence_ids=list(entry.heading.evidence_ids), proposed=False)
                    entry.heading.text = head
    return resume


def curate_skills(resume, evidence, profile):
    """HR-reviewer view of the Skills section: every item is either a job
    keyword or a skill the experience bullets demonstrate. Soft skills,
    credentials, writer-invented phrases and orphan tools are removed; near
    duplicates collapse; unverified job terms are capped, required first."""
    credential_terms = {normalize(k.term) for k in profile.keywords if k.kind == 'credential'}
    soft_terms = {normalize(k.term) for k in profile.keywords if k.kind == 'soft'}
    scorable = [k for k in profile.keywords if k.kind not in ('credential', 'soft')]
    variant_to_keyword = {}
    for k in scorable:
        for v in [normalize(k.term)] + [normalize(a) for a in k.aliases]:
            variant_to_keyword.setdefault(v, k)
    bullet_text = normalize(experience_text(resume))

    def parts_of(text):
        return [normalize(p) for p in re.split(r'[()/,;&]| and ', text) if normalize(p)]

    def job_keyword_for(text):
        for part in parts_of(text) + [normalize(text)]:
            if part in variant_to_keyword:
                return variant_to_keyword[part]
        return None

    def demonstrated(text):
        return any(inflected_occurrences(part, bullet_text) for part in parts_of(text))

    evidenced_keys = {normalize(i.text) for i in skill_items(resume) if i.evidence_ids}
    seen = set()
    unverified = []  # (priority rank, group, item)
    for section in resume.sections:
        for group in section.skill_groups:
            kept = []
            for item in group.items:
                key = normalize(item.text)
                if not key or key in seen or CREDENTIAL.search(item.text) or key in credential_terms or key in soft_terms:
                    continue
                keyword = job_keyword_for(item.text)
                if item.evidence_ids:
                    if keyword is None and not demonstrated(item.text):
                        continue  # orphan: not asked for, not shown in the bullets
                    seen.add(key)
                    kept.append(item)
                else:
                    if keyword is None or keyword.kind == 'soft' or len(item.text.split()) > 6:
                        continue  # only exact job terms may be added unverified
                    shadowed = any(key != e and (key in e.split() or key in e.replace(' ', '') or (len(key) > 4 and key in e))
                                   for e in evidenced_keys)
                    if shadowed:
                        continue
                    seen.add(key)
                    unverified.append((0 if keyword.priority == 'required' else 1, group, item))
            group.items = kept
    # Near-duplicate unverified terms: keep the shorter one ("statistical methods"
    # over "advanced statistical methods"); cap the total, required first.
    unverified.sort(key=lambda t: (t[0], len(t[2].text)))
    chosen = []
    for rank, group, item in unverified:
        key = normalize(item.text)
        if any(key in normalize(c[2].text) or normalize(c[2].text) in key for c in chosen):
            continue
        chosen.append((rank, group, item))
    for rank, group, item in chosen[:MAX_UNVERIFIED_SKILLS]:
        group.items.append(item)
    # "AWS SageMaker" next to "AWS (SageMaker, S3, Glue)": the item whose parts are a
    # subset of another item's parts is redundant.
    all_items = [i for s in resume.sections for g in s.skill_groups for i in g.items]
    parts = {id(i): set(parts_of(i.text)) for i in all_items}
    redundant = {id(a) for a in all_items for b in all_items
                 if a is not b and parts[id(a)] and parts[id(a)] < parts[id(b)]}
    for section in resume.sections:
        for group in section.skill_groups:
            group.items = [i for i in group.items if id(i) not in redundant]
    for section in resume.sections:
        for group in section.skill_groups:
            if len(group.items) > MAX_SKILLS_PER_GROUP:
                # Keep job-keyword items first, then demonstrated ones.
                group.items.sort(key=lambda i: (job_keyword_for(i.text) is None, not i.evidence_ids))
                group.items = group.items[:MAX_SKILLS_PER_GROUP]
        groups = [g for g in section.skill_groups if g.items]
        # Fold tiny groups, then the smallest groups beyond the cap, into the
        # group with the closest label (else the largest group).
        def fold(group, others):
            target = best_group(group.label, others) or max(others, key=lambda g: len(g.items))
            target.items.extend(group.items)
        while groups:
            tiny = next((g for g in groups if len(g.items) <= 1 and len(groups) > 1), None)
            if tiny is None:
                break
            groups.remove(tiny)
            fold(tiny, groups)
        while len(groups) > MAX_SKILL_GROUPS:
            smallest = min(groups, key=lambda g: len(g.items))
            groups.remove(smallest)
            fold(smallest, groups)
        for group in groups:
            if len(group.items) > MAX_SKILLS_PER_GROUP:
                group.items.sort(key=lambda i: (job_keyword_for(i.text) is None, not i.evidence_ids))
                group.items = group.items[:MAX_SKILLS_PER_GROUP]
        section.skill_groups = groups
    return resume


def unverified_terms(resume):
    return list(dict.fromkeys(i.text for i in skill_items(resume) if not i.evidence_ids))


MONTH = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4}'
DATE_RANGE = re.compile(rf'({MONTH})\s*(?:-|–|—|to)\s*({MONTH}|present|current|now)', re.I)


def date_ranges(text):
    """Normalized 'start end' pairs for every date range in the text. Employers,
    degrees and certifications carry them, so they are a cheap integrity key."""
    def month_key(token):
        # "Dec 2019", "December 2019" and "12/2019" all mean the same entry.
        numeric = re.fullmatch(r'\s*(\d{1,2})/(\d{4})\s*', token)
        if numeric:
            month, year = int(numeric.group(1)), numeric.group(2)
            names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            return f'{names[month - 1]} {year}' if 1 <= month <= 12 else normalize(token)
        token = normalize(token)
        word = re.match(r'[a-z]+', token)
        if word:
            return word.group()[:3] + ' ' + token[len(word.group()):].strip()
        return token

    found = []
    for start, end in DATE_RANGE.findall(text.translate(str.maketrans({'–': '-', '—': '-'}))):
        end_key = 'present' if normalize(end) in ('present', 'current', 'now') else month_key(end)
        key = month_key(start) + ' ' + end_key
        if key not in found:
            found.append(key)
    return found


MAX_SKILLS_PER_GROUP = 8
MAX_SKILL_GROUPS = 5
MAX_PROPOSED_PER_ROLE = 3
MAX_PROPOSED_TOTAL = 10
MAX_UNVERIFIED_SKILLS = 10
MAX_BULLETS = 7          # total per role, proposed bullets included
MIN_BULLETS = 5          # evidence-backed bullets per role (or the source count if fewer)
CREDENTIAL = re.compile(r"\b(degree|bachelor|master|phd|doctorate|diploma|certification|certified|assessments?|years?\b.{0,12}experience|"
                        r"experience\b.{0,12}years?|related field|equivalent experience)\b", re.I)
WEAK_STARTS = {'i', 'we', 'my', 'responsible', 'responsibilities', 'worked', 'helped', 'assisted', 'was', 'were',
               'duties', 'tasked', 'involved', 'participated', 'various', 'also', 'the', 'a', 'an', 'utilized', 'utilised'}
PAST_TENSE = {'built', 'led', 'drove', 'ran', 'wrote', 'set', 'cut', 'won', 'grew', 'made', 'took', 'brought', 'held', 'kept',
              'sold', 'spent', 'taught', 'thought', 'oversaw', 'began', 'undertook', 'rebuilt', 'shipped', 'spearheaded',
              'co-led', 'coled', 'overhauled', 'reran', 'rewrote', 'sped', 'laid', 'put', 'read', 'split', 'shut', 'let', 'lit',
              'broke', 'chose', 'drew', 'found', 'got', 'gave', 'grew', 'hired', 'saw', 'sent', 'stood', 'struck', 'swept',
              'thrust', 'understood', 'upheld', 'wove', 'withdrew', 'forecast', 'cast', 'bid', 'bet', 'input', 'output'}
BANNED_PHRASES = ['results-driven', 'results driven', 'responsible for', 'detail-oriented', 'team player',
                  'hard-working', 'go-getter', 'synergy', 'think outside the box', 'proven track record']
# Sentence-initial verbs and connectives that are not technologies, so they are
# never listed as a role's demonstrated stack.
BULLET_VERBS = PAST_TENSE | WEAK_STARTS | {'designed', 'developed', 'implemented', 'created', 'automated', 'migrated',
                                           'engineered', 'orchestrated', 'integrated', 'delivered', 'improved',
                                           'partnered', 'collaborated', 'presented', 'tracked', 'analyzed', 'analysed',
                                           'established', 'maintained', 'trained', 'validated', 'deployed', 'reduced'}

SINGLE_DATE = re.compile(rf'(?:—|–|-|\||,|\()\s*({MONTH})\s*\)?\s*$', re.I)


def entry_keys(text):
    """Date ranges plus trailing single dates on short lines (certifications,
    awards: "Databricks Certified ... — June 2025")."""
    keys = list(date_ranges(text))
    for line in text.splitlines():
        line = line.strip()
        if len(line) <= 110 and not date_ranges(line):
            match = SINGLE_DATE.search(line.translate(str.maketrans({'–': '-', '—': '-'})))
            if match and re.search(r'[a-z]', match.group(1), re.I):
                key = 'single ' + date_ranges(match.group(1) + ' - present')[0].split(' present')[0]
                if key not in keys:
                    keys.append(key)
    return keys


def source_bullet_counts(evidence):
    """Number of source lines under each dated entry, keyed by its entry key.
    Lines are attributed to the most recent dated line above them; short
    all-caps section headings are not counted."""
    counts, current = {}, None
    for line in evidence.values():
        keys = entry_keys(line)
        if keys:
            current = keys[0]
            counts.setdefault(current, 0)
            continue
        if len(line.split()) <= 3 or line.isupper():
            current = None  # section heading such as EDUCATION or Skills
            continue
        if current and len(line.split()) >= 8:
            counts[current] += 1
    return counts


def source_lines_by_entry(evidence):
    """Source bullet lines grouped under each dated entry key, as (eid, text)."""
    groups, current = {}, None
    for eid, line in evidence.items():
        keys = entry_keys(line)
        if keys:
            current = keys[0]
            groups.setdefault(current, [])
            continue
        if len(line.split()) <= 3 or line.isupper():
            current = None
            continue
        if current and len(line.split()) >= 8:
            groups[current].append((eid, line))
    return groups


def backfill_bullets(resume, evidence, profile, floor=MIN_BULLETS):
    """Bring each role up to the floor by appending its most job-relevant
    original source lines verbatim (evidence-backed, skip the audit)."""
    groups = source_lines_by_entry(evidence)
    existing_ids = {c.id for c in claims(resume)}
    added = 0
    for section in resume.sections:
        for entry in section.entries:
            keys = entry_keys(entry.heading.text + ' ' + (entry.detail.text if entry.detail else ''))
            source = next((groups[k] for k in keys if k in groups), [])
            target = min(floor, len(source))
            evidenced = [b for b in entry.bullets if not b.proposed]
            if len(evidenced) >= target:
                continue
            present = {normalize(b.text) for b in entry.bullets}
            used = {e for b in entry.bullets if not b.proposed for e in b.evidence_ids}
            candidates = [(eid, line) for eid, line in source
                          if eid not in used and normalize(line) not in present]
            ranked = sorted(candidates, key=lambda item: -bullet_relevance(
                Claim(id='x', text=item[1], evidence_ids=[item[0]], proposed=False), profile))
            for eid, line in ranked[:target - len(evidenced)]:
                number = 1
                while f'{entry.heading.id}s{number}' in existing_ids:
                    number += 1
                claim_id = f'{entry.heading.id}s{number}'
                existing_ids.add(claim_id)
                entry.bullets.append(Claim(id=claim_id, text=line.rstrip('.') + '.', evidence_ids=[eid], proposed=False))
                added += 1
    return resume, added


def bullet_relevance(bullet, profile):
    text = normalize(bullet.text)
    score = 0
    for keyword in profile.keywords:
        if keyword_hits(keyword, text):
            score += 2 if keyword.priority == 'required' else 1
    if re.search(r'\d', bullet.text):
        score += 1
    return score


def trim_bullets(resume, profile, max_per_entry=MAX_BULLETS):
    """Keep the most job-relevant bullets per role. Proposed bullets stay; a
    bullet that is the only holder of a matched keyword stays."""
    removed = 0
    for section in resume.sections:
        for entry in section.entries:
            while len(entry.bullets) > max_per_entry:
                candidates = [b for b in entry.bullets if not b.proposed]
                if len(candidates) <= MIN_BULLETS:
                    break
                others = normalize(experience_text_without(resume, None))
                ranked = sorted(candidates, key=lambda b: (bullet_relevance(b, profile), -entry.bullets.index(b)))
                victim = None
                for bullet in ranked:
                    rest = normalize(experience_text_without(resume, bullet.id))
                    unique = [k for k in profile.keywords if keyword_hits(k, normalize(bullet.text)) and not keyword_hits(k, rest)]
                    if not unique:
                        victim = bullet
                        break
                if victim is None:
                    break
                entry.bullets.remove(victim)
                removed += 1
    return resume, removed


def experience_text_without(resume, bullet_id):
    lines = [c.text for c in resume.summary]
    for section in resume.sections:
        for entry in section.entries:
            lines.append(entry.heading.text)
            lines.extend(b.text for b in entry.bullets if b.id != bullet_id)
    return '\n'.join(lines)


def bullet_standard_issues(bullet):
    text = bullet.text.strip()
    words = text.split()
    issues = []
    # A deliberately short bullet is good writing; only a fragment is not.
    if len(words) < 6:
        issues.append(f'{bullet.id}: too short ({len(words)} words); name the work and its result.')
    elif len(words) > 38:
        issues.append(f'{bullet.id}: too long ({len(words)} words); keep one idea per bullet.')
    first = re.sub(r'[^a-z]', '', words[0].lower()) if words else ''
    if first in WEAK_STARTS:
        issues.append(f'{bullet.id}: start with a direct past-tense action verb, not "{words[0]}".')
    elif words and not words[0][0].isupper():
        issues.append(f'{bullet.id}: start with a capitalized action verb.')
    elif first and not (first.endswith('ed') or first in PAST_TENSE):
        issues.append(f'{bullet.id}: "{words[0]}" is not past tense; start with a past-tense action verb (Built, Designed, Led).')
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            issues.append(f'{bullet.id}: remove the cliche "{phrase}".')
            break
    return issues


SUMMARY_BANNED = ['leveraged', 'seasoned', 'passionate', 'results-driven', 'results driven', 'proven track record',
                  'dynamic', 'detail-oriented', 'responsible for', 'hard-working', 'go-getter', 'self-starter', 'team player']


def summary_issues(resume, evidence, profile):
    """A recruiter-grade summary: 3 prose sentences, identity + proof + fit."""
    if not resume.summary:
        return []
    first = resume.summary[0].id
    text = ' '.join(c.text for c in resume.summary)
    words = len(text.split())
    sentences = sum(max(1, len(re.findall(r'[.!?](?:\s|$)', c.text))) for c in resume.summary)
    issues = []
    if words < 40:
        issues.append(f'{first}: summary too short ({words} words); write 3 sentences, 50-75 words: identity, proof with metrics, fit.')
    elif words > 95:
        issues.append(f'{first}: summary too long ({words} words); keep 3 sentences under 75 words.')
    if sentences < 2:
        issues.append(f'{first}: summary is a single sentence; write 3 sentences: identity, proof with metrics, fit.')
    lowered = text.lower()
    for phrase in SUMMARY_BANNED:
        if phrase in lowered:
            issues.append(f'{first}: remove "{phrase}" from the summary; state facts instead.')
            break
    if re.search(r"\b(i|my|me)\b", lowered):
        issues.append(f'{first}: no first person in the summary.')
    opener = resume.summary[0].text.split()
    head = re.sub(r'[^a-z]', '', opener[0].lower()) if opener else ''
    if head and (head.endswith('ed') or head in PAST_TENSE) and head not in ('experienced', 'skilled', 'versed', 'focused', 'specialized', 'certified', 'accomplished'):
        issues.append(f'{first}: the summary opens with an identity sentence ("<Title> with N+ years ..."), not a bullet starting with "{opener[0]}".')
    core = normalize(profile.title)
    role_noun = ' '.join(w for w in core.split() if w not in ('senior', 'sr', 'junior', 'jr', 'lead', 'principal', 'staff', 'ii', 'iii', 'i'))
    first_sentence = normalize(resume.summary[0].text)
    if role_noun and role_noun not in first_sentence and normalize(resume.headline) not in first_sentence:
        issues.append(f'{first}: the first sentence must name the role ("{profile.title}" or the closest honest variant).')
    if re.search(r'\d+\s*\+?\s*years', ' '.join(evidence.values()), re.I) and not re.search(r'\d+\s*\+?\s*years', text, re.I):
        issues.append(f'{first}: state the years of experience the source gives.')
    scorable = [k for k in profile.keywords if k.kind not in ('credential', 'soft')]
    hits = sum(1 for k in scorable if keyword_hits(k, normalize(text)))
    if len(scorable) >= 3 and hits < 3:
        issues.append(f'{first}: the summary names only {hits} job keyword(s); include at least 3 that the evidence supports.')
    return issues


def validate_draft(resume, evidence, profile):
    """Deterministic checks that never need a model call."""
    items = claims(resume)
    ids = [c.id for c in items]
    issues = []
    # Every dated entry in the source (job, degree, certification) must survive.
    present = set(entry_keys(resume_plain_text(resume)))
    for eid, line in evidence.items():
        for key in entry_keys(line):
            if key not in present:
                issues.append(f'{eid}: the entry dated "{line[:70]}" is missing; keep every employer, degree and certification.')
                break
    for section in resume.sections:
        if len(section.heading.strip()) < 3 or not re.search(r'[A-Za-z]{3}', section.heading):
            issues.append(f'section "{section.heading}": use a real section heading such as Work Experience, Skills or Education.')
    if len(ids) != len(set(ids)):
        issues.append('Every resume field must have a unique claim id.')
    if not resume.sections or not any(s.entries for s in resume.sections):
        issues.append('The resume must retain its experience and education entries.')
    if not any(s.skill_groups for s in resume.sections):
        issues.append('Add a Skills section with skill_groups.')
    if not resume.headline.strip() or len(resume.headline) > 80:
        issues.append('headline: provide a short target-title headline.')
    issues.extend(summary_issues(resume, evidence, profile))
    all_evidence = ' '.join(evidence.values())
    normalized_evidence = normalize(all_evidence)
    evidence_numbers = set(re.findall(NUMBER, all_evidence))
    if normalize(resume.name.text) not in normalized_evidence:
        issues.append(f'{resume.name.id}: the name must be copied from the source.')
    source_counts = source_bullet_counts(evidence)
    for section in resume.sections:
        for entry in section.entries:
            for bullet in entry.bullets:
                if re.match(r'^\s*[-–—•*]', bullet.text):
                    issues.append(f'{bullet.id}: remove the leading bullet marker; Word supplies bullets.')
            keys = entry_keys(entry.heading.text + ' ' + (entry.detail.text if entry.detail else ''))
            available = max((source_counts.get(k, 0) for k in keys), default=0)
            required = min(MIN_BULLETS, available)
            evidenced = [b for b in entry.bullets if not b.proposed]
            if required and len(evidenced) < required:
                issues.append(f'{entry.heading.id}: keep at least {required} evidence-backed bullets for this role '
                              f'(the source has {available}); add back the most job-relevant ones.')
    bullet_ids = {b.id for s in resume.sections for e in s.entries for b in e.bullets}
    for c in items:
        if not c.text.strip() or not c.evidence_ids or not set(c.evidence_ids) <= evidence.keys():
            issues.append(f'{c.id}: missing text or invalid evidence ids.')
            continue
        if c.proposed:
            if c.id not in bullet_ids:
                issues.append(f'{c.id}: only experience bullets may be proposed; set proposed=false or move it.')
            if re.search(r'\d', c.text):
                issues.append(f'{c.id}: a proposed bullet must not contain numbers.')
            continue
        numbers = set(re.findall(NUMBER, c.text))
        if not numbers <= evidence_numbers:
            issues.append(f'{c.id}: number {sorted(numbers - evidence_numbers)[0]} does not appear anywhere in the evidence; remove it.')
        # A job keyword may only be written into a bullet, summary or heading
        # when the candidate's own evidence contains that keyword somewhere.
        text = normalize(c.text)
        for keyword in profile.keywords:
            if keyword.kind in ('soft',):
                continue
            if keyword_hits(keyword, text) and not keyword_in_evidence(keyword, normalized_evidence):
                issues.append(f'{c.id}: "{keyword.term}" is not in the candidate evidence; '
                              'remove it here (it may only be an unevidenced Skills item).')
    return issues


def apply_repairs(resume, repaired, allowed_ids):
    """Replace claim text/evidence in place for the ids that were sent for repair."""
    replacements = {c.id: c for c in repaired if c.id in allowed_ids}
    for c in claims(resume):
        if c.id in replacements:
            c.text = replacements[c.id].text
            c.evidence_ids = replacements[c.id].evidence_ids
    return resume


async def repair_locally(provider, evidence, profile, draft, local):
    """Fix deterministic failures with one small targeted call instead of a
    full rewrite. Returns (draft, remaining_issues)."""
    known = {c.id for c in claims(draft)}
    headings = {e.heading.id for s in draft.sections for e in s.entries}
    repairable = [issue for issue in local
                  if ':' in issue and issue.split(':', 1)[0] in known - headings]
    ids = {issue.split(':', 1)[0] for issue in repairable}
    if not ids or not hasattr(provider, 'repair'):
        return draft, local
    targets = [c for c in claims(draft) if c.id in ids]
    repaired = await provider.repair(evidence, profile, targets, repairable)
    draft = apply_repairs(draft, repaired.claims, ids)
    return draft, validate_draft(draft, evidence, profile)


def claims_to_audit(resume, evidence):
    """Skip claims that are verbatim source lines; the model audits the rest."""
    lines = {normalize(v) for v in evidence.values()}
    selected = []
    for c in claims(resume):
        if c.proposed:
            continue
        text = normalize(c.text)
        if text in lines or any(text and text in line for line in lines):
            continue
        selected.append(c)
    return selected


def strip_unsupported(resume, unsupported_ids):
    """Remove unsupported bullets, summary lines and details. Returns None when
    an identity, contact or heading claim is unsupported (cannot be fixed by removal)."""
    stripped = resume.model_copy(deep=True)
    fixed = {resume.name.id, *(c.id for c in resume.contact)}
    stripped.summary = [c for c in stripped.summary if c.id not in unsupported_ids]
    for section in stripped.sections:
        for entry in section.entries:
            fixed.add(entry.heading.id)
            if entry.detail and entry.detail.id in unsupported_ids:
                entry.detail = None
            entry.bullets = [b for b in entry.bullets if b.id not in unsupported_ids]
    if fixed & set(unsupported_ids):
        return None
    return stripped


def proposals(resume, evidence, profile, fit_reasons=None, review_notes=None):
    """Proposed bullets with their role context, the job skills they add, why
    the writer thinks they fit that company, and the reviewer's verdict."""
    normalized_evidence = normalize(' '.join(evidence.values()))
    fit_reasons = fit_reasons or {}
    review_notes = review_notes or {}
    result = []
    for section in resume.sections:
        for entry in section.entries:
            for bullet in entry.bullets:
                if not bullet.proposed:
                    continue
                text = normalize(bullet.text)
                terms = [k.term for k in profile.keywords
                         if keyword_hits(k, text) and not keyword_in_evidence(k, normalized_evidence)]
                result.append({'id': bullet.id, 'text': bullet.text, 'terms': terms,
                               'fit_reason': fit_reasons.get(bullet.id, ''),
                               'review_note': review_notes.get(bullet.id, ''),
                               'role_context': entry.heading.text + (' | ' + entry.detail.text if entry.detail else '')})
    return result


def experience_text(resume):
    """Summary plus every entry heading, detail and bullet; no Skills section."""
    lines = [c.text for c in resume.summary]
    for section in resume.sections:
        for entry in section.entries:
            lines.append(entry.heading.text)
            if entry.detail:
                lines.append(entry.detail.text)
            lines.extend(b.text for b in entry.bullets)
    return '\n'.join(lines)


# First year each tool was realistically in production use. A proposed bullet
# naming a tool that did not exist during the role's period is rejected.
TOOL_ERA = {
    'hadoop': 2008, 'kafka': 2011, 'docker': 2013, 'spark': 2014, 'pyspark': 2014, 'terraform': 2014,
    'snowflake': 2014, 'kubernetes': 2015, 'airflow': 2015, 'tensorflow': 2015, 'azure data factory': 2015,
    'adf': 2015, 'databricks': 2015, 'pytorch': 2016, 'sagemaker': 2017, 'dbt': 2018, 'mlflow': 2018,
    'fastapi': 2018, 'bert': 2018, 'delta lake': 2019, 'gpt 2': 2019, 'transformers': 2019,
    'gpt 3': 2020, 'dagster': 2020, 'vertex ai': 2021, 'copilot': 2021, 'langchain': 2022,
    'chatgpt': 2022, 'unity catalog': 2022, 'gpt 4': 2023, 'llama': 2023, 'fabric': 2023,
    'agentic ai': 2023, 'llm as judge': 2023, 'rag': 2023, 'retrieval augmented generation': 2023,
}


def role_period(entry):
    """(start, end) years for a role; end is the current year for 'Present'."""
    from datetime import date
    text = entry.heading.text + ' ' + (entry.detail.text if entry.detail else '')
    years = [int(y) for y in re.findall(r'\b((?:19|20)\d{2})\b', text)]
    if not years:
        return None, None
    end = date.today().year if re.search(r'present|current|now', text, re.I) else max(years)
    return min(years), end


def anachronistic(text, entry):
    """Tools named in the text that did not exist during the role's period."""
    _, end = role_period(entry)
    if end is None:
        return []
    normalized = normalize(text)
    return [tool for tool, year in TOOL_ERA.items()
            if count_occurrences(tool, normalized) and year > end]


def role_profiles(resume):
    """What the propose call needs to place a skill in the right company:
    title, period and the technologies that role already demonstrates."""
    profiles = []
    for section in resume.sections:
        for entry in section.entries:
            own = [b for b in entry.bullets if not b.proposed]
            if not own:
                continue
            start, end = role_period(entry)
            tech = []
            for bullet in own:
                for word in re.findall(r'[A-Za-z][A-Za-z0-9+#./-]+', bullet.text):
                    cleaned = word.strip('.,;:()')
                    if (len(cleaned) > 2 and any(c.isupper() for c in cleaned)
                            and cleaned not in tech and cleaned.lower() not in BULLET_VERBS):
                        tech.append(cleaned)
            profiles.append({'entry_id': entry.heading.id, 'company_and_title': entry.heading.text,
                             'period': entry.detail.text if entry.detail else '',
                             'years': f'{start}-{end}' if start else '',
                             'technologies_already_shown': tech[:18],
                             'existing_bullets': [b.text for b in own]})
    return profiles


def skill_gaps(resume, profile):
    """Every job skill, required or preferred, missing from the experience
    bullets. Required first, so the per-role budget is spent on those."""
    text = normalize(experience_text(resume))

    def wanted(keyword):
        return (keyword.kind in ('skill', 'tool', 'method', 'domain') and not CREDENTIAL.search(keyword.term)
                and len(keyword.term.split()) <= 6 and not keyword_hits(keyword, text))

    return ([k.term for k in profile.keywords if k.priority == 'required' and wanted(k)]
            + [k.term for k in profile.keywords if k.priority == 'preferred' and wanted(k)])


def bullet_gaps(resume, profile):
    """Required skill/tool/method/domain keywords that appear nowhere in the
    experience text. These must be reflected in bullets, not only in Skills."""
    text = normalize(experience_text(resume))
    return [k.term for k in profile.keywords
            if k.priority == 'required' and k.kind in ('skill', 'tool', 'method', 'domain')
            and not CREDENTIAL.search(k.term) and len(k.term.split()) <= 6 and not keyword_hits(k, text)]


async def add_proposed_bullets(provider, evidence, profile, draft, gaps):
    """Ask for bullets covering the gap skills and insert them as proposed
    claims. Invalid, anachronistic or misplaced bullets are dropped
    deterministically. Returns (draft, added, fit_reasons)."""
    fit_reasons = {}
    if not gaps or not hasattr(provider, 'propose'):
        return draft, 0, fit_reasons
    # Only roles that have bullets (jobs, projects) can host a proposed bullet;
    # never degrees or certifications.
    entries = {e.heading.id: e for s in draft.sections for e in s.entries if any(not b.proposed for b in e.bullets)}
    by_term = {normalize(k.term): k for k in profile.keywords}
    existing = {c.id for c in claims(draft)}
    proposals = await provider.propose(evidence, profile, draft, gaps, role_profiles(draft))
    # Distinctive tokens (tools, products, acronyms) per role, from the role's own source lines.
    def distinctive(text):
        return {w.strip('(),.;:') for w in text.split()
                if (any(c.isupper() for c in w[1:]) or w[:1].isupper() or any(c.isdigit() for c in w)) and len(w.strip('(),.;:')) > 2}
    role_tokens = {}
    for section in draft.sections:
        for e in section.entries:
            lines = [evidence.get(x, '') for b in e.bullets if not b.proposed for x in b.evidence_ids] + \
                    [evidence.get(x, '') for x in e.heading.evidence_ids]
            role_tokens[e.heading.id] = distinctive(' '.join(lines))
    added = 0
    # The writer may already have proposed bullets; the per-role cap of 2 includes them.
    per_entry = {e.heading.id: sum(1 for b in e.bullets if b.proposed) for s in draft.sections for e in s.entries}
    for bullet in proposals.bullets:
        entry = entries.get(bullet.entry_id)
        text = ' '.join(bullet.text.split()).lstrip('-–—•* ')
        if (entry is None or not text or re.search(r'\d', text)
                or per_entry.get(bullet.entry_id, 0) >= MAX_PROPOSED_PER_ROLE or added >= MAX_PROPOSED_TOTAL):
            continue
        covered = [g for g in gaps if keyword_hits(by_term[normalize(g)], normalize(text))]
        probe = Claim(id='p', text=text, evidence_ids=['E1'], proposed=True)
        if not covered or bullet_standard_issues(probe):
            continue
        # A tool that did not exist while the candidate held that role.
        if anachronistic(text, entry):
            continue
        # A rationale naming a different employer means the skill was placed in
        # the wrong company; the whole bullet is untrustworthy.
        reason_words = set((bullet.fit_reason or '').replace(',', ' ').split())
        own_company = company_words(entry)
        other_companies = set().union(*(company_words(e) for k, e in entries.items() if k != bullet.entry_id)) - own_company
        if reason_words & other_companies:
            continue
        # A proposal that names tools from a different role's evidence but not this
        # role's is borrowed work (Prophet/LSTM from TCS placed under Con Edison).
        own = role_tokens.get(bullet.entry_id, set())
        others = set().union(*(t for k, t in role_tokens.items() if k != bullet.entry_id)) - own
        borrowed = {w for w in distinctive(text) if w in others and normalize(w) not in {normalize(k.term) for k in profile.keywords}}
        if len(borrowed) >= 2:
            continue
        context = [e for e in bullet.context_evidence_ids if e in evidence] or entry.heading.evidence_ids
        per_entry[bullet.entry_id] = per_entry.get(bullet.entry_id, 0) + 1
        number = per_entry[bullet.entry_id]
        while f'{bullet.entry_id}p{number}' in existing:
            number += 1
        claim_id = f'{bullet.entry_id}p{number}'
        existing.add(claim_id)
        entry.bullets.append(Claim(id=claim_id, text=text, evidence_ids=context, proposed=True))
        fit_reasons[claim_id] = ' '.join((bullet.fit_reason or '').split())[:200]
        added += 1
    return draft, added, fit_reasons


AI_TELLS = ['leveraged', 'leveraging', 'leverage ', 'utilized', 'utilizing', 'utilising', 'utilise', 'spearheaded', 'robust', 'seamless',
            'seamlessly', 'comprehensive', 'cutting-edge', 'state-of-the-art', 'best-in-class', 'world-class',
            'transformative', 'innovative solutions', 'meticulous', 'pivotal', 'holistic', 'synergy', 'synergies',
            'delve', 'underscore', 'tapestry', 'in order to', 'a testament to', 'plays a key role',
            'designed to ensure', 'to drive', 'drive impactful', 'wide range of', 'various stakeholders']
GERUND_TAIL = re.compile(r',\s+\w+ing\b[^,]*\.?\s*$')
MAX_SAME_OPENER = 2


def restore_source_wording(resume, evidence, profile):
    """Put the candidate's own sentence back whenever the rewrite added no job
    keyword to it. Their prose is genuinely human: it varies in length, carries
    odd specifics, and cannot read as generated. Rewrites are kept only where
    they actually earn something."""
    scorable = [k for k in profile.keywords if k.kind not in ('credential', 'soft')]
    restored, used = 0, set()
    for section in resume.sections:
        for entry in section.entries:
            for bullet in entry.bullets:
                if bullet.proposed:
                    continue
                current = normalize(bullet.text)
                gained = {k.term for k in scorable if keyword_hits(k, current)}
                best = None
                for eid in bullet.evidence_ids:
                    line = evidence.get(eid, '')
                    if eid in used or len(line.split()) < 6 or re.match(r'^\s*[-–—•*]', line):
                        continue
                    source = normalize(line)
                    # The source must already carry every job keyword the
                    # rewrite carries, or the rewrite is doing real work.
                    if gained - {k.term for k in scorable if keyword_hits(k, source)}:
                        continue
                    # Token-level: character overlap calls unrelated sentences similar.
                    score = SequenceMatcher(None, current.split(), source.split()).ratio()
                    if best is None or score > best[0]:
                        best = (score, eid, ' '.join(line.split()))
                if best and best[0] >= 0.5 and len(best[2].split()) <= 38 and normalize(best[2]) != current:
                    bullet.text = best[2].rstrip('.') + '.'
                    used.add(best[1])
                    restored += 1
    return resume, restored


def source_style_report(evidence):
    """Measure the candidate's own resume the way a detector would. If the
    source is already machine-written, no downstream rewriting can make the
    result read as human, and the candidate needs to know that."""
    import statistics
    from collections import Counter
    lines = [' '.join(v.split()) for v in evidence.values() if len(v.split()) >= 8]
    if len(lines) < 5:
        return None
    lengths = [len(l.split()) for l in lines]
    tails = [l for l in lines if GERUND_TAIL.search(l)]
    openers = Counter(l.split()[0].lower() for l in lines)
    top, repeats = openers.most_common(1)[0]
    tail_share = round(100 * len(tails) / len(lines))
    machine_like = tail_share >= 35 or statistics.pstdev(lengths) < 4
    return {'sentences': len(lines), 'length_stdev': round(statistics.pstdev(lengths), 1),
            'gerund_tail_percent': tail_share, 'most_repeated_opener': top, 'opener_repeats': repeats,
            'reads_machine_written': machine_like,
            'advice': ('Your source resume itself shows the patterns AI detectors key on '
                       f'({tail_share}% of its sentences end in a ", ...ing ..." clause and "{top}" opens {repeats}). '
                       'This app keeps your wording wherever it can, so those patterns carry through. '
                       'Rewriting those source bullets in your own voice is the single biggest change you can make.')
            if machine_like else 'Your source resume reads as human-written.'}


def style_issues(resume, source_lines=frozenset()):
    """Detectors key on uniformity, so uniformity is a defect. Human resumes
    vary bullet length and shape; generated ones settle into one rhythm and
    lean on the same trailing ", ...ing ..." clause. These checks push a draft
    back to the writer until it reads like a person wrote it."""
    import statistics
    from collections import Counter
    everything = [b for section in resume.sections for entry in section.entries for b in entry.bullets]
    # A bullet that is the candidate's own sentence is exempt: rewriting their
    # words to satisfy a style rule would defeat the point of keeping them.
    bullets = [b for b in everything if normalize(b.text) not in source_lines]
    issues = []
    if not bullets:
        return issues
    for bullet in bullets:
        # Length, opening verb and cliches: wording, never a question of honesty.
        issues.extend(bullet_standard_issues(bullet))
    for bullet in bullets:
        lowered = bullet.text.lower()
        tell = next((phrase for phrase in AI_TELLS if phrase in lowered), None)
        if tell:
            issues.append(f'{bullet.id}: replace "{tell}" with the plain word a person would use.')
    # The trailing ", enabling/supporting/improving ..." clause is the loudest
    # generated-text signature; allow it in a minority of bullets only.
    tails = [b for b in bullets if GERUND_TAIL.search(b.text)]
    allowed = max(1, len(bullets) // 5)
    for bullet in tails[allowed:]:
        issues.append(f'{bullet.id}: end on the result itself instead of a ", ...ing ..." clause; '
                      'too many bullets share that shape.')
    openers = Counter(b.text.split()[0].lower() for b in bullets if b.text.split())
    for bullet in bullets:
        first = bullet.text.split()[0] if bullet.text.split() else ''
        if first and openers[first.lower()] > MAX_SAME_OPENER:
            openers[first.lower()] -= 1
            issues.append(f'{bullet.id}: "{first}" opens too many bullets; start this one with a different verb.')
    lengths = [len(b.text.split()) for b in bullets]
    if len(bullets) >= 6 and (statistics.pstdev(lengths) < 3.5 or max(lengths) - min(lengths) < 10):
        issues.append(f'{bullets[0].id}: every bullet is {min(lengths)}-{max(lengths)} words, which reads as generated; '
                      'vary the lengths deliberately, some 10-14 words and some 24-30.')
    return issues


def company_words(entry):
    """Capitalized words of a role heading that are not job titles: the company."""
    titles = {'senior', 'junior', 'lead', 'principal', 'staff', 'associate', 'intern', 'engineer', 'scientist',
              'analyst', 'developer', 'manager', 'consultant', 'specialist', 'data', 'machine', 'learning', 'ml', 'ai'}
    return {w.strip('.,|-') for w in entry.heading.text.split()
            if len(w.strip('.,|-')) > 2 and w[0].isupper() and w.strip('.,|-').lower() not in titles}


def prune_pointless_proposals(resume, evidence, profile):
    """A proposed bullet is unconfirmed content, so it must earn its place by
    carrying at least one job skill the candidate's evidence lacks. Anything
    else is filler and is removed before the candidate ever sees it."""
    normalized_evidence = normalize(' '.join(evidence.values()))
    pointless = set()
    for section in resume.sections:
        for entry in section.entries:
            for bullet in entry.bullets:
                if not bullet.proposed:
                    continue
                text = normalize(bullet.text)
                if not any(keyword_hits(k, text) and not keyword_in_evidence(k, normalized_evidence)
                           for k in profile.keywords if k.kind not in ('credential', 'soft')):
                    pointless.add(bullet.id)
    return drop_claims(resume, pointless), len(pointless)


def drop_claims(resume, ids):
    """Remove bullets by id (used for unsuitable or unconfirmed proposals)."""
    for section in resume.sections:
        for entry in section.entries:
            entry.bullets = [b for b in entry.bullets if b.id not in ids]
    return resume


def evaluate(profile, resume, audit, local_issues):
    ats = score_resume(profile, resume_plain_text(resume))
    unsupported = [c for c in audit.claim_checks if not c.supported]
    rules = {r.id: r for r in profile.eligibility}
    seen = set()
    eligibility = []
    for check in audit.eligibility:
        rule = rules.get(check.rule_id)
        if rule and rule.id not in seen:
            seen.add(rule.id)
            eligibility.append({'id': rule.id, 'text': rule.text, 'kind': rule.kind,
                                'status': check.status, 'reason': check.reason})
    for rule in profile.eligibility:
        if rule.id not in seen:
            eligibility.append({'id': rule.id, 'text': rule.text, 'kind': rule.kind,
                                'status': 'unknown', 'reason': 'Not assessed.'})
    factual_issues = list(local_issues) + [f'{c.claim_id}: {c.reason}' for c in unsupported]
    return {
        'score': ats['score'], 'components': ats['components'], 'keywords': ats['keywords'],
        'missing': ats['missing'], 'factual_pass': not factual_issues,
        'factual_issues': factual_issues, 'unsupported_ids': [c.claim_id for c in unsupported],
        'eligibility': eligibility, 'eligible': all(e['status'] == 'met' for e in eligibility),
        'questions': list(audit.questions)[:5], 'unverified_terms': unverified_terms(resume),
        'proposed_count': len(proposed_claims(resume)),
    }


async def complete_audit(provider, evidence, draft, profile):
    """Audit every non-verbatim claim exactly once; a missed claim gets one
    targeted retry and is otherwise treated as unsupported."""
    targets = claims_to_audit(draft, evidence)
    expected = {c.id for c in targets}
    proposed = proposed_claims(draft)
    audit = await provider.audit(evidence, targets, profile.eligibility, proposed)
    # A proposal survives only if the reviewer finds it practical for that
    # company and role, relevant to the job, and possible in that period.
    suitable = {c.claim_id for c in audit.proposal_checks
                if c.practical and c.relevant and c.period_consistent}
    rejected = [c.id for c in proposed if c.id not in suitable]
    drop_claims(draft, set(rejected))
    audit.proposal_checks = [c for c in audit.proposal_checks if c.claim_id in suitable]
    checks = {}
    for check in audit.claim_checks:
        if check.claim_id in expected:
            prior = checks.get(check.claim_id)
            if prior is None or (prior.supported and not check.supported):
                checks[check.claim_id] = check
    missing = expected - set(checks)
    if missing:
        retry = await provider.audit(evidence, [c for c in targets if c.id in missing], [])
        for check in retry.claim_checks:
            if check.claim_id in missing:
                checks[check.claim_id] = check
        for claim_id in missing - set(checks):
            checks[claim_id] = ClaimCheck(claim_id=claim_id, supported=False,
                                          reason='Not audited; reworded or removed conservatively.')
    audit.claim_checks = list(checks.values())
    return audit, rejected


async def run_workflow(request, provider, emit):
    resume_text = clean_markdown(strip_pasted_job(request.resume_text, request.job_text))
    evidence = make_evidence(resume_text + '\n' + request.additional_facts)
    await emit('analyzing', 'Extracting the job’s ATS keywords', 10)
    profile = ground_profile(await provider.profile(request.job_text), request.job_text)
    skeleton = parse_resume(evidence)
    payload = skeleton_payload(skeleton)
    before = score_resume(profile, resume_text)
    status = keyword_status(before, profile, evidence)
    best = None
    previous = feedback = None
    history = []
    last_result = None
    fit_reasons = {}
    review_notes = {}
    rounds = request.max_revisions + 1
    stop_reason = 'Revision limit reached; the best factual version was kept.'
    try:
        for version in range(rounds):
            step = 80 // rounds
            await emit('writing', 'Writing your tailored resume' if not version else f'Revising · round {version + 1}',
                       15 + version * step)
            rewrite = await provider.write(evidence, profile, status, payload, previous, feedback)
            draft = normalize_skills(assemble(skeleton, rewrite, evidence), evidence, profile)
            draft, backfilled = backfill_bullets(draft, evidence, profile)
            draft = tidy_layout(draft)
            local = validate_draft(draft, evidence, profile)
            source_lines = {normalize(v) for v in evidence.values()}
            style = style_issues(draft, source_lines)
            repaired = []
            if local or style:
                await emit('writing', 'Repairing bullets that overstate your evidence or read as generated',
                           15 + version * step + step // 3)
                repaired = local + style
                draft, local = await repair_locally(provider, evidence, profile, draft, local + style)
            # Wording is polish: it drives another writing round but must never
            # fail a run the way an unsupported claim does.
            summary_ids = {c.id for c in draft.summary}
            polish = [i for i in local if i.split(':', 1)[0] in summary_ids]
            local = [i for i in local if i not in polish]
            gaps = [] if local else skill_gaps(draft, profile)
            if gaps:
                await emit('writing', 'Drafting practical bullets for the job skills your resume does not mention',
                           15 + version * step + step // 3)
                draft, _, reasons = await add_proposed_bullets(provider, evidence, profile, draft, gaps)
                fit_reasons.update(reasons)
            draft, _ = prune_pointless_proposals(draft, evidence, profile)
            draft, trimmed = trim_bullets(draft, profile)
            draft, restored = restore_source_wording(draft, evidence, profile)
            # Measure the final bullet set, proposals included, and give it one
            # cheap pass to break up any remaining uniformity.
            remaining_style = style_issues(draft, source_lines)
            if remaining_style:
                await emit('writing', 'Varying the wording so the resume reads as human-written',
                           15 + version * step + step // 2)
                draft, _ = await repair_locally(provider, evidence, profile, draft, remaining_style)
            polish += style_issues(draft, source_lines)
            ats = score_resume(profile, resume_plain_text(draft))
            status = keyword_status(ats, profile, evidence)
            if local and version < rounds - 1:
                # Cheap deterministic failure: rewrite before paying for an audit.
                history.append({'version': version, 'score': ats['score'], 'factual_pass': not local, 'audited': False,
                                'issues': (local + polish)[:6], 'repaired': repaired[:6]})
                previous, feedback = draft, {'local_issues': local + polish, 'missing_keywords': status['missing'],
                                             'factual_issues': []}
                continue
            await emit('reviewing', 'Auditing every claim against your evidence', 15 + version * step + step // 2)
            audit, rejected = await complete_audit(provider, evidence, draft, profile)
            review_notes.update({c.claim_id: c.reason for c in audit.proposal_checks})
            ats = score_resume(profile, resume_plain_text(draft))
            status = keyword_status(ats, profile, evidence)
            result = evaluate(profile, draft, audit, local)
            result['rejected_proposals'] = len(rejected)
            last_result = result
            result['bullet_gaps'] = bullet_gaps(draft, profile)
            history.append({'version': version, 'score': result['score'], 'factual_pass': result['factual_pass'],
                            'audited': True, 'issues': result['factual_issues'][:6], 'repaired': repaired[:6],
                            'polish': polish[:4], 'bullet_gaps': result['bullet_gaps'],
                            'proposed': result['proposed_count'], 'trimmed': trimmed, 'backfilled': backfilled,
                            'source_wording_restored': restored})
            candidate = (draft, result)
            if not result['factual_pass'] and not local:
                stripped = strip_unsupported(draft, set(result['unsupported_ids']))
                if stripped is not None:
                    kept = Audit(claim_checks=[c for c in audit.claim_checks if c.supported],
                                 proposal_checks=audit.proposal_checks,
                                 eligibility=audit.eligibility, questions=audit.questions)
                    stripped_result = evaluate(profile, stripped, kept, [])
                    stripped_result['stripped'] = len(result['unsupported_ids'])
                    candidate = (stripped, stripped_result)
            # Higher score wins; on a tie a draft that needed no stripping wins.
            rank = lambda item: (item['score'], not item.get('stripped'))
            if candidate[1]['factual_pass'] and (best is None or rank(candidate[1]) > rank(best[1])):
                best = (*candidate, version)
            if result['factual_pass'] and result['score'] >= request.target_score and not result['bullet_gaps']:
                stop_reason = 'Target ATS score reached with every statement supported by your evidence.'
                break
            if result['factual_pass'] and not result['missing']:
                stop_reason = 'Every job keyword is present; remaining points depend on title and format checks.'
                break
            previous_scores = [h['score'] for h in history[:-1] if h.get('audited')]
            if result['factual_pass'] and previous_scores and result['score'] - max(previous_scores) < 2:
                stop_reason = 'Further revision would not raise the score meaningfully; the best factual version was kept.'
                break
            previous = draft
            feedback = {'missing_keywords': result['missing'], 'factual_issues': result['factual_issues'],
                        'local_issues': local + polish,
                        'required_keywords_absent_from_experience_bullets': result['bullet_gaps']}
    except BudgetExceeded:
        if best is None:
            raise
        stop_reason = 'The token budget was reached; the best audited version was kept.'
    if best is None:
        detail = '; '.join((last_result or {}).get('factual_issues', [])[:3])
        raise ValueError('No draft passed the factual audit. ' + detail)
    draft, result, version = best
    if result.get('stripped'):
        stop_reason += f' {result["stripped"]} unsupported statement(s) were removed automatically.'
    common = {'resume': draft.model_dump(), 'report': result, 'job': profile.model_dump(),
              'history': history, 'selected_version': version, 'stop_reason': stop_reason,
              'before': {'score': before['score'], 'components': before['components'],
                         'matched': len(before['keywords']) - len(before['missing']), 'total': len(before['keywords'])},
              'usage': provider.usage() if hasattr(provider, 'usage') else {'calls': provider.calls, 'tokens': provider.tokens},
              'source_style': source_style_report(evidence), 'evidence': evidence}
    terms = unverified_terms(draft)
    proposed = proposals(draft, evidence, profile, fit_reasons, review_notes)
    if terms or proposed:
        from .confirmation import without_terms
        floor_draft = drop_claims(without_terms(draft, set(terms)), {p['id'] for p in proposed})
        floor = score_resume(profile, resume_plain_text(floor_draft))
        common.update(unverified_terms=terms, proposals=proposed, floor_score=floor['score'],
                      document_checks=None, docx=None,
                      stop_reason=stop_reason + ' Confirm the proposed bullets and highlighted skills you can stand behind before download.',
                      _pending={'profile': profile.model_dump(), 'draft': draft.model_dump(),
                                'audit': audit_for(result), 'evidence': evidence, 'proposals': proposed})
        return common
    await emit('exporting', 'Preparing your Word document', 96)
    data, document_checks = render_docx(draft)
    return {**common, 'unverified_terms': [], 'proposals': [], 'document_checks': document_checks, 'docx': data}


def audit_for(result):
    return {'eligibility': result['eligibility'], 'questions': result['questions']}
