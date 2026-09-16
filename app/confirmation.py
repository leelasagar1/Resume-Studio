"""Final user attestation for unevidenced skill keywords."""
from .models import JobProfile, Resume, resume_plain_text
from .scoring import normalize, score_resume


def without_terms(resume: Resume, removed_terms: set[str]) -> Resume:
    """Return a copy with the given unevidenced skill items removed."""
    removed = {normalize(t) for t in removed_terms}
    result = resume.model_copy(deep=True)
    for section in result.sections:
        for group in section.skill_groups:
            group.items = [i for i in group.items if i.evidence_ids or normalize(i.text) not in removed]
        section.skill_groups = [g for g in section.skill_groups if g.items]
    return result


def finalize_pending(pending, accepted_terms, accepted_claim_ids=()):
    from .documents import render_docx
    from .workflow import drop_claims
    profile = JobProfile.model_validate(pending['profile'])
    draft = Resume.model_validate(pending['draft'])
    offered = {normalize(i.text): i.text for s in draft.sections for g in s.skill_groups
               for i in g.items if not i.evidence_ids}
    accepted = {normalize(t) for t in accepted_terms}
    if len(accepted) != len(accepted_terms) or not accepted <= offered.keys():
        raise ValueError('Confirmation references an unknown or duplicate skill.')
    offered_claims = {p['id']: p for p in pending.get('proposals', [])}
    accepted_claims = set(accepted_claim_ids)
    if len(accepted_claims) != len(accepted_claim_ids) or not accepted_claims <= offered_claims.keys():
        raise ValueError('Confirmation references an unknown or duplicate proposed bullet.')
    final = drop_claims(without_terms(draft, set(offered) - accepted), set(offered_claims) - accepted_claims)
    evidence = dict(pending['evidence'])
    for section in final.sections:
        for group in section.skill_groups:
            for item in group.items:
                if not item.evidence_ids:
                    # The candidate's explicit attestation is the evidence for this item.
                    evidence['U1'] = 'Candidate confirmed these skills at download time.'
                    item.evidence_ids = ['U1']
        for entry in section.entries:
            for bullet in entry.bullets:
                if bullet.proposed:
                    evidence['U2'] = 'Candidate confirmed these proposed responsibilities at download time.'
                    bullet.proposed = False
                    bullet.evidence_ids = ['U2']
    ats = score_resume(profile, resume_plain_text(final))
    report = {'score': ats['score'], 'components': ats['components'], 'keywords': ats['keywords'],
              'missing': ats['missing'], 'factual_pass': True, 'factual_issues': [], 'unsupported_ids': [],
              'eligibility': pending['audit']['eligibility'],
              'eligible': all(e['status'] == 'met' for e in pending['audit']['eligibility']),
              'questions': pending['audit']['questions'], 'unverified_terms': [], 'proposed_count': 0}
    data, checks = render_docx(final)
    return {'resume': final.model_dump(), 'report': report, 'docx': data, 'document_checks': checks, 'evidence': evidence,
            'proposals': [],
            'confirmation_decisions': [{'term': text, 'accepted': key in accepted} for key, text in offered.items()] +
                                      [{'term': p['text'], 'accepted': pid in accepted_claims} for pid, p in offered_claims.items()]}
