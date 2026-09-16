from pathlib import Path
from app.skeleton import clean_markdown, parse_resume, strip_pasted_job
from app.workflow import make_evidence

FIXTURES = Path(__file__).parent / 'fixtures'
JOB = (Path(__file__).resolve().parents[1] / 'demo_files' / 'job description.txt').read_text()


def skeleton_of(name):
    text = clean_markdown(strip_pasted_job((FIXTURES / name).read_text(), JOB))
    return parse_resume(make_evidence(text)), text


def test_tab_dated_layout():
    sk, _ = skeleton_of('resume_layout_tabs.txt')
    assert sk.headline.text == 'Senior AI / ML Engineer' and len(sk.contact) == 1
    kinds = [(s.kind, s.heading) for s in sk.sections]
    assert kinds == [('summary', 'Professional Summary'), ('entries', 'Work Experience'), ('skills', 'Technical Skills'),
                     ('entries', 'Education'), ('entries', 'Certifications')]
    jobs = sk.sections[1].entries
    assert [e.heading.text for e in jobs] == ['Con Edison, Senior AI/ML Engineer', 'Munich Re, ML Engineer',
                                              'Tata Consultancy Services, Machine Learning Engineer', 'Sosio Technologies, Associate Data Scientist']
    assert [len(e.bullets) for e in jobs] == [12, 9, 7, 6]
    assert jobs[0].detail.text == 'January 2025 – Present | New York City, NY'
    assert [e.detail.text for e in sk.sections[3].entries] == ['January 2023 - December 2024', 'June 2016 – May 2020']
    assert len(sk.sections[2].lines) == 4


def test_location_lines_year_education_and_pasted_job_posting():
    sk, text = skeleton_of('resume_layout_lines_with_pasted_jd.txt')
    assert 'Key Responsibilities' not in text and 'Benefits' not in text, 'pasted job posting removed'
    assert 'AI safety, Generative AI, LLMs, Python' not in text, 'keyword soup above the posting removed'
    kinds = [(s.kind, s.heading) for s in sk.sections]
    assert kinds == [('summary', 'Professional Summary'), ('skills', 'Skills'), ('entries', 'Experience'),
                     ('entries', 'Education'), ('entries', 'Certifications')]
    jobs = sk.sections[2].entries
    assert [e.heading.text for e in jobs] == ['Senior AI/ML Engineer — Con Edison', 'ML Engineer — Munich Re',
                                              'Machine Learning Engineer — Tata Consultancy Services', 'Associate Data Scientist — Sosio Technologies']
    assert [e.detail.text for e in jobs] == ['January 2025 – Present | New York City, NY', 'May 2024 – December 2024 | Princeton, NJ',
                                             'June 2020 – January 2022 | Hyderabad, India', 'Dec 2019 – May 2020 | Hyderabad, India']
    assert [len(e.bullets) for e in jobs] == [6, 6, 6, 6]
    education = sk.sections[3].entries
    assert [(e.heading.text, e.detail.text) for e in education] == [('Master in Data Analytics Engineering', '2024 | George Mason University'),
                                                                     ('Bachelor in Information Technology', '2020 | GMR Institute of Technology')]
    cert = sk.sections[4].entries
    assert [e.heading.text for e in cert] == ['Databricks Certified Data Engineer Associate'] and cert[0].bullets == []
    assert len(sk.sections[1].lines) == 8 and len(sk.sections[0].lines) == 20
