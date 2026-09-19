# Resume Studio

A local web app that rewrites your resume for a specific job description so that
applicant tracking systems (ATS) score it highly, while every statement of fact
stays backed by your own resume text.

A dozen small model calls do the writing; Python owns the parsing, scoring,
factual gates, confirmation step and Word export.

## Run it on any laptop

Needs Python 3.11 or newer and an OpenAI API key. Nothing else.

**macOS / Linux**

```bash
./setup.sh      # creates .venv, installs dependencies, creates .env
./start.sh      # http://127.0.0.1:8765
```

**Windows (PowerShell)**

```powershell
.\setup.ps1
.\start.ps1
```

After `setup`, open `.env` and paste your key into `OPENAI_API_KEY`. That is
the only required edit; the defaults use `gpt-4.1-mini` and `gpt-4.1-nano`
(about $0.02 per resume).

**Docker** (alternative, no Python install needed)

Install Docker Desktop (https://www.docker.com/products/docker-desktop/) and
start it. Then, in the project folder:

```bash
cp .env.example .env          # Windows PowerShell: copy .env.example .env
# open .env and paste your OPENAI_API_KEY
docker compose up --build -d  # first run builds the image (~1 minute)
```

Open http://127.0.0.1:8765. `docker compose logs -f` shows the server log,
`docker compose down` stops it, `docker compose up --build -d` again after
pulling code changes. The port is published on localhost only; set
`PORT=9000` before the command to use another host port.

Open http://127.0.0.1:8765. **Try an example** runs a fixed fictional demo
with no API calls, so you can check the install before adding a key.

### Sharing the project

Share the folder without `.env` (your API key) and without `.venv`
(machine-specific). The easiest way is git:

```bash
git clone <your-repo-url>
cd resume-studio
./setup.sh && ./start.sh
```

`.gitignore` already excludes `.env`, `.venv`, caches and generated Word
files. `demo_files/` contains a real resume; delete it before sharing with
anyone who should not see it.

### Providers

`AI_PROVIDER` in `.env` selects `openai` (default), `openrouter` or
`claude`; each has its own key and model variables in `.env.example`. The
model must support structured JSON output. Tested live with `gpt-4.1-mini`
and `claude-sonnet-5`.

## How it works

1. **Extract** (1 call). The model pulls the job title, up to 60 ATS keywords
   with aliases (`Machine Learning` → `ML`, `Spark` → `PySpark`), a
   required/preferred priority for each, and any explicit screening rules
   (degree, years, location). Python discards any keyword that does not
   literally appear in the job text.
2. **Score the original** (no call). Deterministic keyword matching gives the
   "before" score.
3. **Write** (one small call per role plus one for the header, in parallel).
   Python first parses your resume into a fixed skeleton:
   name, contact, section order, and every job, degree and certification with
   its heading and dates. The writer receives that skeleton, the numbered
   evidence lines, the keyword list and exactly which keywords are still
   missing. Each role call sees only that role's source lines and returns
   5-6 bullets; the header call returns summary, headline and skill groups.
   Small outputs are what cheap models do reliably. The header call runs
   after the role calls so the summary can cite the strongest final
   bullets: three sentences (identity with years and specialization,
   proof with real metrics, fit with the job's tools), checked by Python
   for length, role name, years, keyword count and banned filler. On
   revision only the roles named in the feedback are rewritten. Python reassembles the resume
   around the verbatim skeleton,
   so companies, titles, dates, degrees and certifications cannot be changed
   or dropped. Every bullet and summary line cites the evidence lines that
   support it.
   Keywords your resume never mentions go into the **Skills section** tagged
   as unevidenced. An evidence-backed bullet can never contain one, and Python
   enforces that check. If a bullet slips one in, one small **repair** call
   rewrites just those bullets instead of regenerating the resume.
   Python also verifies that every dated entry in your source survived.
   Each role keeps 5 to 6 bullets: the writer picks the most relevant, Python
   trims the least relevant beyond 6 (never a proposed bullet or the only
   bullet carrying a matched keyword) and backfills below 5 with the role's
   most relevant original lines. Every bullet is checked against a standard
   (capitalized action verb, 10 to 30 words, no first person, no cliches);
   failures go to the repair call.
   The Skills section is curated the way a recruiter reads it: at most 5
   groups of 8, every item either a job keyword or a tool the bullets
   demonstrate (orphan tools are dropped), the candidate's own spelling
   (TensorFlow, not tensorflow), no soft skills, degrees or screening
   phrases, compounds deduplicated ("AWS SageMaker" folds into "AWS
   (SageMaker, S3, Glue)"), and at most 10 unverified job terms, required
   first. Soft skills and degree fields are reported separately and do not
   count toward the keyword score.
4. **Propose** (1 small call when needed). Job skills should show up in
   experience bullets, not only in Skills. Python lists every job skill,
   required and preferred, that appears in no bullet. The model receives a
   profile of each role (company, title, period, and the technologies that
   role already demonstrates) and writes a practical bullet for each gap
   skill under the role where such work would naturally have happened: a
   cloud tool with the role on that cloud, a modelling method with the
   modelling role, governance with the role that owns pipelines. Each bullet
   comes with a `fit_reason` saying why it is realistic there.
   Python then rejects any proposal that names a tool which did not exist
   during that employment period, whose rationale names a different employer,
   that borrows another role's tools, that carries a number, or that adds no
   missing skill at all. At most 3 per role and 10 per run.
   Proposed bullets are highlighted in the preview and **deleted unless you
   tick them** on the confirm screen. They are a prompt to remember work you
   did but never wrote down, not a claim about you.
5. **Audit** (1 call, chunked in parallel for long resumes). A second model
   pass checks each reworded claim against its cited evidence. It also judges
   every proposed bullet on three separate verdicts: **practical** (ordinary
   work for that company, team and title), **relevant** (matters for the
   target job) and **period consistent** (those tools existed then). A
   proposal survives only if all three hold; the reviewer's one-line reason
   is shown next to it on the confirm screen. The audit also judges the
   screening rules and lists facts that would help. Verbatim lines are not
   sent for audit.
6. **Revise** (up to 3 rounds, default 2). If the score is below target or the
   audit found an unsupported statement, the writer gets the exact missing
   keywords and factual issues back. If revisions run out, unsupported bullets
   are removed automatically rather than shipped.
7. **Confirm**. Proposed bullets and unevidenced skills are listed with
   checkboxes and highlighted in the preview. Uncheck anything you did not do
   or could not discuss in an interview. The score is recomputed and the Word
   file is generated.

Typical live run with gpt-4.1-mini everywhere and gpt-4.1-nano for the
repair/propose calls: 10-15 small calls, 45-80 seconds, about 50k input and
8k output tokens, roughly $0.03. The run report shows the estimate.

### Model settings for cheap runs

| variable | role | suggestion |
| --- | --- | --- |
| `OPENAI_MODEL` | keyword extraction from the job | `gpt-4.1-mini` (drives everything; do not go lower) |
| `WRITER_MODEL` | per-role bullets, header | `gpt-4.1-mini` |
| `REVIEWER_MODEL` | factual audit | `gpt-4.1-mini` (or `gpt-4.1` for a stricter audit) |
| `CHEAP_MODEL` | repair and propose | `gpt-4.1-nano`; invalid output falls back to the writer |

OpenRouter and Claude have the same four slots with `OPENROUTER_` / `CLAUDE_`
prefixes (`claude-haiku-4-5` is the cheap Claude choice). Prompt caching is enabled for the shared evidence block on the
Anthropic path.

## Does it read as AI-written?

Detectors key on uniformity, not on truth, so the app treats uniformity as a
defect:

- **Your sentences are kept.** After the rewrite, any bullet whose rewording
  added no job keyword is replaced by your original sentence, verbatim. Your
  own prose is human text, varies naturally in length, and carries the odd
  specifics that generated writing lacks. Restored sentences are then exempt
  from every style rule: the app will not "fix" your words.
- **Uniformity is checked in Python** on what the model did write: bullet
  length spread, the trailing ", ...ing ..." clause that generated text leans
  on, any verb opening more than two bullets, and a list of giveaway words
  (leveraged, utilized, robust, seamless, comprehensive, spearheaded ...).
  Failures go to a cheap repair call. They are polish, so they can never fail
  a run.
- **Your source is measured too**, and the report says plainly when the source
  resume itself reads as machine-written.

That last point matters most. If the resume you upload was itself written by
an AI, its patterns carry through, because the app deliberately preserves your
wording and cannot invent facts to replace it. Measured on the two sample
resumes, output style tracks input style closely. The fix is upstream:
rewrite your source bullets in your own voice and rerun.

The app does not try to defeat detectors with invisible characters, homoglyphs
or deliberate typos. Those are deceptive, and they break the ATS text
extraction this whole tool exists to pass.

## The ATS score

Deterministic, reproducible, and independent of the model:

| Component | Points | Rule |
| --- | ---: | --- |
| Required keywords | 70 | fraction present, exact match after normalization, aliases count |
| Preferred keywords | 15 | same rule |
| Job title | 10 | exact title in resume (5 if only the role noun, e.g. "Data Scientist" for "Senior Data Scientist") |
| Format | 5 | email, phone, Experience / Education / Skills headings present |

Matching is case-insensitive, tolerant of hyphen and spacing differences, and
accepts simple plurals. Substrings do not count (`PostgreSQL` is not `SQL`).
This mirrors how keyword-based ATS scanners rank resumes. It is not an
interview prediction.

## Honesty model

- Name, contact, employers, titles, dates, degrees and certifications are
  copied from your source.
- A number may appear only if the same number appears somewhere in your
  evidence.
- A job keyword may appear in an evidence-backed bullet, heading or summary
  only if your evidence contains it (or an inflection or alias). Otherwise it
  can appear only in a proposed bullet or an unevidenced Skills item, both of
  which you confirm before download.
- Proposed bullets are suggestions, never assertions. They exist because most
  missing keywords are work people did and forgot to write down. Tick only
  what you actually did and could discuss in an interview; everything you
  leave unticked is deleted and the score is recomputed without it.
- Every employer, degree and certification with dates in your source must
  appear in the draft, or the draft is rejected.
- The audit model can still misjudge a paraphrase. Read the preview before
  applying.

## Code map

| File | Purpose |
| --- | --- |
| `app/main.py` | API, local request boundary, background jobs, confirmation, download |
| `app/models.py` | Structured-output contracts (job profile, resume, audit) |
| `app/scoring.py` | Deterministic keyword extraction grounding and ATS scoring |
| `app/skeleton.py` | Deterministic resume parser and reassembly around the verbatim skeleton |
| `app/agents.py` | Prompts and the OpenAI, OpenRouter and Claude adapters |
| `app/workflow.py` | Validation, audit orchestration, revision loop, best-draft selection |
| `app/confirmation.py` | Skill confirmation and final export |
| `app/documents.py` | Upload extraction and Word export with round-trip check |
| `app/demo.py` | Fictional fixture used by the demo and the tests |
| `static/` | Single-page frontend, no build step |
| `setup.sh`, `start.sh`, `setup.ps1`, `start.ps1` | One-command install and start on macOS/Linux and Windows |
| `Dockerfile`, `docker-compose.yml` | Container alternative |

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

83 tests cover skeleton parsing, keyword matching, profile grounding, evidence enforcement,
entry integrity, bullet limits and standards, the targeted repair and propose
calls, the revision loop,
automatic stripping of unsupported statements, budget handling, the
confirmation flow, the API boundary and the Word round trip.
Live generation was verified with Claude Sonnet on a real resume and job
description: before 40 / after 100 with the factual audit passing.

## Privacy and limits

Single-user local app bound to 127.0.0.1. Uploaded files are parsed locally
and not retained. Confirmed text is sent to the selected provider during live
generation. Run state and generated documents stay in memory for one hour or
until cleared. Do not expose this process to the internet without adding
authentication, durable storage and rate limits.
