# Resume Studio

A local web app that rewrites your resume for a specific job description to improve
job-keyword coverage, while every statement of fact
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
the only required edit; the OpenAI defaults use `gpt-5.6-luna`. Runtime and
cost depend on resume length, reasoning tokens and revision count.

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

## Long master resumes

Paste or upload a multi-page master resume (up to 60,000 characters, PDF up to
25 pages) with every role and every bullet you have. The app picks what fits
the job:

- Repeated page headers (your name and contact on every page) and page
  numbers are removed before parsing, and bullet markers from Word lists and
  PDFs are kept, so a short bullet is never mistaken for a section name.
- Each role writer sees that role's 12 most job-relevant source bullets and
  returns 5-6; the summary writer sees the 4 strongest per role.
- Past about 120 source lines, each call sees only the lines it needs (the
  audit checks each claim against the lines it cites), so tokens grow with
  what is used, not with the length of the document.
- The token budget scales with the resume's length; `MAX_RUN_TOKENS` stays
  the floor.

A fictional 22,000-character, 7-role, 142-bullet master resume ran in about
75 seconds for about $0.025 with gpt-5.6-luna.

## How it works

0. **Check the job's keywords** (the first screen after you submit; untick
   "Let me check the job's keywords before writing" to skip it). You see every
   keyword the job will be scored on, split into "already in your resume" and
   "not in your resume". For each missing one you choose:
   - **I have this**: pick the role and write one line on what you did, and it
     becomes a real source bullet of that role, cited and audited like the
     rest of your resume. With no detail it is listed under Skills.
   - **I don't have this**: it is never proposed, in a bullet or in Skills.
   - **Not a real requirement**: it leaves the score.
   - **Suggest a bullet to confirm** (the default): the old behaviour.
   Saying what is true up front replaces most of the invented bullets with
   your own facts, so there is far less to confirm at the end. The analysis is
   cached per job posting (`data/profiles/`): the same ad always gives the
   same keyword list, and the writing step that follows does not pay for it
   again.
1. **Extract** (1 call, skipped when the posting was already analysed). The model pulls the job title, up to 60 ATS keywords
   with aliases (`Machine Learning` → `ML`, `Spark` → `PySpark`), a
   required/preferred priority for each, and any explicit screening rules
   (degree, years, location). Python discards any keyword that does not
   literally appear in the job text. A section heading is never taken as the
   title ("MINIMUM REQUIREMENTS"): enter the **Target job title** field when the
   posting does not start with one; with no title, its 10 points go to the
   keywords. Items that appear only in an "at least 2 of the following",
   "such as" or "e.g." list are preferred, not required, since any one of
   them satisfies the ad. An "Alternatively, ..." screening rule counts as
   either/or with the rule before it.
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
   in parallel with the role calls and reads the candidate's own bullets:
   three sentences (identity with years and specialization, proof with real
   metrics, fit with the job's tools), checked by Python for length, role
   name, years, keyword count and banned filler. The identity is the title
   the candidate actually held; the headline carries the target title. A
   seniority the evidence never shows ("Senior" when no role was senior) is
   removed by Python before the audit ever sees it. On
   revision only the roles named in the feedback are rewritten. Python reassembles the resume
   around the verbatim skeleton,
   so companies, titles, dates, degrees and certifications cannot be changed
   or dropped. Every bullet and summary line cites the evidence lines that
   support it.
   Keywords your resume never mentions go into the **Skills section** tagged
   as unevidenced; Python adds any the writer left out, required first. An evidence-backed bullet can never contain one, and Python
   enforces that check. If a bullet slips one in, one small **repair** call
   rewrites just those bullets instead of regenerating the resume.
   Python also verifies that every dated entry in your source survived.
   Each role keeps 5 to 6 bullets: the writer picks the most relevant, Python
   trims the least relevant beyond 6 (never a proposed bullet or the only
   bullet carrying a matched keyword) and backfills below 5 with the role's
   most relevant original lines. Roles that ended more than ten years ago are
   condensed to 2-3 bullets and never receive proposed bullets. Every bullet is checked against a standard
   (capitalized action verb, 10 to 30 words, no first person, no cliches);
   failures go to the repair call.
   The Skills section is curated the way a recruiter reads it: at most 5
   groups of 8, every item either a job keyword or a tool the bullets
   demonstrate (orphan tools are dropped), the candidate's own spelling
   (TensorFlow, not tensorflow), no soft skills, degrees or screening
   phrases, compounds deduplicated ("AWS SageMaker" folds into "AWS
   (SageMaker, S3, Glue)"), and at most 12 unverified job terms, required
   first. Soft skills and degree fields are reported separately and do not
   count toward the keyword score, and neither do qualities or lone generic
   words ("robust models", "decision-driven", "engineering"): no recruiter
   searches for them, and scoring them made 90 unreachable.
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
   missing skill at all. At most 2 per role and 10 per run. Each skill is
   offered to the proposer once; accepted proposals are carried into later
   rounds instead of being drafted and reviewed again.
   Proposed bullets are highlighted in the preview and **deleted unless you
   tick them** on the confirm screen. They are a prompt to remember work you
   did but never wrote down, not a claim about you.
5. **Audit** (small parallel calls: claims in chunks of 12, plus one for
   proposals and screening rules). A second model
   pass checks each reworded claim against its cited evidence. It also judges
   every proposed bullet on three separate verdicts: **practical** (ordinary
   work for that company, team and title), **relevant** (matters for the
   target job) and **period consistent** (those tools existed then). A
   proposal survives only if all three hold; the reviewer's one-line reason
   is shown next to it on the confirm screen. The audit also judges the
   screening rules and lists facts that would help. Verbatim lines, and the
   name, contact, role headings and dates Python copied from your resume, are
   not sent for audit. Verdicts are cached by claim text, so a later round
   audits only what changed.
6. **Revise** (up to 3 rounds, default 2). Up to four unsupported statements
   are fixed in place: one repair call and a re-audit of just those lines.
   A new writing round runs only when something can still change: more
   unsupported statements, or job keywords your evidence has that no bullet
   shows yet. The writer then gets the exact missing keywords and factual
   issues back. If revisions run out, unsupported bullets
   are removed automatically rather than shipped.
7. **Confirm**. Proposed bullets and unevidenced skills are listed with
   checkboxes, **all unticked**, and highlighted in the preview. Tick only what
   you did and could discuss in an interview. Until you confirm, the headline
   score is what your resume supports on its own; the screen also shows the
   score with every suggestion ticked. The score is recomputed and the Word
   file is generated.

Typical live run with gpt-5.6-luna on the demo resume: one round, 10-13
small calls, 60-80 seconds, about 35k input and 8k output tokens, roughly
$0.02. The run report shows the estimate.

### Model settings for GPT-5.6 Luna

| variable | role | suggestion |
| --- | --- | --- |
| `OPENAI_MODEL` | keyword extraction from the job | `gpt-5.6-luna` |
| `WRITER_MODEL` | per-role bullets, header | blank: inherits extraction model |
| `REVIEWER_MODEL` | factual audit | blank: inherits extraction model |
| `CHEAP_MODEL` | repair and propose | blank: inherits writer; an explicit override is preserved |

Luna uses `low` reasoning for extraction/writing/repair/proposals and `medium`
for factual audits. Output limits include an additional 4,096 reasoning tokens
(8,192 for audits); a token-truncated OpenAI response gets one retry with twice
the output allowance, subject to the run budget. These are initial tuning
choices, not measured guarantees of better quality. Strict JSON schemas,
`store=False`, evidence checks and confirmation remain enabled.

Existing installations: set `OPENAI_MODEL=gpt-5.6-luna`, clear the other three
OpenAI model variables to inherit it, and restart the server. The browser
remembers previous choices, so select `gpt-5.6-luna` in the model field too.
OpenRouter and Claude defaults remain unchanged; selecting Luna through
OpenRouter uses the Luna reasoning settings, subject to that provider's access.

[Official Luna documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
lists structured outputs and configurable reasoning. It describes Luna as a
cost-sensitive model; benchmark factual quality instead of assuming a newer
name guarantees better resumes. Estimates use standard uncached token prices;
provider discounts, cache billing and long-context adjustments are not modeled.

To compare both models on the **fictional built-in sample** (paid API calls):

```bash
python scripts/evaluate_models.py --live --repeats 3 > /tmp/resume-model-evaluation.json
```

This reports factual pass, evidence-only coverage, draft coverage, proposals,
revisions, latency and usage. Model self-audit is not independent ground truth;
review the text manually and add varied, human-labeled fixtures before treating
this small benchmark as a quality decision. See [project review](PROJECT_REVIEW.md).

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
| `app/review.py` | Keyword review before writing: per-posting analysis cache, the candidate's decisions |
| `app/workflow.py` | Validation, audit orchestration, revision loop, best-draft selection |
| `app/confirmation.py` | Skill confirmation and final export |
| `app/documents.py` | Upload extraction and Word export with round-trip check |
| `app/demo.py` | Fictional fixture used by the demo and the tests |
| `static/` | Single-page frontend, no build step |
| `setup.sh`, `start.sh`, `setup.ps1`, `start.ps1` | One-command install and start on macOS/Linux and Windows |
| `Dockerfile`, `docker-compose.yml` | Container alternative |
| `scripts/benchmark.py`, `benchmarks/` | Paid regression benchmark over fictional resumes and job ads |
| `data/` (git-ignored) | Cached job analyses, `runs.jsonl` run log, benchmark results |

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Offline tests cover skeleton parsing, keyword matching, profile grounding, evidence enforcement,
entry integrity, bullet limits and standards, the targeted repair and propose
calls, the revision loop,
automatic stripping of unsupported statements, budget handling, the
confirmation flow, the API boundary and the Word round trip.
Live generation was verified with Claude Sonnet on a real resume and job
description: before 40 / after 100 with the factual audit passing.

## Run log

Every live run appends one line to `data/runs.jsonl`: time per step, calls,
tokens, cost, the three scores (original, tailored without suggestions, with
every suggestion), rounds, whether the first audit passed, your review
decisions as counts, and the stop reason. It never contains resume or job
text, so you can share it when asking why a run was slow or scored low.
A later `confirmed` line records how many suggestions you accepted and the
final score. In Docker the folder is mounted from `./data`, so it survives
rebuilds.

```bash
tail -n 5 data/runs.jsonl
```

## Benchmark

```bash
python scripts/benchmark.py --live --repeats 2
```

Runs the full workflow over nine fixed pairs of fictional resumes and job
ads in `benchmarks/` (a requirements-only ad with no title, "at least 2 of"
lists, a filler-heavy ad, a 22,000-character master resume, and one
deliberate misfit) and prints per-run time, cost, rounds, first-audit result,
the three scores, junk Skills items and the score spread across repeats.
About $0.02 per run. Use it before and after any change to the prompts or the
rules: one good run proves nothing, because scores move a few points between
runs. Results are saved under `data/benchmarks/`.

## Privacy and limits

Single-user local app bound to 127.0.0.1. Uploaded files are parsed locally
and not retained. Confirmed text is sent to the selected provider during live
generation. Run state and generated documents stay in memory for one hour or
until cleared. The only things written to disk are in `data/`: cached job
analyses (job-ad keywords, no resume content) and the run log (numbers only).
Do not expose this process to the internet without adding
authentication, durable storage and rate limits.
