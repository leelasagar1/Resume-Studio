# Resume Studio review and Luna migration

Reviewed September 19, 2026. Goal: produce relevant, readable resumes whose
claims remain supported by the candidate, with a predictable local workflow.

## What is already working well

Python owns structure, scoring and confirmation; model output cannot directly
change employers or dates. Role-scoped writing limits context leakage, revisions
target affected roles, and unsupported additions require explicit confirmation.
The existing offline suite covers parsing, auditing, confirmation and export.
Keep these boundaries when changing models.

## Prioritized improvements still recommended

1. **P1 — Strengthen claim-level grounding.** `app/workflow.py:validate_draft`
   checks numbers and job keywords against the entire resume, rather than each
   claim's cited lines and employment context. A metric copied from a different
   role can pass these deterministic checks. The model audit is the remaining
   defense. Validate numbers against cited evidence and associate each source
   line with its role; add fixtures for cross-employer metric/tool leakage.
2. **P1 — Reserve tokens across concurrent calls.** `app/agents.py:_reserve`
   checks completed usage but does not hold capacity for in-flight requests.
   Parallel role or audit calls can individually pass and collectively exceed
   `MAX_RUN_TOKENS`. Add atomic reservation/release across every provider and
   track uncertain usage after transport errors. Treat the current setting as
   a best-effort guard, not a hard spend cap.
3. **P1 — Rank drafts by confirmed evidence coverage.** `app/workflow.py:run_workflow`
   selects the highest draft score while unconfirmed proposals/skills still
   contribute. A high-scoring draft may lose much of its score after rejection.
   Compute the existing evidence-only `floor_score` for every candidate and use
   it as the primary ranking signal; display proposed coverage separately.
4. **P2 — Evaluate quality independently.** The same model writes and audits,
   and the keyword score is not a measured ATS or hiring outcome. Add a small
   human-labeled set spanning sparse resumes, career changes, long employment
   histories, adversarial document instructions and missing qualifications.
   Compare factual precision, retained achievements, readability, evidence-only
   coverage, failure rate, latency and total cost. The new fictional benchmark
   is a starting point, not proof of a model quality improvement.
5. **P2 — Reduce style repair pressure.** Fixed bullet counts, sentence lengths,
   three-sentence summaries and required keyword quotas can conflict with sparse
   evidence. Prefer a shorter truthful summary or fewer bullets when appropriate;
   tune prompt and Python checks together so they enforce the same policy.
   The prompt now explicitly makes evidence outrank style and keyword coverage.

## Implemented in this migration

- OpenAI defaults and local OpenAI model settings now target `gpt-5.6-luna`.
- Luna uses low reasoning for short extraction/writing tasks and medium for
  auditing, with additional output allowance for reasoning tokens.
- OpenAI response status and usage are inspected before local JSON validation.
  Token truncation gets one larger, budget-checked retry; refusals do not retry.
- Strict JSON schemas, evidence gates and `store=False` remain in use.
- Explicit model overrides stay supported; unpinned repair/proposal calls follow
  the selected writer. Independent writer/reviewer API overrides now work even
  when the extraction model is omitted.
- Luna has its own price estimate. Unknown GPT-5 family members no longer inherit
  the old GPT-5 price just because their names share a prefix.
- Removed a prompt example that turned training/validation into deployment.
- Added request-contract regressions and an opt-in fictional model comparison.

OpenRouter and Claude defaults are preserved. OpenRouter's Luna request uses the
Luna-specific effort instead of the legacy GPT-5 `minimal` setting. Availability
through OpenRouter has not been verified.

## Sources and validation limits

[Official Luna model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
confirms structured outputs and supported reasoning efforts.
[Reasoning guidance](https://developers.openai.com/api/docs/guides/reasoning)
explains that output budgets include reasoning tokens.

Reasoning choices and output allowances here are initial application settings,
not provider guarantees. Pricing remains an estimate using uncached token rates.

## Validation results

The complete offline suite passes (93 tests). One live run per model on the
fictional built-in example completed successfully on September 19, 2026:

| Model | App factual audit | Seconds | Calls | Tokens | Estimated USD |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-4.1 mini | pass | 13.35 | 7 | 10,003 | 0.0054 |
| GPT-5.6 Luna | pass | 17.61 | 7 | 11,626 | 0.0047 |

This single run verifies API access and end-to-end compatibility, not general
quality or latency. Luna was slightly cheaper and slower in this example.
Both draft scores reached 100, but their extracted job profiles differed:
original scores were 66.5 and 49, and evidence-only output scores were 67.5 and
50 respectively. Do not compare these percentages as a quality benchmark.
For a fair scoring comparison, use a fixed human-reviewed job profile and
independent factual labels. App audit passes also remain model judgments.
