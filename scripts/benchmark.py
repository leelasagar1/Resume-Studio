"""Run the whole workflow over a fixed set of fictional resumes and job ads.

    python scripts/benchmark.py --live                 # every pair once
    python scripts/benchmark.py --live --repeats 3     # measure run-to-run spread
    python scripts/benchmark.py --live --only data-analyst

Paid: about $0.02 per run with gpt-5.6-luna (9 pairs = about $0.20). Use it
before and after a change to tell an improvement from noise. Nothing here is
anyone's real data. The keyword review is skipped, so every missing skill is
left as a suggestion: this is the workflow's hardest case.

Columns: before = the resume as given; floor = tailored, no suggestion
confirmed; score = every suggestion confirmed.
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

BENCH = ROOT / 'benchmarks'


def junk_skills(resume, profile_keywords):
    """Skills-section items that are not a skill a recruiter would search:
    a job-ad filler word, or a lone verb form taken from prose."""
    from app.scoring import GENERIC_WORDS, normalize
    junk = []
    for section in resume['sections']:
        for group in section['skill_groups']:
            for item in group['items']:
                words = normalize(item['text']).split()
                if not words:
                    continue
                if all(w in GENERIC_WORDS for w in words) or (len(words) == 1 and words[0].endswith(('ized', 'ised', 'ated'))):
                    junk.append(item['text'])
    return junk


async def run_pair(pair, repeat, provider_name, model, semaphore):
    from app.agents import PROVIDERS
    from app.models import GenerateRequest
    from app.workflow import run_workflow

    async def emit(*_):
        pass
    row = {'name': pair['name'], 'repeat': repeat + 1}
    async with semaphore:
        provider = PROVIDERS[provider_name]()
        if model:
            provider.model = provider.writer_model = provider.reviewer_model = provider.cheap_model = model
        started = time.monotonic()
        try:
            request = GenerateRequest(resume_text=(BENCH / 'resumes' / pair['resume']).read_text(),
                                      job_text=(BENCH / 'jobs' / pair['job']).read_text(),
                                      target_title=pair.get('title', ''), confirmed=True)
            result = await run_workflow(request, provider, emit)
            report = result['report']
            required = report['components']['required_keywords']
            row.update(
                ok=True, seconds=round(time.monotonic() - started, 1), cost=result['usage'].get('estimated_cost_usd'),
                calls=result['usage']['calls'], rounds=len(result['history']),
                first_audit_pass=next((h['factual_pass'] for h in result['history'] if h.get('audited')), None),
                before=result['before']['score'], floor=result.get('floor_score', report['score']), score=report['score'],
                required=f"{required['matched']}/{required['total']}", keywords=len(report['keywords']),
                title=result['job']['title'], proposals=len(result.get('proposals') or []),
                empty_proposals=sum(1 for p in result.get('proposals') or [] if not p['terms']),
                unconfirmed_skills=len(result.get('unverified_terms') or []),
                junk_skills=junk_skills(result['resume'], result['job']['keywords']),
                bullets=sum(len(e['bullets']) for s in result['resume']['sections'] for e in s['entries']),
                cached=result.get('profile_cached'), steps=result.get('timings'),
                # Why a round was lost: claim ids and reviewer reasons, no resume text beyond them.
                round_issues=[[i[:140] for i in h.get('issues', [])[:4]] for h in result['history']])
        except Exception as exc:  # report the kind of failure, never provider bodies or content
            row.update(ok=False, seconds=round(time.monotonic() - started, 1), error=type(exc).__name__ + ': ' + str(exc)[:120])
        finally:
            await provider.close()
    print(f"  {row['name']} #{row['repeat']}: " + (f"{row['score']} in {row['seconds']}s" if row['ok'] else row['error']), flush=True)
    return row


def table(rows):
    head = ['pair', 'run', 'sec', 'cost', 'rounds', 'audit1', 'before', 'floor', 'score', 'required', 'kw', 'props', 'unconf', 'junk']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in rows:
        if not r['ok']:
            lines.append(f"| {r['name']} | {r['repeat']} | {r['seconds']} | FAILED: {r['error']} |" + ' |' * (len(head) - 4))
            continue
        lines.append('| ' + ' | '.join(str(x) for x in [
            r['name'], r['repeat'], r['seconds'], f"${r['cost']:.3f}" if r['cost'] is not None else '?', r['rounds'],
            'pass' if r['first_audit_pass'] else 'FAIL', r['before'], r['floor'], r['score'], r['required'], r['keywords'],
            r['proposals'], r['unconfirmed_skills'], len(r['junk_skills'])]) + ' |')
    return '\n'.join(lines)


def summary(rows, target):
    ok = [r for r in rows if r['ok']]
    if not ok:
        return 'No run succeeded.'
    fit = [r for r in ok if not r['name'].startswith('MISFIT')]
    mid = lambda key, rs=ok: statistics.median(r[key] for r in rs)
    lines = [
        f"runs: {len(ok)} ok, {len(rows) - len(ok)} failed",
        f"score (all confirmed), fitting pairs: median {mid('score', fit)}, min {min(r['score'] for r in fit)}, "
        f"{sum(r['score'] >= target for r in fit)}/{len(fit)} at {target}+",
        f"floor (nothing confirmed), fitting pairs: median {mid('floor', fit)}",
        f"time: median {mid('seconds')}s, max {max(r['seconds'] for r in ok)}s",
        f"cost: median ${statistics.median(r['cost'] or 0 for r in ok):.3f}, total ${sum(r['cost'] or 0 for r in ok):.2f}",
        f"one round: {sum(r['rounds'] == 1 for r in ok)}/{len(ok)}; first audit passed: {sum(bool(r['first_audit_pass']) for r in ok)}/{len(ok)}",
        f"junk skills: {sum(len(r['junk_skills']) for r in ok)}; proposals adding nothing: {sum(r['empty_proposals'] for r in ok)}",
    ]
    spread = {}
    for r in ok:
        spread.setdefault(r['name'], []).append(r['score'])
    wide = {name: max(v) - min(v) for name, v in spread.items() if len(v) > 1}
    if wide:
        lines.append('score spread across repeats: ' + ', '.join(f'{n} {round(d, 1)}' for n, d in wide.items())
                     + f' (largest {round(max(wide.values()), 1)})')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--live', action='store_true', help='explicitly enable paid API calls')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--only', default='', help='run only pairs whose name contains this text')
    parser.add_argument('--provider', default='openai', choices=['openai', 'openrouter', 'claude'])
    parser.add_argument('--model', default='')
    parser.add_argument('--concurrency', type=int, default=3)
    parser.add_argument('--target', type=int, default=90)
    args = parser.parse_args()
    if not args.live:
        parser.error('Use --live to enable paid API calls on the fictional benchmark set.')
    if not 1 <= args.repeats <= 5:
        parser.error('--repeats must be between 1 and 5')
    load_dotenv(ROOT / '.env')
    out_dir = Path(os.getenv('DATA_DIR') or ROOT / 'data') / 'benchmarks'
    # A fresh cache per invocation: the first run of each posting pays for its
    # analysis, repeats reuse it, exactly as the app behaves.
    os.environ['DATA_DIR'] = tempfile.mkdtemp(prefix='resume-studio-bench-')
    pairs = [p for p in json.loads((BENCH / 'pairs.json').read_text()) if args.only in p['name']]
    if not pairs:
        parser.error('No pair matches --only.')

    async def run_all():
        semaphore = asyncio.Semaphore(args.concurrency)
        rows = []
        for repeat in range(args.repeats):  # repeats run after each other so the cache behaves as in the app
            rows += await asyncio.gather(*[run_pair(p, repeat, args.provider, args.model, semaphore) for p in pairs])
        return rows
    print(f'{len(pairs)} pair(s) x {args.repeats} repeat(s) on {args.provider} {args.model or "(default model)"}')
    rows = sorted(asyncio.run(run_all()), key=lambda r: (r['name'], r['repeat']))
    report = table(rows) + '\n\n' + summary(rows, args.target)
    print('\n' + report)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    (out_dir / f'{stamp}.json').write_text(json.dumps(rows, indent=1))
    (out_dir / f'{stamp}.md').write_text(report + '\n')
    print(f'\nSaved to {out_dir / stamp}.json and .md')


if __name__ == '__main__':
    main()
