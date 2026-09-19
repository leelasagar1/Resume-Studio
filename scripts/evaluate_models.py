"""Opt-in paid benchmark using only the built-in fictional resume and job."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from app.agents import OpenAIProvider
from app.demo import SAMPLE_JOB, SAMPLE_RESUME
from app.models import GenerateRequest
from app.workflow import run_workflow


async def evaluate(models, repeats):
    rows = []
    async def emit(*args):
        pass
    for model in models:
        for repeat in range(repeats):
            provider = OpenAIProvider()
            provider.model = provider.writer_model = provider.reviewer_model = provider.cheap_model = model
            started = time.monotonic()
            row = {'model': model, 'repeat': repeat + 1}
            try:
                result = await run_workflow(GenerateRequest(
                    resume_text=SAMPLE_RESUME, job_text=SAMPLE_JOB, confirmed=True), provider, emit)
                row.update(factual_pass=result['report']['factual_pass'],
                           original_score=result['before']['score'],
                           draft_score=result['report']['score'],
                           evidence_only_score=result.get('floor_score', result['report']['score']),
                           proposals=len(result.get('proposals', [])),
                           revisions=len(result['history']) - 1)
            except Exception as exc:
                # Never serialize provider error bodies or uploaded content.
                row.update(factual_pass=False, error_type=type(exc).__name__)
            finally:
                row.update(seconds=round(time.monotonic() - started, 2), usage=provider.usage())
                await provider.close()
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['gpt-4.1-mini', 'gpt-5.6-luna'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--live', action='store_true', help='Explicitly enable paid API calls')
    args = parser.parse_args()
    if not args.live:
        parser.error('Use --live to enable paid API calls on fictional sample data.')
    if not 1 <= args.repeats <= 10:
        parser.error('--repeats must be between 1 and 10')
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    print(json.dumps(asyncio.run(evaluate(args.models, args.repeats)), indent=2))


if __name__ == '__main__':
    main()
