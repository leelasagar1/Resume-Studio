import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .agents import PROVIDERS, BudgetExceeded, provider_settings
from .demo import DemoProvider, SAMPLE_JOB, SAMPLE_RESUME
from .documents import MAX_FILE, extract_text
from .models import AnalyzeRequest, GenerateRequest, ConfirmationRequest
from .review import analyze, data_dir, job_profile
from .skeleton import strip_page_furniture, strip_pasted_job
from .workflow import run_workflow

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')
JOBS = {}
TTL = 3600
ACTIVE = {'queued', 'analyzing', 'writing', 'reviewing', 'exporting'}


async def expire_jobs():
    while True:
        await asyncio.sleep(60)
        for key, job in list(JOBS.items()):
            if time.time() - job['created'] > TTL:
                job['task'].cancel()
                JOBS.pop(key, None)


@asynccontextmanager
async def lifespan(app):
    cleanup = asyncio.create_task(expire_jobs())
    yield
    cleanup.cancel()
    tasks = [j['task'] for j in JOBS.values()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(cleanup, *tasks, return_exceptions=True)
    JOBS.clear()


app = FastAPI(title='Resume Studio', lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]', 'testserver'])


class BodyLimitMiddleware:
    """Bound chunked bodies too, before multipart parsing can spool an upload."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in ('POST', 'PUT', 'PATCH'):
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunk = message.get('body', b'')
            size += len(chunk)
            if size > 6 * 1024 * 1024:
                return await JSONResponse({'detail': 'Upload limit is 5 MB.'}, status_code=413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get('more_body'):
                break
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': b''.join(chunks), 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)


app.add_middleware(BodyLimitMiddleware)


@app.middleware('http')
async def local_boundary(request: Request, call_next):
    if request.method in ('POST', 'DELETE', 'PUT', 'PATCH'):
        origin = request.headers.get('origin')
        if origin and (urlsplit(origin).netloc != request.headers.get('host') or
                       urlsplit(origin).scheme != request.url.scheme):
            return JSONResponse({'detail': 'Cross-origin requests are not allowed.'}, status_code=403)
        if request.headers.get('x-resume-studio') != '1':
            return JSONResponse({'detail': 'Missing application request header.'}, status_code=403)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    return response


@app.get('/')
async def home():
    return FileResponse(ROOT / 'static/index.html')


@app.get('/api/config')
async def config():
    try:
        provider, key_name = provider_settings()
    except ValueError:
        provider, key_name = 'invalid', ''
    providers = []
    for name, label, model_key, fallback in [
        ('openai', 'OpenAI', 'OPENAI_MODEL', 'gpt-5.6-luna'),
        ('openrouter', 'OpenRouter', 'OPENROUTER_MODEL', 'openai/gpt-4.1-mini'),
        ('claude', 'Claude (Anthropic)', 'CLAUDE_MODEL', 'claude-sonnet-5')]:
        _, key = provider_settings(name)
        default_model = os.getenv(model_key) or fallback
        providers.append({'id': name, 'label': label, 'configured': bool(os.getenv(key)),
                          'default_model': default_model,
                          'models': list(dict.fromkeys([default_model, fallback]))})
    return {'provider': provider, 'providers': providers,
            'live_enabled': bool(os.getenv(key_name)) if key_name else False,
            'max_file_mb': 5, 'retention_minutes': 60, 'max_revisions': 3, 'target_score': 90}


@app.get('/api/sample')
async def sample():
    return {'resume_text': SAMPLE_RESUME, 'job_text': SAMPLE_JOB}


@app.post('/api/extract')
async def extract(file: UploadFile = File(...)):
    try:
        data = await file.read(MAX_FILE + 1)
        text = await asyncio.to_thread(extract_text, file.filename or '', data)
        return {'text': text, 'filename': file.filename,
                'notice': 'Review extracted text, especially dates, reading order and contact details.'}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()


def public_job(job):
    return {k: v for k, v in job.items() if k not in ('task', 'docx', '_pending')}


def log_run(record):
    """One JSON line per run in data/runs.jsonl: timings, tokens, cost and
    scores. Never resume or job text, so the file is safe to share when asking
    why a run was slow or scored low."""
    try:
        path = data_dir() / 'runs.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            handle.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **record}) + '\n')
    except OSError:
        pass  # logging must never fail a run


def run_record(job, request, result):
    report = result['report']
    return {
        'event': 'generated', 'job': job['id'][:8], 'provider': job['provider'], 'model': job['model'],
        'seconds': result.get('seconds'), 'steps': result.get('timings', []), 'usage': result['usage'],
        'profile_cached': result.get('profile_cached'), 'review': result.get('review_decisions'),
        'resume_chars': len(request.resume_text), 'job_chars': len(request.job_text),
        'keywords': len(report['keywords']), 'score_before': result['before']['score'], 'score': report['score'],
        'score_without_unconfirmed': result.get('floor_score', report['score']),
        'rounds': [{'score': h['score'], 'audited': h.get('audited'), 'factual_pass': h['factual_pass']}
                   for h in result['history']],
        'proposals': len(result.get('proposals') or []), 'unconfirmed_skills': len(result.get('unverified_terms') or []),
        'eligible': report['eligible'], 'stop_reason': result['stop_reason'],
    }


def friendly_error(exc, provider):
    """A message that never exposes request contents, credentials or SDK bodies."""
    import openai
    import anthropic
    provider_name, key_name = provider.provider_name, provider.key_name
    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return f'{provider_name} rejected the server API key. Check {key_name} in .env and restart.'
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return 'The AI provider returned a rate or quota limit. Check your API account before retrying.'
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return 'Could not connect to the AI provider. Check the server connection and try again.'
    if isinstance(exc, (openai.BadRequestError, anthropic.BadRequestError)):
        return 'The model rejected this request. Check that your configured model supports Structured Outputs.'
    if isinstance(exc, (openai.NotFoundError, anthropic.NotFoundError)):
        return ('OpenRouter found no endpoint supporting the requested model and JSON Schema parameters. '
                'Select an OpenRouter model with structured-output support (for example openai/gpt-4.1-mini), '
                'then generate again.'
                if provider_name == 'OpenRouter' else
                f'{provider_name} could not find the configured model. Check the selected model and your account model access.')
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return f'{provider_name} returned HTTP {exc.status_code}. Check provider credits, model availability and request limits before retrying.'
    return 'Generation failed. Please try again.'


async def worker(key, request, provider):
    job = JOBS[key]
    started = time.time()

    def log_failure(kind):
        if not job['demo']:
            log_run({'event': 'failed', 'job': key[:8], 'provider': job['provider'], 'model': job['model'],
                     'seconds': round(time.time() - started, 1), 'error': kind,
                     'usage': provider.usage() if hasattr(provider, 'usage') else None})

    async def emit(status, message, progress):
        job.update(status=status, message=message, progress=progress)
    try:
        result = await run_workflow(request, provider, emit)
        job['docx'] = result.pop('docx')
        pending = result.pop('_pending', None)
        if pending:
            job['_pending'] = pending
        job.update(result=result, status='needs_confirmation' if pending else 'complete', progress=100,
                   message='Confirm proposed content' if pending else 'Your reviewed resume is ready')
        if not job['demo']:
            try:
                log_run(run_record(job, request, result))
            except Exception:
                pass  # a logging fault must not turn a finished run into an error
    except asyncio.CancelledError:
        job.update(status='cancelled', message='Run cancelled')
        raise
    except (ValueError, BudgetExceeded) as exc:
        job.update(status='error', message=str(exc))
        log_failure(type(exc).__name__)
    except Exception as exc:
        job.update(status='error', message=friendly_error(exc, provider))
        log_failure(type(exc).__name__)
    finally:
        await provider.close()


def make_provider(request, demo=False):
    """The provider for this request, with the request's model choices applied."""
    try:
        provider_name, key_name = provider_settings(request.provider)
    except ValueError as exc:
        if not demo:
            raise HTTPException(503, str(exc)) from exc
        provider_name, key_name = 'openai', 'OPENAI_API_KEY'
    if not demo and not os.getenv(key_name):
        raise HTTPException(503, f'Live generation needs {key_name} in the server .env file. You can try the example now.')
    provider = DemoProvider() if demo else PROVIDERS[provider_name]()
    if not demo and request.model:
        provider.model = request.model
        provider.writer_model = getattr(request, 'writer_model', None) or request.model
        provider.reviewer_model = getattr(request, 'reviewer_model', None) or request.model
    if not demo:
        provider.writer_model = getattr(request, 'writer_model', None) or provider.writer_model
        provider.reviewer_model = getattr(request, 'reviewer_model', None) or provider.reviewer_model
        cheap_key = {'openai': 'CHEAP_MODEL', 'openrouter': 'OPENROUTER_CHEAP_MODEL',
                     'claude': 'CLAUDE_CHEAP_MODEL'}[provider_name]
        if not os.getenv(cheap_key):
            provider.cheap_model = provider.writer_model
    return provider, provider_name


def start(request, demo=False):
    if sum(j['status'] in ACTIVE for j in JOBS.values()) >= 2:
        raise HTTPException(429, 'Two runs are already active. Wait for one to finish.')
    # Bound memory even when many completed runs are requested within the TTL.
    if len(JOBS) >= 30:
        raise HTTPException(429, 'Clear an earlier run or wait for its one-hour expiry.')
    provider, provider_name = make_provider(request, demo)
    key = secrets.token_urlsafe(24)
    job = {'id': key, 'status': 'queued', 'message': 'Starting your workflow',
           'progress': 3, 'created': time.time(), 'demo': demo,
           'provider': 'demo' if demo else provider_name,
           'model': None if demo else provider.model}
    if not demo:
        job['writer_model'] = provider.writer_model
        job['reviewer_model'] = provider.reviewer_model
    JOBS[key] = job
    job['task'] = asyncio.create_task(worker(key, request, provider))
    return {'id': key, 'demo': demo}


@app.post('/api/analyze')
async def analyze_job(request: AnalyzeRequest):
    """Step 1: the job's keywords against this resume, for the candidate to
    review before anything is written. The analysis is cached per posting, so
    the generate step that follows does not pay for it again."""
    provider, _ = make_provider(request)
    try:
        profile, cached = await job_profile(provider, request.job_text)
        if request.target_title.strip():
            profile.title = ' '.join(request.target_title.split())
        marked = strip_page_furniture(strip_pasted_job(request.resume_text, request.job_text))
        return {**analyze(profile, marked, request.additional_facts), 'cached': cached}
    except (ValueError, BudgetExceeded) as exc:
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, friendly_error(exc, provider)) from exc
    finally:
        await provider.close()


@app.post('/api/generate', status_code=202)
async def generate(request: GenerateRequest):
    return start(request)


@app.post('/api/demo', status_code=202)
async def demo():
    return start(GenerateRequest(resume_text=SAMPLE_RESUME, job_text=SAMPLE_JOB, confirmed=True), demo=True)


def get_job(key):
    job = JOBS.get(key)
    if job is None or time.time() - job['created'] > TTL:
        raise HTTPException(404, 'This run expired or the server restarted. Generate a new resume.')
    return job


@app.get('/api/jobs/{key}')
async def status(key: str):
    return public_job(get_job(key))


@app.get('/api/jobs/{key}/download')
async def download(key: str):
    job = get_job(key)
    if job['status'] != 'complete':
        raise HTTPException(409, 'The Word file is not ready yet.')
    filename = 'example-resume.docx' if job['demo'] else 'tailored-resume.docx'
    return Response(job['docx'], media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@app.post('/api/jobs/{key}/confirm')
async def confirm_additions(key: str, request: ConfirmationRequest):
    job = get_job(key)
    if job['status'] != 'needs_confirmation' or '_pending' not in job:
        raise HTTPException(409, 'This run is not waiting for confirmation. Refresh its status.')
    from .confirmation import finalize_pending
    try:
        # No await during validation/commit: duplicate concurrent confirmations cannot interleave.
        finalized = finalize_pending(job['_pending'], request.accepted_terms, request.accepted_claim_ids)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    job['docx'] = finalized.pop('docx')
    job['result'].update(finalized, unverified_terms=[], proposals=[],
        stop_reason='Confirmed skills were kept; unconfirmed skills were removed. Final ATS score recomputed.')
    job.pop('_pending')
    job.update(status='complete', message='Your confirmed resume is ready', progress=100)
    if not job['demo']:
        log_run({'event': 'confirmed', 'job': key[:8], 'accepted_skills': len(request.accepted_terms),
                 'accepted_bullets': len(request.accepted_claim_ids), 'score': job['result']['report']['score']})
    return public_job(job)


@app.delete('/api/jobs/{key}')
async def delete(key: str):
    job = get_job(key)
    job['task'].cancel()
    await asyncio.gather(job['task'], return_exceptions=True)
    JOBS.pop(key, None)
    return {'deleted': True}


app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
