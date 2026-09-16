import asyncio
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
from .models import GenerateRequest, ConfirmationRequest
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
        ('openai', 'OpenAI', 'OPENAI_MODEL', 'gpt-4.1-mini'),
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


async def worker(key, request, provider):
    job = JOBS[key]

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
    except asyncio.CancelledError:
        job.update(status='cancelled', message='Run cancelled')
        raise
    except (ValueError, BudgetExceeded) as exc:
        job.update(status='error', message=str(exc))
    except Exception as exc:
        # Do not expose API request contents, credentials or SDK exception bodies.
        import openai
        import anthropic
        provider_name, key_name = provider.provider_name, provider.key_name
        message = 'Generation failed. Please try again.'
        if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
            message = f'{provider_name} rejected the server API key. Check {key_name} in .env and restart.'
        elif isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
            message = 'The AI provider returned a rate or quota limit. Check your API account before retrying.'
        elif isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
            message = 'Could not connect to the AI provider. Check the server connection and try again.'
        elif isinstance(exc, (openai.BadRequestError, anthropic.BadRequestError)):
            message = 'The model rejected this request. Check that your configured model supports Structured Outputs.'
        elif isinstance(exc, (openai.NotFoundError, anthropic.NotFoundError)):
            message = ('OpenRouter found no endpoint supporting the requested model and JSON Schema parameters. '
                       'Select an OpenRouter model with structured-output support (for example openai/gpt-4.1-mini), '
                       'then generate again.'
                       if provider_name == 'OpenRouter' else
                       f'{provider_name} could not find the configured model. Check the selected model and your account model access.')
        elif isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
            message = f'{provider_name} returned HTTP {exc.status_code}. Check provider credits, model availability and request limits before retrying.'
        job.update(status='error', message=message)
    finally:
        await provider.close()


def start(request, demo=False):
    if sum(j['status'] in ACTIVE for j in JOBS.values()) >= 2:
        raise HTTPException(429, 'Two runs are already active. Wait for one to finish.')
    # Bound memory even when many completed runs are requested within the TTL.
    if len(JOBS) >= 30:
        raise HTTPException(429, 'Clear an earlier run or wait for its one-hour expiry.')
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
        provider.writer_model = request.writer_model or request.model
        provider.reviewer_model = request.reviewer_model or request.model
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
    return public_job(job)


@app.delete('/api/jobs/{key}')
async def delete(key: str):
    job = get_job(key)
    job['task'].cancel()
    await asyncio.gather(job['task'], return_exceptions=True)
    JOBS.pop(key, None)
    return {'deleted': True}


app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
