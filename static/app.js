const $ = id => document.getElementById(id);
let jobId = null, pollTimer = null, liveEnabled = false, busy = false, providers = [];
const showError = message => { $('error').textContent = message; $('error').hidden = false; };
const clearError = () => { $('error').hidden = true; };

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { 'X-Resume-Studio': '1', ...(options.headers || {}) } });
  const body = await response.json();
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail :
      Array.isArray(body.detail) ? body.detail.map(e => `${e.loc.at(-1)}: ${e.msg}`).join('; ') : 'Request failed.';
    throw new Error(detail);
  }
  return body;
}

function setBusy(value) {
  busy = value;
  document.querySelectorAll('#resume-form input, #resume-form textarea, #resume-form select, #resume-form button').forEach(e => e.disabled = value);
  $('example').disabled = value;
  $('generate').textContent = value ? 'Working on your resume…' : 'Generate my resume ↗';
}

function setView(name) {
  for (const state of ['empty', 'progress', 'result']) $(state + '-state').hidden = state !== name;
}

async function upload(kind) {
  const file = $(kind + '-file').files[0];
  if (!file) return;
  clearError();
  if (file.size > 5 * 1024 * 1024) { showError('Choose a file smaller than 5 MB.'); return; }
  const form = new FormData(); form.append('file', file);
  setBusy(true);
  $(kind + '-file-status').textContent = 'Reading your file…';
  try {
    const data = await api('/api/extract', { method: 'POST', body: form });
    const target = $(kind + '-text');
    if (data.text.length > target.maxLength) throw new Error('This file has too much text. Paste a shorter version.');
    target.value = data.text;
    $('confirmed').checked = false;
    $(kind + '-file-status').textContent = `${data.filename} · Extracted. Please check the text above.`;
    target.focus();
  } catch (error) { showError(error.message); $(kind + '-file-status').textContent = 'File could not be read.'; }
  finally { setBusy(false); $(kind + '-file').value = ''; }
}

async function startRun(demo) {
  clearError();
  if (!demo && !liveEnabled) { showError('Configure the selected provider’s API key in .env and restart the server.'); return; }
  try {
    // A new run replaces the previous result and clears its server-side copy.
    if (jobId) await api(`/api/jobs/${jobId}`, { method: 'DELETE' }).catch(() => {});
    jobId = null;
    setBusy(true); setView('progress');
    $('progress-message').textContent = 'Starting your workflow'; $('progress-bar').value = 3;
    $('run-kind').textContent = demo ? 'FICTIONAL EXAMPLE · NO AI CALLS' : 'YOUR RESUME IS TAKING SHAPE';
    const result = await api(demo ? '/api/demo' : '/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: demo ? '{}' : JSON.stringify({ resume_text: $('resume-text').value.trim(),
        job_text: $('job-text').value.trim(), additional_facts: $('facts').value.trim(),
        provider: $('provider').value, model: $('model').value.trim(),
        confirmed: $('confirmed').checked, max_revisions: Number($('revisions').value), target_score: Number($('target').value) })
    });
    jobId = result.id;
    poll(jobId);
  } catch (error) { setBusy(false); setView('empty'); showError(error.message); }
}

async function poll(id) {
  if (jobId !== id) return;
  try {
    const job = await api(`/api/jobs/${id}`);
    if (jobId !== id) return;
    $('progress-message').textContent = job.message; $('progress-bar').value = job.progress;
    document.querySelectorAll('.stages li').forEach(e => e.classList.toggle('active', e.dataset.stage === job.status));
    if (['complete', 'needs_confirmation'].includes(job.status)) { setBusy(false); renderResult(job); return; }
    if (['error', 'cancelled'].includes(job.status)) { setBusy(false); setView('empty'); showError(job.message); return; }
    pollTimer = setTimeout(() => poll(id), 900);
  } catch (error) { setBusy(false); setView('empty'); showError(error.message); }
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function renderResult(job) {
  setView('result');
  const data = job.result, report = data.report, resume = data.resume;
  const pending = job.status === 'needs_confirmation';
  const unverified = new Set((data.unverified_terms || []).map(t => t.toLowerCase()));
  $('result-label').textContent = job.demo ? 'FICTIONAL EXAMPLE · ILLUSTRATIVE SCORES' : pending ? 'DRAFT · CONFIRM CONTENT BEFORE DOWNLOAD' : 'REVIEW COMPLETE';
  $('score').textContent = report.score;
  $('score-caption').textContent = 'Exact-match keyword coverage of this job. Not a hiring prediction.';
  const comparison = $('score-comparison'); comparison.replaceChildren();
  const req = report.components.required_keywords, pref = report.components.preferred_keywords;
  comparison.append(el('p', `Before: ${data.before.score}/100 (${data.before.matched} of ${data.before.total} keywords) · After: ${report.score}/100 (${req.matched + pref.matched} of ${req.total + pref.total} keywords)`));
  if (pending) comparison.append(el('p', `With every item below confirmed: ${report.score}/100 · with all of them removed: ${data.floor_score}/100.`));
  $('stop-reason').textContent = data.stop_reason + (!report.eligible ? ' An explicit eligibility rule is unmet or unknown; see the report.' : '');
  $('download').href = `/api/jobs/${job.id}/download`;
  $('download').setAttribute('download', job.demo ? 'example-resume.docx' : 'tailored-resume.docx');
  $('download').hidden = pending;
  $('confirm-section').hidden = !pending;
  $('reviewed-additions').checked = false;
  $('finalize').disabled = false;
  const proposalList = $('proposal-list'); proposalList.replaceChildren();
  for (const term of data.unverified_terms || []) {
    const label = el('label', undefined, 'proposal');
    const input = el('input'); input.type = 'checkbox'; input.value = term; input.checked = true;
    const body = el('span'); body.append(el('span', term));
    label.append(input, body); proposalList.append(label);
  }
  $('skills-title').hidden = !(data.unverified_terms || []).length;
  const bulletList = $('bullet-list'); bulletList.replaceChildren();
  const proposedIds = new Set((data.proposals || []).map(p => p.id));
  for (const proposal of data.proposals || []) {
    const label = el('label', undefined, 'proposal');
    const input = el('input'); input.type = 'checkbox'; input.value = proposal.id; input.checked = true;
    const body = el('span');
    body.append(el('strong', proposal.role_context), el('span', proposal.text));
    if (proposal.terms.length) body.append(el('small', 'Covers: ' + proposal.terms.join(', ')));
    label.append(input, body); bulletList.append(label);
  }
  $('bullets-title').hidden = !proposedIds.size;
  const preview = $('resume-preview'); preview.replaceChildren();
  preview.append(el('h2', resume.name.text));
  if (resume.headline) preview.append(el('p', resume.headline, 'headline'));
  preview.append(el('div', resume.contact.map(c => c.text).join(' | '), 'contact-line'));
  if (resume.summary.length) { preview.append(el('h3', 'Summary')); resume.summary.forEach(c => preview.append(el('p', c.text))); }
  for (const section of resume.sections) {
    preview.append(el('h3', section.heading));
    for (const entry of section.entries) {
      preview.append(el('h4', entry.heading.text));
      if (entry.detail) preview.append(el('p', entry.detail.text, 'entry-detail'));
      if (entry.bullets.length) { const list = el('ul'); entry.bullets.forEach(c => list.append(el('li', c.text, proposedIds.has(c.id) ? 'unverified' : ''))); preview.append(list); }
    }
    for (const group of section.skill_groups || []) {
      const p = el('p'); p.append(el('strong', group.label + ': '));
      group.items.forEach((item, index) => {
        const span = el('span', item.text, unverified.has(item.text.toLowerCase()) ? 'unverified' : '');
        p.append(span); if (index < group.items.length - 1) p.append(', ');
      });
      preview.append(p);
    }
  }
  const coverage = $('coverage-list'); coverage.replaceChildren();
  coverage.append(el('h3', 'Job keywords'));
  for (const item of report.keywords) {
    const box = el('div', undefined, 'coverage-item'), heading = el('div', undefined, 'coverage-heading');
    const state = !item.matched ? 'Missing' : unverified.has(item.term.toLowerCase()) ? 'Added · confirm' : 'Matched';
    heading.append(el('h4', item.term), el('span', state, 'coverage-badge ' + (!item.matched ? 'gap' : state === 'Matched' ? '' : 'partial')));
    box.append(heading, el('p', `${item.priority} · ${item.kind}${item.matched ? ` · appears ${item.count}×` : ''}`));
    coverage.append(box);
  }
  const eligibility = $('eligibility'); eligibility.replaceChildren();
  if (report.eligibility.length) {
    eligibility.append(el('h3', 'Screening rules'));
    for (const rule of report.eligibility) {
      const box = el('div', undefined, 'coverage-item'), heading = el('div', undefined, 'coverage-heading');
      heading.append(el('h4', rule.text), el('span', rule.status, 'coverage-badge ' + (rule.status === 'met' ? '' : rule.status === 'unknown' ? 'partial' : 'gap')));
      box.append(heading, el('p', rule.reason)); eligibility.append(box);
    }
  }
  const questions = $('questions'); questions.replaceChildren();
  if (report.questions.length) {
    questions.append(el('h3', 'Facts that could strengthen your resume'), el('p', 'If applicable, add your answers under additional experience and generate again.'));
    const list = el('ul'); report.questions.forEach(q => list.append(el('li', q))); questions.append(list);
  }
  const details = $('score-details'); details.replaceChildren();
  const table = el('table');
  const labels = { required_keywords: 'Required keywords', preferred_keywords: 'Preferred keywords', job_title: 'Job title', format: 'Format' };
  for (const [name, value] of Object.entries(report.components)) {
    const extra = value.total !== undefined ? ` (${value.matched}/${value.total})` : value.reason ? ` (${value.reason})` : value.missing?.length ? ` (missing: ${value.missing.join(', ')})` : '';
    const row = el('tr'); row.append(el('td', (labels[name] || name) + extra), el('td', `${value.earned} / ${value.available}`)); table.append(row);
  }
  details.append(table, el('p', `Selected draft: ${data.selected_version + 1}. Drafts: ${data.history.length}. Provider: ${job.provider || 'default'}. Model: ${job.model || 'fictional demo'}. Agent calls: ${data.usage.calls}. Tokens: ${data.usage.tokens.toLocaleString()}${data.usage.estimated_cost_usd != null ? ` (≈ $${data.usage.estimated_cost_usd.toFixed(3)})` : ''}.`),
    el('p', data.document_checks ? 'Word text extraction check passed. ' + data.document_checks.visual_check : 'Word export is waiting for your confirmation.'));
  const evidence = $('evidence-list'); evidence.replaceChildren();
  Object.values(data.evidence).forEach(text => evidence.append(el('p', text)));
  selectTab('preview');
}

$('finalize').addEventListener('click', async () => {
  clearError();
  if (!$('reviewed-additions').checked) { showError('Review the additions and confirm your selections first.'); return; }
  $('finalize').disabled = true;
  try {
    const accepted = [...$('proposal-list').querySelectorAll('input:checked')].map(e => e.value);
    const acceptedBullets = [...$('bullet-list').querySelectorAll('input:checked')].map(e => e.value);
    const job = await api(`/api/jobs/${jobId}/confirm`, {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({accepted_terms:accepted, accepted_claim_ids:acceptedBullets, reviewed_all:true})});
    renderResult(job);
  } catch (error) { showError(error.message); $('finalize').disabled = false; }
});

function selectTab(tab) {
  for (const name of ['preview', 'report']) {
    $(name + '-tab').setAttribute('aria-selected', String(name === tab));
    $(name + '-tab').tabIndex = name === tab ? 0 : -1;
    $(name + '-pane').hidden = name !== tab;
  }
}

async function clearRun() {
  clearTimeout(pollTimer);
  const id = jobId; jobId = null;
  if (id) {
    try { await api(`/api/jobs/${id}`, { method: 'DELETE' }); }
    catch (error) { showError(error.message); }
  }
  $('resume-preview').replaceChildren(); $('coverage-list').replaceChildren(); $('eligibility').replaceChildren();
  $('evidence-list').replaceChildren(); $('questions').replaceChildren(); $('score-details').replaceChildren();
  $('download').removeAttribute('href');
  setBusy(false); setView('empty');
}

$('resume-file').addEventListener('change', () => upload('resume'));
$('job-file').addEventListener('change', () => upload('job'));
for (const id of ['resume-text', 'facts']) $(id).addEventListener('input', () => { $('confirmed').checked = false; });
$('resume-form').addEventListener('submit', event => { event.preventDefault(); if (!busy) startRun(false); });
$('example').addEventListener('click', async () => {
  clearError();
  try {
    const sample = await api('/api/sample');
    $('resume-text').value = sample.resume_text; $('job-text').value = sample.job_text;
    $('facts').value = ''; $('confirmed').checked = true;
    $('resume-file-status').textContent = 'Fictional sample resume'; $('job-file-status').textContent = 'Fictional sample role';
    await startRun(true);
  } catch (error) { showError(error.message); }
});
$('cancel').addEventListener('click', clearRun); $('clear').addEventListener('click', clearRun);
for (const name of ['preview', 'report']) {
  $(name + '-tab').addEventListener('click', () => selectTab(name));
  $(name + '-tab').addEventListener('keydown', e => {
    if (['ArrowLeft', 'ArrowRight'].includes(e.key)) { e.preventDefault(); const next = name === 'preview' ? 'report' : 'preview'; selectTab(next); $(next + '-tab').focus(); }
  });
}
$('setup-toggle').addEventListener('click', () => { $('setup-help').hidden = !$('setup-help').hidden; $('setup-toggle').setAttribute('aria-expanded', String(!$('setup-help').hidden)); });
function updateProvider() {
  const selected = providers.find(p => p.id === $('provider').value);
  liveEnabled = Boolean(selected?.configured);
  $('setup-notice').hidden = liveEnabled;
  const savedModel = selected ? localStorage.getItem(`resume-studio-model-${selected.id}`) : '';
  $('model').value = savedModel || selected?.default_model || '';
  if (selected) localStorage.setItem('resume-studio-provider', selected.id);
  $('model-options').replaceChildren();
  for (const model of selected?.models || []) {
    const option = el('option'); option.value = model; $('model-options').append(option);
  }
  const destination = selected?.id === 'openrouter' ? 'OpenRouter and its selected model provider' : selected?.label || 'your selected provider';
  document.querySelector('.privacy-note').textContent = `Live generation sends this text to ${destination}. Files are processed locally; run data clears after one hour or when you clear the run.`;
}
$('provider').addEventListener('change', updateProvider);
$('model').addEventListener('input', () => { if ($('provider').value) localStorage.setItem(`resume-studio-model-${$('provider').value}`, $('model').value.trim()); });
api('/api/config').then(config => {
  providers = config.providers;
  $('provider').replaceChildren();
  for (const provider of providers) {
    const option = el('option', provider.label + (provider.configured ? '' : ' · API key missing'));
    option.value = provider.id; $('provider').append(option);
  }
  const saved = localStorage.getItem('resume-studio-provider');
  $('provider').value = providers.some(p => p.id === saved) ? saved : (providers.find(p => p.configured)?.id || providers[0].id);
  updateProvider();
}).catch(error => showError(error.message));
