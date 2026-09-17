'use strict';
// ── Documents module ──────────────────────────────────────────────────────────
// Depends on: state.js, api.js, chat.js

async function uploadFile(file) {
  if (!file) return;
  toast(`Uploading ${file.name}…`, 'info');
  const fd = new FormData();
  fd.append('file', file);
  try {
    const token = getToken();
    const headers = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch('/chat/upload', { method: 'POST', headers, body: fd });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    state.loadedSessions.push({
      session_id: data.session_id,
      source: data.source || file.name,
      arxiv_id: null,
      chunk_count: data.chunk_count || 0,
    });
    renderLoadedDocs();
    toast(`✓ ${file.name} loaded (${data.chunk_count} chunks)`, 'ok');
    if (chatArea.style.display !== 'none') {
      showChat();
      appendUserMessage(`📄 Uploaded: ${file.name}`);
    }
  } catch (err) {
    toast(`Upload failed: ${err.message}`, 'error');
  }
}

async function loadArxivPaper(arxivId) {
  arxivId = (arxivId || '').trim();
  if (!arxivId) return;
  toast(`Loading ${arxivId}…`, 'info');
  try {
    const data = await callApi('/chat/load-arxiv', { arxiv_id: arxivId });
    if (!state.loadedSessions.find(s => s.session_id === data.session_id)) {
      state.loadedSessions.push({
        session_id: data.session_id,
        source: data.source || arxivId,
        arxiv_id: arxivId,
        chunk_count: data.chunk_count || 0,
      });
    }
    renderLoadedDocs();
    toast(`✓ ${arxivId} loaded (${data.chunk_count} chunks)`, 'ok');
    if (arxivInput) arxivInput.value = '';
  } catch (err) {
    toast(`Could not load ${arxivId}: ${err.message}`, 'error');
  }
}

function renderLoadedDocs() {
  if (!loadedDocs) return;
  loadedDocs.innerHTML = '';
  if (!state.loadedSessions.length) return;
  for (const s of state.loadedSessions) {
    const row = document.createElement('div');
    row.className = 'doc-item';
    const label = s.arxiv_id || s.source || s.session_id.slice(0, 12);
    row.innerHTML = `
      <svg width="11" height="11" viewBox="0 0 11 11" fill="none" style="flex-shrink:0">
        <rect x="1" y="1" width="9" height="9" rx="1.5" stroke="currentColor" stroke-width="1.1"/>
        <path d="M3 4h5M3 6h3" stroke="currentColor" stroke-width="1.1" stroke-linecap="round"/>
      </svg>
      <span title="${esc(s.source || label)}">${esc(label)}</span>
      <span class="doc-chunks">${s.chunk_count}ch</span>`;
    loadedDocs.appendChild(row);
  }
}

// ── Event wiring ──────────────────────────────────────────────────────────────
if (pdfUpload) {
  pdfUpload.addEventListener('change', async () => {
    for (const f of Array.from(pdfUpload.files || [])) await uploadFile(f);
    pdfUpload.value = '';
  });
}
if (composerFile) {
  composerFile.addEventListener('change', async () => {
    for (const f of Array.from(composerFile.files || [])) await uploadFile(f);
    composerFile.value = '';
  });
}
if (composerAttach) composerAttach.addEventListener('click', () => composerFile.click());
if (loadArxivBtn)  loadArxivBtn.addEventListener('click', () => loadArxivPaper(arxivInput.value));
if (arxivInput)    arxivInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') loadArxivPaper(arxivInput.value);
});
