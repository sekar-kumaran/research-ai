'use strict';
/**
 * Research AI — Unified Chat Frontend
 *
 * ARCHITECTURE
 * ────────────
 * The user types a message and hits Enter. That's it. The AI orchestrator
 * on the backend automatically decides:
 *   - Whether to retrieve, classify, summarize, or chain tools
 *   - Which local/cloud model to use
 *   - How to synthesize and cite evidence
 *
 * The frontend responsibility is purely UX:
 *   1. Send query to /chat/stream (Server-Sent Events)
 *   2. Render streaming answer token-by-token
 *   3. Render source cards when the 'sources' event arrives
 *   4. Show confidence badge
 *   5. Maintain conversation history in UI
 *   6. Handle document uploads → paper chat sessions
 *
 * No mode selector. No manual tool picking. Just chat.
 */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  conversationId: null,       // UUID from server — enables multi-turn memory
  loadedSessions: [],         // [{session_id, source, arxiv_id, chunk_count}]
  topK: 5,
  debug: false,
  streaming: false,
  theme: localStorage.getItem('theme') || 'dark',
  history: [],                // [{id, title, conversationId}]
};

// ── DOM refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const welcome        = $('welcome');
const chatArea       = $('chatArea');
const chatInput      = $('chatInput');
const sendBtn        = $('sendBtn');
const historyList    = $('historyList');
const topKSlider     = $('topKSlider');
const topKVal        = $('topKVal');
const debugToggle    = $('debugToggle');
const statusDot      = $('statusDot');
const statusText     = $('statusText');
const pdfUpload      = $('pdfUpload');
const arxivInput     = $('arxivInput');
const loadArxivBtn   = $('loadArxivBtn');
const loadedDocs     = $('loadedDocs');
const modelsSection  = $('modelsSection');
const modelsList     = $('modelsList');
const themeToggle    = $('themeToggle');
const themeIcon      = $('themeIcon');
const modalOverlay   = $('modalOverlay');
const paperModal     = $('paperModal');
const modalTitle     = $('modalTitle');
const modalBody      = $('modalBody');
const modalClose     = $('modalClose');
const composerAttach = $('composerAttach');
const composerFile   = $('composerFile');
const loginOverlay   = $('loginOverlay');
const loginPassword  = $('loginPassword');
const loginBtn       = $('loginBtn');
const loginError     = $('loginError');
const exportBtn      = $('exportBtn');
const kgBtn          = $('kgBtn');
const kgOverlay      = $('kgOverlay');
const kgClose        = $('kgClose');
const kgContent      = $('kgContent');
const loginHeaderBtn = $('loginHeaderBtn');

// ── Theme ──────────────────────────────────────────────────────────────────
function applyTheme(t) {
  state.theme = t;
  document.documentElement.dataset.theme = t;
  themeIcon.textContent = t === 'dark' ? '☀' : '☾';
  localStorage.setItem('theme', t);
}
applyTheme(state.theme);
themeToggle.addEventListener('click', () =>
  applyTheme(state.theme === 'dark' ? 'light' : 'dark')
);

// ── Escape / markdown helpers ──────────────────────────────────────────────
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function mdToHtml(text) {
  let out = esc(text || '');
  // Fenced code blocks
  out = out.replace(/```[\w]*\n?([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  // Inline code
  out = out.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  // Markdown links
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  // Bare URLs
  out = out.replace(/(^|[\s>])(https?:\/\/[^\s<"&]+)/g,
    '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
  // Bold / italic
  out = out.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  // Headers
  out = out.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  out = out.replace(/^## (.+)$/gm, '<h3>$1</h3>');
  out = out.replace(/^# (.+)$/gm, '<h3>$1</h3>');
  // Lists
  const lines = out.split('\n');
  let html = '', inUl = false, inOl = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (/^[-*•]\s+/.test(line)) {
      if (inOl) { html += '</ol>'; inOl = false; }
      if (!inUl) { html += '<ul>'; inUl = true; }
      html += `<li>${line.replace(/^[-*•]\s+/, '')}</li>`;
    } else if (/^\d+\.\s+/.test(line)) {
      if (inUl) { html += '</ul>'; inUl = false; }
      if (!inOl) { html += '<ol>'; inOl = true; }
      html += `<li>${line.replace(/^\d+\.\s+/, '')}</li>`;
    } else {
      if (inUl) { html += '</ul>'; inUl = false; }
      if (inOl) { html += '</ol>'; inOl = false; }
      if (line === '') html += '<br/>';
      else if (/^<(h[1-6]|ul|ol|pre|div|blockquote)/i.test(line)) html += line;
      else html += `<p>${line}</p>`;
    }
  }
  if (inUl) html += '</ul>';
  if (inOl) html += '</ol>';
  return html;
}

function nowStr() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ── Toast ──────────────────────────────────────────────────────────────────
function toast(msg, type = 'info', dur = 4000) {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.classList.add('toast-visible'), 10);
  setTimeout(() => { el.classList.remove('toast-visible'); setTimeout(() => el.remove(), 300); }, dur);
}

// ── API helpers ─────────────────────────────────────────────────────────────
function getToken() {
  return localStorage.getItem('rai-token') || '';
}

async function callApi(endpoint, body, method = 'POST') {
  const headers = { 'Content-Type': 'application/json' };
  const token = getToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(endpoint, {
    method,
    headers,
    body: method === 'GET' ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = Array.isArray(err.detail)
        ? err.detail.map(i => i.msg || JSON.stringify(i)).join('; ')
        : String(err.detail || detail);
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

// ── Welcome / Chat visibility ──────────────────────────────────────────────
function showChat() {
  document.getElementById('main').classList.remove('welcome-active');
  welcome.style.display = 'none';
  chatArea.style.display = 'flex';
}
function showWelcome() {
  document.getElementById('main').classList.add('welcome-active');
  welcome.style.display = '';
  chatArea.style.display = 'none';
}

// ── Message construction ───────────────────────────────────────────────────

/** Build a user message bubble */
function appendUserMessage(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg user';
  wrap.innerHTML = `
    <div class="msg-avatar user-avatar">You</div>
    <div class="msg-content">
      <div class="msg-bubble user-bubble">${mdToHtml(text)}</div>
      <div class="msg-time">${nowStr()}</div>
    </div>`;
  chatArea.appendChild(wrap);
  chatArea.scrollTop = chatArea.scrollHeight;
  return wrap;
}

/** Build an AI message bubble shell (answer filled in later via streaming) */
function createAssistantShell() {
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant';
  wrap.innerHTML = `
    <div class="msg-avatar ai-avatar">AI</div>
    <div class="msg-content">
      <div class="msg-bubble ai-bubble" id="streamTarget">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
      <div class="msg-meta" style="display:none">
        <span class="msg-time"></span>
        <span class="confidence-badge" title="Evidence confidence"></span>
        <span class="intent-badge"></span>
      </div>
      <div class="sources-section" style="display:none">
        <button class="sources-toggle">
          <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
            <path d="M2 4l3.5 3.5L9 4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
          </svg>
          Sources
        </button>
        <div class="sources-list"></div>
      </div>
    </div>`;
  chatArea.appendChild(wrap);
  chatArea.scrollTop = chatArea.scrollHeight;

  const bubble    = wrap.querySelector('#streamTarget');
  bubble.removeAttribute('id');
  const meta      = wrap.querySelector('.msg-meta');
  const timeEl    = wrap.querySelector('.msg-time');
  const confBadge = wrap.querySelector('.confidence-badge');
  const intentBadge = wrap.querySelector('.intent-badge');
  const srcSection = wrap.querySelector('.sources-section');
  const srcToggle  = wrap.querySelector('.sources-toggle');
  const srcList    = wrap.querySelector('.sources-list');

  // Toggle sources accordion
  srcToggle.addEventListener('click', () => {
    const open = srcList.style.display !== 'none';
    srcList.style.display = open ? 'none' : '';
    srcToggle.classList.toggle('open', !open);
  });

  return { wrap, bubble, meta, timeEl, confBadge, intentBadge, srcSection, srcList };
}

/** Finalize an assistant bubble after streaming completes */
function finalizeAssistantBubble({ bubble, meta, timeEl, confBadge, intentBadge, srcSection, srcList }, data) {
  const { sources = [], confidence = 0, intent = '', tools_used = [] } = data;

  // Timestamp
  timeEl.textContent = nowStr();

  // Confidence badge
  const pct = Math.round(confidence * 100);
  const confClass = pct >= 70 ? 'conf-high' : pct >= 40 ? 'conf-mid' : 'conf-low';
  confBadge.textContent = `${pct}% confidence`;
  confBadge.className = `confidence-badge ${confClass}`;
  confBadge.title = `Evidence confidence: ${pct}% (tools: ${tools_used.join(', ')})`;

  // Intent badge (only shown in debug mode or for non-trivial intents)
  if (intent && intent !== 'research_analysis') {
    intentBadge.textContent = intent.replace(/_/g, ' ');
    intentBadge.className = 'intent-badge';
  }

  meta.style.display = '';

  // Sources
  if (sources.length) {
    renderSources(srcList, sources);
    const label = `Sources (${sources.length})`;
    srcSection.querySelector('.sources-toggle').childNodes[1].textContent = ' ' + label;
    srcSection.style.display = '';
    // Auto-expand sources if 3 or fewer
    if (sources.length <= 3) {
      srcList.style.display = '';
      srcSection.querySelector('.sources-toggle').classList.add('open');
    }
  }
}

/** Instantly append a completed assistant message (used for history) */
function appendAssistantMessage(text) {
  const shell = createAssistantShell();
  shell.bubble.innerHTML = mdToHtml(text);
  // We don't have historical sources/confidence in the current /conversations endpoint format
  // so we pass empty data to hide those sections cleanly.
  finalizeAssistantBubble(shell, { sources: [], confidence: 1.0, intent: '' });
}

/** Render source cards into the sources list element */
function renderSources(srcList, sources) {
  srcList.innerHTML = '';
  if (!sources || !sources.length) return;
  sources.forEach((src, i) => {
    const card = document.createElement('div');
    card.className = 'source-card';
    const pid = src.paper_id || '';
    const url = src.arxiv_url || (pid ? `https://arxiv.org/abs/${pid}` : '');
    const score = src.score ? `${(src.score * 100).toFixed(0)}%` : '';
    card.innerHTML = `
      <div class="source-num">[${i + 1}]</div>
      <div class="source-body">
        <div class="source-title">${esc(src.title || 'Untitled')}</div>
        <div class="source-meta">
          ${src.year ? `<span class="source-tag">${esc(src.year)}</span>` : ''}
          ${src.category ? `<span class="source-tag">${esc(src.category)}</span>` : ''}
          ${score ? `<span class="source-tag source-score">${score}</span>` : ''}
        </div>
        ${src.abstract_snippet
          ? `<div class="source-snippet">${esc(src.abstract_snippet)}</div>`
          : ''}
        <div class="source-actions">
          ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer" class="source-link">arXiv ↗</a>` : ''}
          ${pid ? `<button class="source-link chat-btn" data-arxiv="${esc(pid)}">Chat with paper</button>` : ''}
        </div>
      </div>`;

    // Wire "Chat with paper" to load the paper into session
    const chatBtn = card.querySelector('.chat-btn');
    if (chatBtn) {
      chatBtn.addEventListener('click', () => loadArxivPaper(chatBtn.dataset.arxiv));
    }
    srcList.appendChild(card);
  });
}

// ── Core send flow ──────────────────────────────────────────────────────────

async function sendMessage(query) {
  if (state.streaming) return;
  query = (query || '').trim();
  if (!query) return;

  // Transition to chat view
  showChat();
  appendUserMessage(query);
  chatInput.value = '';
  autoGrow();
  sendBtn.disabled = true;
  state.streaming = true;

  const shell = createAssistantShell();
  chatArea.scrollTop = chatArea.scrollHeight;

  // Session IDs for paper-chat context
  const sessionId = state.loadedSessions.length
    ? state.loadedSessions.map(s => s.session_id).join(',')
    : null;

  let accumulated = '';
  let streamDone = false;

  // Inactivity timeout: reset when any SSE data arrives (including keepalive).
  const timeoutMs = 120_000;
  let timeoutHandle = null;

  function armTimeout() {
    if (timeoutHandle) clearTimeout(timeoutHandle);
    timeoutHandle = setTimeout(() => {
      if (!streamDone) {
        if (!accumulated) {
          shell.bubble.innerHTML = '<em>Response timed out. Please try again.</em>';
        }
        finishStreaming();
      }
    }, timeoutMs);
  }

  armTimeout();

  const msgSteps = document.createElement('div');
  msgSteps.className = 'msg-steps';
  shell.bubble.parentNode.insertBefore(msgSteps, shell.bubble);
  
  const stepMessages = [
    "Analyzing question intent...",
    "Selecting ML models...",
    "Retrieving vector embeddings...",
    "Extracting paper context...",
    "Processing with LLM..."
  ];
  let currentStepIdx = 0;
  
  function advanceStep() {
    if (accumulated || currentStepIdx >= stepMessages.length) {
      if (msgSteps.parentNode) msgSteps.style.display = 'none';
      return;
    }
    // Mark previous as completed
    const prev = msgSteps.querySelector('.msg-step.active');
    if (prev) {
      prev.classList.remove('active');
      prev.classList.add('completed');
    }
    // Add new step
    const stepEl = document.createElement('div');
    stepEl.className = 'msg-step active';
    stepEl.innerHTML = `<div class="step-icon"></div> <span>${stepMessages[currentStepIdx]}</span>`;
    msgSteps.appendChild(stepEl);
    currentStepIdx++;
    chatArea.scrollTop = chatArea.scrollHeight;
  }
  
  advanceStep(); // Initial step

  function finishStreaming() {
    streamDone = true;
    if (timeoutHandle) clearTimeout(timeoutHandle);
    state.streaming = false;
    sendBtn.disabled = false;
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  try {
    const headers = { 'Content-Type': 'application/json' };
    const token = getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        query,
        conversation_id: state.conversationId,
        session_id: sessionId || undefined,
        top_k: state.topK,
        debug: state.debug,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let pendingData = {};   // accumulates sources, confidence, etc. from events

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      armTimeout();
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop() || '';

      for (const part of parts) {
        armTimeout();
        const rawLine = part.trim();
        const line = rawLine.replace(/^data:\s*/, '').trim();
        if (line === '[DONE]') { break; }
        // Backend sends ": keepalive" (SSE comment, no data: prefix)
        if (rawLine === ': keepalive' || rawLine === ':keepalive') { advanceStep(); continue; }
        if (!line) continue;
        try {
          const obj = JSON.parse(line);

          if (obj.delta !== undefined) {
            // Streaming text delta
            if (!accumulated) {
               shell.bubble.innerHTML = '';  // clear typing dots
               if (msgSteps.parentNode) msgSteps.style.display = 'none'; // Hide steps
            }
            accumulated += obj.delta;
            shell.bubble.innerHTML = mdToHtml(accumulated);
            chatArea.scrollTop = chatArea.scrollHeight;

          } else if (obj.event === 'start') {
            // Record conversation_id for multi-turn continuity
            if (obj.conversation_id) state.conversationId = obj.conversation_id;

          } else if (obj.event === 'sources') {
            pendingData.sources = obj.sources || [];

          } else if (obj.event === 'done') {
            if (obj.conversation_id) state.conversationId = obj.conversation_id;
            pendingData.confidence = obj.confidence || 0;
            pendingData.latency_ms = obj.latency_ms || 0;

          } else if (obj.event === 'error') {
            shell.bubble.innerHTML = `<span class="error-text">Error: ${esc(obj.message || 'Unknown error')}</span>`;
          }
        } catch (_) { /* ignore malformed SSE frame */ }
      }
    }

    // If no text was streamed (very short response), ensure something shows
    if (!accumulated && !shell.bubble.querySelector('.error-text')) {
      shell.bubble.innerHTML = '<em>No response received.</em>';
    }

    // Finalize the bubble with sources, confidence, etc.
    finalizeAssistantBubble(shell, pendingData);

    // Save to history
    addToHistory(query);

  } catch (err) {
    shell.bubble.innerHTML = `<span class="error-text">${esc(err.message)}</span>`;
    shell.meta.style.display = '';
    shell.timeEl.textContent = nowStr();
  } finally {
    finishStreaming();
  }
}

// ── Composer ────────────────────────────────────────────────────────────────
function autoGrow() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
}

chatInput.addEventListener('input', () => {
  autoGrow();
  sendBtn.disabled = !chatInput.value.trim() || state.streaming;
});

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(chatInput.value);
  }
});

sendBtn.addEventListener('click', () => sendMessage(chatInput.value));

// ── Settings controls ───────────────────────────────────────────────────────
topKSlider.addEventListener('input', () => {
  state.topK = parseInt(topKSlider.value, 10);
  topKVal.textContent = state.topK;
});
debugToggle.addEventListener('change', () => {
  state.debug = debugToggle.checked;
});

// ── Example chips ───────────────────────────────────────────────────────────
document.querySelectorAll('.example-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    chatInput.value = chip.textContent.trim();
    autoGrow();
    sendBtn.disabled = false;
    sendMessage(chatInput.value);
  });
});

// ── New chat ────────────────────────────────────────────────────────────────
function startNewChat() {
  state.conversationId = null;
  chatArea.innerHTML = '';
  chatInput.value = '';
  autoGrow();
  sendBtn.disabled = true;
  showWelcome();
}
if ($('newChatBtn')) $('newChatBtn').addEventListener('click', startNewChat);
if ($('topbarNewChat')) $('topbarNewChat').addEventListener('click', startNewChat);

// ── History ─────────────────────────────────────────────────────────────────
function addToHistory(title) {
  const entry = {
    id: Date.now().toString(),
    title: title.slice(0, 60),
    conversationId: state.conversationId,
  };
  state.history.unshift(entry);
  if (state.history.length > 50) state.history.pop();
  renderHistory();
  try { localStorage.setItem('rai-history', JSON.stringify(state.history.slice(0, 30))); } catch (_) {}
}

function renderHistory() {
  historyList.innerHTML = '';
  if (!state.history.length) {
    historyList.innerHTML = '<div class="history-empty">No conversations yet</div>';
    return;
  }
  for (const item of state.history) {
    const btn = document.createElement('button');
    btn.className = 'history-item';
    btn.title = item.title;
    btn.innerHTML = `
      <svg width="11" height="11" viewBox="0 0 11 11" fill="none" style="flex-shrink:0">
        <path d="M5.5 1a4.5 4.5 0 1 1 0 9 4.5 4.5 0 0 1 0-9zM5.5 3v3l2 1" stroke="currentColor" stroke-width="1.1" stroke-linecap="round"/>
      </svg>
      <span>${esc(item.title)}</span>`;
    btn.addEventListener('click', async () => {
      await loadConversation(item.conversationId, item.title);
    });
    historyList.appendChild(btn);
  }
}

async function loadConversation(id, title) {
  try {
    toast(`Loading conversation: "${title.slice(0, 40)}"`, 'info');
    const data = await callApi(`/conversations/${id}`, {}, 'GET');
    
    // Clear chat area and set state
    state.conversationId = id;
    chatArea.innerHTML = '';
    showChat();
    
    // Render turns
    if (data.turns && data.turns.length > 0) {
      for (const turn of data.turns) {
        if (turn.role === 'user') {
          appendUserMessage(turn.content);
        } else if (turn.role === 'assistant') {
          appendAssistantMessage(turn.content);
        }
      }
    } else {
      chatArea.innerHTML = '<div class="history-empty">Conversation is empty.</div>';
    }
    chatArea.scrollTop = chatArea.scrollHeight;
    chatInput.focus();
  } catch (err) {
    toast(`Failed to load conversation: ${err.message}`, 'error');
  }
}

function loadHistory() {
  try {
    const raw = localStorage.getItem('rai-history');
    if (raw) state.history = JSON.parse(raw);
    renderHistory();
  } catch (_) {}
}

// ── Document upload ──────────────────────────────────────────────────────────
async function uploadFile(file) {
  if (!file) return;
  toast(`Uploading ${file.name}…`, 'info');
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/chat/upload', { method: 'POST', body: fd });
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${res.status}`); }
    const data = await res.json();
    state.loadedSessions.push({ session_id: data.session_id, source: data.source || file.name, arxiv_id: null, chunk_count: data.chunk_count || 0 });
    renderLoadedDocs();
    toast(`✓ ${file.name} loaded (${data.chunk_count} chunks)`, 'ok');

    // Show a system message in chat if chat is already open
    if (chatArea.style.display !== 'none') {
      showChat();
      const msg = appendUserMessage(`📄 Uploaded: ${file.name}`);
    }
  } catch (err) { toast(`Upload failed: ${err.message}`, 'error'); }
}

async function loadArxivPaper(arxivId) {
  arxivId = (arxivId || '').trim();
  if (!arxivId) return;
  toast(`Loading ${arxivId}…`, 'info');
  try {
    const data = await callApi('/chat/load-arxiv', { arxiv_id: arxivId });
    if (!state.loadedSessions.find(s => s.session_id === data.session_id)) {
      state.loadedSessions.push({ session_id: data.session_id, source: data.source || arxivId, arxiv_id: arxivId, chunk_count: data.chunk_count || 0 });
    }
    renderLoadedDocs();
    toast(`✓ ${arxivId} loaded (${data.chunk_count} chunks)`, 'ok');
    arxivInput.value = '';
  } catch (err) { toast(`Could not load ${arxivId}: ${err.message}`, 'error'); }
}

function renderLoadedDocs() {
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

// Wire upload triggers
pdfUpload.addEventListener('change', async () => {
  for (const f of Array.from(pdfUpload.files || [])) await uploadFile(f);
  pdfUpload.value = '';
});
composerFile.addEventListener('change', async () => {
  for (const f of Array.from(composerFile.files || [])) await uploadFile(f);
  composerFile.value = '';
});
composerAttach.addEventListener('click', () => composerFile.click());
loadArxivBtn.addEventListener('click', () => loadArxivPaper(arxivInput.value));
arxivInput.addEventListener('keydown', e => { if (e.key === 'Enter') loadArxivPaper(arxivInput.value); });

// ── Ollama model list ────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const data = await callApi('/models/list', {}, 'GET').catch(() => null);
    if (!data || !data.available || !data.models.length) return;
    modelsSection.style.display = '';
    modelsList.innerHTML = data.models.map(m => `
      <div class="model-item">
        <span class="model-name">${esc(m.name)}</span>
        <span class="model-tier tier-${m.tier}">${esc(m.tier_label)}</span>
        ${m.size_gb ? `<span class="model-size">${m.size_gb}GB</span>` : ''}
      </div>`).join('');
  } catch (_) { /* Ollama offline — hide section */ }
}

// ── Health check ────────────────────────────────────────────────────────────
async function checkHealth() {
  statusDot.className = 'status-dot loading';
  statusText.textContent = 'Connecting…';
  try {
    const data = await fetch('/health').then(r => r.json());
    const c = data.components || {};
    const ready = c.hybrid_retrieval || c.classifier || c.paper_chat;
    statusDot.className = `status-dot ${ready ? 'ok' : 'warn'}`;
    const parts = [];
    if (c.hybrid_retrieval) parts.push('Search');
    if (c.classifier) parts.push('Classify');
    if (c.summarizer) parts.push('Summarize');
    if (c.paper_chat) parts.push('Chat');
    statusText.textContent = parts.length ? parts.join(' · ') : data.version || 'Online';
  } catch (_) {
    statusDot.className = 'status-dot err';
    statusText.textContent = 'API offline';
  }
}

// ── Mobile sidebar ───────────────────────────────────────────────────────────
const sidebar   = document.querySelector('.sidebar');
const sidebarOv = $('sidebarOverlay');

function openSidebar() { sidebar.classList.add('open'); sidebarOv.classList.add('visible'); }
function closeSidebar() { sidebar.classList.remove('open'); sidebarOv.classList.remove('visible'); }

const mobileSidebarToggle = $('mobileSidebarToggle');
if (mobileSidebarToggle) {
  mobileSidebarToggle.addEventListener('click', () =>
    sidebar.classList.contains('open') ? closeSidebar() : openSidebar()
  );
}
if (sidebarOv) sidebarOv.addEventListener('click', closeSidebar);

// ── Paper modal ─────────────────────────────────────────────────────────────
if (modalClose) modalClose.addEventListener('click', () => { modalOverlay.style.display = 'none'; });
if (modalOverlay) modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) modalOverlay.style.display = 'none'; });

// ── Topbar Actions ──────────────────────────────────────────────────────────
if (exportBtn) {
  exportBtn.addEventListener('click', () => {
    if (!state.conversationId) return toast('No active conversation to export.', 'info');
    toast('Preparing export...', 'info');
    callApi(`/conversations/${state.conversationId}`, {}, 'GET').then(data => {
      let md = `# Research AI Export\nDate: ${new Date().toLocaleString()}\n\n---\n\n`;
      if (data.turns) {
        data.turns.forEach(turn => {
          md += `### ${turn.role === 'user' ? 'User' : 'Research AI'}\n\n`;
          md += `${turn.content}\n\n---\n\n`;
        });
      }
      const blob = new Blob([md], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `research_ai_${state.conversationId.slice(0, 8)}.md`;
      a.click();
      URL.revokeObjectURL(url);
      toast('Export complete!', 'ok');
    }).catch(e => toast('Failed to export conversation.', 'error'));
  });
}

if (kgBtn) {
  kgBtn.addEventListener('click', async () => {
    kgOverlay.style.display = 'flex';
    kgContent.innerHTML = 'Loading knowledge graph...';
    try {
      const data = await callApi('/knowledge-graph', {}, 'GET');
      if (data.concepts && Object.keys(data.concepts).length > 0) {
        let html = '<div style="display:flex; flex-wrap:wrap; gap:8px; padding-top:4px;">';
        // Sort by count descending
        const sorted = Object.entries(data.concepts).sort((a, b) => b[1] - a[1]);
        for (const [concept, count] of sorted) {
          html += `<span style="padding:6px 12px; background:var(--bg-3); border:1px solid var(--border); border-radius:16px; font-size:13px; color:var(--text); box-shadow:var(--shadow-sm); cursor:default; transition:var(--tx);" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--border)'">
                     <strong>${concept}</strong> <span style="color:var(--text-3); font-size:11px; margin-left:4px;">${count}</span>
                   </span>`;
        }
        html += '</div>';
        kgContent.innerHTML = html;
      } else {
        kgContent.innerHTML = '<div style="text-align:center; padding:20px; color:var(--text-3);">No concepts extracted yet. Start chatting about papers to build your knowledge graph!</div>';
      }
    } catch (e) {
      kgContent.innerHTML = '<div style="color:var(--conf-low);">Failed to load knowledge graph.</div>';
    }
  });
}
if (kgClose) { kgClose.addEventListener('click', () => { kgOverlay.style.display = 'none'; }); }
kgOverlay.addEventListener('click', e => { if (e.target === kgOverlay) kgOverlay.style.display = 'none'; });

// ── Auth ────────────────────────────────────────────────────────────────────
if (loginHeaderBtn) {
  loginHeaderBtn.addEventListener('click', () => {
    loginOverlay.style.display = 'flex';
    loginPassword.focus();
  });
}

async function checkAuth() {
  const token = getToken();
  // If no token stored and APP_PASSWORD is set, we'll know from 401 on health check
  // Try /health first (unprotected) to see if auth is even required
  try {
    const healthRes = await fetch('/health');
    if (healthRes.ok) {
      const data = await healthRes.json();
      // If server is up without auth, no login needed
      if (!token) {
        // Try a protected endpoint to check if password is required
        const authCheck = await fetch('/login', {
          method: 'POST',
          headers: {}
        });
        if (authCheck.status === 401) {
          // Password required but none stored — show login
          loginOverlay.style.display = 'flex';
          setTimeout(() => loginPassword.focus(), 100);
          return false;
        }
        return true;
      }
      // Token is stored — validate it
      const authCheck = await fetch('/login', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (authCheck.status === 401) {
        localStorage.removeItem('rai-token');
        loginOverlay.style.display = 'flex';
        setTimeout(() => loginPassword.focus(), 100);
        return false;
      }
      return true;
    }
  } catch (e) {
    // Network error — allow usage, auth will fail on actual API calls
    return true;
  }
  return true;
}

loginBtn.addEventListener('click', async () => {
  const pwd = loginPassword.value.trim();
  if (!pwd) return;
  loginBtn.textContent = 'Verifying...';
  loginBtn.disabled = true;
  loginError.style.display = 'none';
  try {
    const res = await fetch('/login', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${pwd}` }
    });
    if (res.ok) {
      localStorage.setItem('rai-token', pwd);
      loginOverlay.style.display = 'none';
      checkHealth();
      loadModels();
    } else {
      let msg = 'Invalid password';
      try { const err = await res.json(); if (err.detail) msg = err.detail; } catch (e) {}
      loginError.textContent = msg;
      loginError.style.display = 'block';
    }
  } catch (e) {
    loginError.style.display = 'block';
    loginError.textContent = 'Connection error';
  } finally {
    loginBtn.textContent = 'Login';
    loginBtn.disabled = false;
  }
});

loginPassword.addEventListener('keydown', e => {
  if (e.key === 'Enter') loginBtn.click();
});

// ── Init ────────────────────────────────────────────────────────────────────
async function init() {
  loadHistory();
  showWelcome();
  const authed = await checkAuth();
  if (authed) {
    checkHealth();
    loadModels();
  }
  setInterval(checkHealth, 60_000);
}
init();
