'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  mode: 'ask',
  topK: 5,
  sessionId: null,      // active paper-chat session
  sessionSource: null,
  messages: [],         // [{role, content, raw}]
  history: [],          // [{id, title, messages, mode}]
  streaming: false,
};

// ── DOM refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const chatBox         = $('chatBox');
const hero            = $('hero');
const chatInput       = $('chatInput');
const sendBtn         = $('sendBtn');
const newChatBtn      = $('newChatBtn');
const historyList     = $('historyList');
const modePills       = $('modePills');
const topKSlider      = $('topKSlider');
const topKVal         = $('topKVal');
const statusDot       = $('statusDot');
const statusText      = $('statusText');
const composerPrefix  = $('composerPrefix');
const topbarMode      = $('topbarMode');
const chatLoaderSection = $('chatLoaderSection');
const arxivIdInput    = $('arxivIdInput');
const loadArxivBtn    = $('loadArxivBtn');
const pdfUpload       = $('pdfUpload');
const sessionStatus   = $('sessionStatus');
const mobileSidebarToggle = $('mobileSidebarToggle');
const sidebar         = document.querySelector('.sidebar');
const quickChips      = document.querySelectorAll('.chip');

// ── Mode config ─────────────────────────────────────────────────────────────
const MODE_CONFIG = {
  ask:       { label: 'Ask',       prefix: 'Ask',      placeholder: 'Ask a research question…', endpoint: '/agent/run/stream' },
  search:    { label: 'Search',    prefix: 'Search',   placeholder: 'Search for papers on…', endpoint: '/search' },
  classify:  { label: 'Classify',  prefix: 'Classify', placeholder: 'Enter title + abstract to classify…', endpoint: '/classify' },
  summarize: { label: 'Summarize', prefix: 'Sum',      placeholder: 'Paste abstract or text to summarize…', endpoint: '/summarize' },
  chat:      { label: 'Chat',      prefix: 'Chat',     placeholder: 'Ask about the loaded paper…', endpoint: '/chat/ask' },
};

// ── Utilities ──────────────────────────────────────────────────────────────
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function markdownToHtml(text) {
  let out = escapeHtml(text || '');
  // Code blocks
  out = out.replace(/```[\w]*\n?([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  // Inline code
  out = out.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  // Links
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  // Raw URLs → links
  out = out.replace(/(?<![">])(https?:\/\/[^\s<]+)/g,
    '<a href="$1" target="_blank" rel="noopener">$1</a>');
  // Bold & italic
  out = out.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Headers
  out = out.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  out = out.replace(/^## (.+)$/gm, '<h3>$1</h3>');
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
      if (line === '') {
        html += '<br/>';
      } else if (!line.startsWith('<')) {
        html += `<p>${line}</p>`;
      } else {
        html += line;
      }
    }
  }
  if (inUl) html += '</ul>';
  if (inOl) html += '</ol>';
  return html;
}

function now() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function showToast(msg, duration = 4000) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

// ── Mode switching ──────────────────────────────────────────────────────────
function setMode(mode) {
  state.mode = mode;
  const cfg = MODE_CONFIG[mode] || MODE_CONFIG.ask;
  composerPrefix.textContent = cfg.prefix;
  chatInput.placeholder = cfg.placeholder;
  topbarMode.textContent = cfg.label;

  document.querySelectorAll('.mode-pill').forEach(p => {
    p.classList.toggle('active', p.dataset.mode === mode);
  });

  chatLoaderSection.style.display = mode === 'chat' ? 'flex' : 'none';
}

modePills.addEventListener('click', e => {
  const pill = e.target.closest('.mode-pill');
  if (pill) setMode(pill.dataset.mode);
});

// ── TopK ───────────────────────────────────────────────────────────────────
topKSlider.addEventListener('input', () => {
  state.topK = parseInt(topKSlider.value, 10);
  topKVal.textContent = state.topK;
});

// ── Hero / Chat visibility ──────────────────────────────────────────────────
function showChat() {
  hero.classList.add('hidden');
  chatBox.style.display = 'flex';
}

function showHero() {
  hero.classList.remove('hidden');
  chatBox.style.display = 'none';
  chatBox.innerHTML = '';
  state.messages = [];
  state.sessionId = null;
  state.sessionSource = null;
  sessionStatus.textContent = '';
  sessionStatus.className = 'session-status';
}

// ── Render messages ─────────────────────────────────────────────────────────
function appendMessage(role, content, raw = null) {
  state.messages.push({ role, content, raw });
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  const avatarEl = document.createElement('div');
  avatarEl.className = 'msg-avatar';
  avatarEl.textContent = role === 'user' ? 'U' : 'AI';

  const bodyEl = document.createElement('div');
  bodyEl.className = 'msg-body';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = markdownToHtml(content);

  // Paper cards for search results
  if (raw && Array.isArray(raw.results) && raw.results.length > 0 && state.mode === 'search') {
    bubble.innerHTML = `<p><strong>${raw.results.length} papers found</strong></p>`;
    const cards = document.createElement('div');
    cards.className = 'paper-cards';
    for (const doc of raw.results.slice(0, 8)) {
      cards.appendChild(buildPaperCard(doc));
    }
    bubble.appendChild(cards);
  }

  // Classify result with confidence bars
  if (raw && raw.predicted_category && state.mode === 'classify') {
    bubble.innerHTML = buildClassifyHtml(raw);
  }

  const timeEl = document.createElement('div');
  timeEl.className = 'msg-time';
  timeEl.textContent = now();

  bodyEl.appendChild(bubble);
  bodyEl.appendChild(timeEl);
  wrap.appendChild(avatarEl);
  wrap.appendChild(bodyEl);
  chatBox.appendChild(wrap);
  chatBox.scrollTop = chatBox.scrollHeight;
  return bubble;
}

function buildPaperCard(doc) {
  const card = document.createElement('div');
  card.className = 'paper-card';
  const pid = doc.paper_id || '';
  const url = pid ? `https://arxiv.org/abs/${pid}` : '';
  const score = typeof doc.score === 'number' ? doc.score.toFixed(3) : '';
  const abstract = (doc.abstract || '').slice(0, 260) + (doc.abstract?.length > 260 ? '…' : '');
  card.innerHTML = `
    <div class="paper-card-header">
      <div class="paper-title">${escapeHtml(doc.title || 'Untitled')}</div>
      ${score ? `<div class="paper-score">${score}</div>` : ''}
    </div>
    <div class="paper-meta">
      ${doc.year ? `<span class="paper-tag">${escapeHtml(doc.year)}</span>` : ''}
      ${doc.category ? `<span class="paper-tag">${escapeHtml(doc.category)}</span>` : ''}
      ${pid ? `<span class="paper-tag">${escapeHtml(pid)}</span>` : ''}
    </div>
    ${abstract ? `<div class="paper-abstract">${escapeHtml(abstract)}</div>` : ''}
    <div class="paper-actions">
      ${url ? `<a href="${url}" target="_blank" rel="noopener" class="paper-action-btn">Open arXiv ↗</a>` : ''}
      <button class="paper-action-btn" onclick="loadPaperChat('${escapeHtml(pid)}')">Chat with paper</button>
      <button class="paper-action-btn" onclick="summarisePaper('${escapeHtml(doc.abstract || '')}')">Summarise</button>
    </div>
  `;
  return card;
}

function buildClassifyHtml(raw) {
  const cat = raw.predicted_category || 'Unknown';
  const conf = raw.confidence || {};
  const entries = Object.entries(conf).slice(0, 5);
  const bars = entries.map(([label, pct]) => `
    <div class="conf-row">
      <span class="conf-label">${escapeHtml(label)}</span>
      <div class="conf-track"><div class="conf-fill" style="width:${(pct*100).toFixed(1)}%"></div></div>
      <span class="conf-pct">${(pct*100).toFixed(1)}%</span>
    </div>
  `).join('');
  return `
    <div class="classify-result">
      <div class="classify-category">${escapeHtml(cat)}</div>
      ${bars ? `<div class="confidence-bar">${bars}</div>` : ''}
    </div>
  `;
}

// ── Typing indicator ─────────────────────────────────────────────────────────
function addTypingIndicator() {
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant';
  wrap.id = 'typingWrap';
  const avatarEl = document.createElement('div');
  avatarEl.className = 'msg-avatar';
  avatarEl.textContent = 'AI';
  const ind = document.createElement('div');
  ind.className = 'typing-indicator';
  ind.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
  const bodyEl = document.createElement('div');
  bodyEl.className = 'msg-body';
  bodyEl.appendChild(ind);
  wrap.appendChild(avatarEl);
  wrap.appendChild(bodyEl);
  chatBox.appendChild(wrap);
  chatBox.scrollTop = chatBox.scrollHeight;
  return { wrap, bodyEl };
}

function removeTypingIndicator() {
  const el = $('typingWrap');
  if (el) el.remove();
}

// ── API calls ───────────────────────────────────────────────────────────────
async function callApi(endpoint, body, method = 'POST') {
  const res = await fetch(endpoint, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function streamAgent(body, onDelta, onDone) {
  const res = await fetch('/agent/run/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split('\n\n');
    buf = parts.pop();
    for (const part of parts) {
      const line = part.replace(/^data: /, '').trim();
      if (line === '[DONE]') { onDone(); return; }
      if (!line) continue;
      try {
        const obj = JSON.parse(line);
        if (obj.delta) onDelta(obj.delta);
      } catch {}
    }
  }
  onDone();
}

// ── Send message ─────────────────────────────────────────────────────────────
async function sendMessage(query) {
  if (state.streaming) return;
  query = query.trim();
  if (!query) return;

  showChat();
  appendMessage('user', query);
  chatInput.value = '';
  autoGrow();
  sendBtn.disabled = true;
  state.streaming = true;

  const { wrap: typingWrap, bodyEl: typingBody } = addTypingIndicator();

  try {
    const mode = state.mode;

    if (mode === 'search') {
      const data = await callApi('/search', { query, top_k: state.topK });
      removeTypingIndicator();
      const text = data.results?.length
        ? `Found **${data.count}** papers for: *${query}*`
        : 'No results found for your query.';
      appendMessage('assistant', text, data);

    } else if (mode === 'classify') {
      // Parse title / abstract from input
      const parts = query.split(/\n+/);
      const title = parts[0] || query;
      const abstract = parts.slice(1).join(' ') || query;
      const data = await callApi('/classify', { title, abstract });
      removeTypingIndicator();
      appendMessage('assistant', `Predicted: **${data.predicted_category}**`, data);

    } else if (mode === 'summarize') {
      const data = await callApi('/summarize', { text: query });
      removeTypingIndicator();
      appendMessage('assistant', data.summary || 'No summary generated.');

    } else if (mode === 'chat') {
      if (!state.sessionId) {
        removeTypingIndicator();
        showToast('Load a paper first using the sidebar panel.');
        return;
      }
      const data = await callApi('/chat/ask', {
        session_id: state.sessionId,
        question: query,
        top_k: state.topK,
      });
      removeTypingIndicator();
      appendMessage('assistant', data.answer || 'No answer generated.', data);

    } else {
      // ask mode with streaming
      let accumulated = '';
      let bubble = null;

      typingWrap.id = '';
      removeTypingIndicator();

      // Create a streaming bubble
      const msgWrap = document.createElement('div');
      msgWrap.className = 'msg assistant';
      const avatarEl2 = document.createElement('div');
      avatarEl2.className = 'msg-avatar';
      avatarEl2.textContent = 'AI';
      const bodyEl2 = document.createElement('div');
      bodyEl2.className = 'msg-body';
      bubble = document.createElement('div');
      bubble.className = 'msg-bubble';
      bubble.innerHTML = '<span class="typing-dot"></span>';
      bodyEl2.appendChild(bubble);
      msgWrap.appendChild(avatarEl2);
      msgWrap.appendChild(bodyEl2);
      chatBox.appendChild(msgWrap);
      chatBox.scrollTop = chatBox.scrollHeight;

      await streamAgent(
        { mode: 'ask', query, top_k: state.topK },
        delta => {
          accumulated += delta;
          bubble.innerHTML = markdownToHtml(accumulated);
          chatBox.scrollTop = chatBox.scrollHeight;
        },
        () => {
          if (!accumulated) bubble.innerHTML = '<em>No response.</em>';
          state.messages.push({ role: 'assistant', content: accumulated });
        }
      );
    }

    // Save to history
    if (state.messages.length > 0) {
      saveToHistory(query);
    }
  } catch (err) {
    removeTypingIndicator();
    appendMessage('assistant', `⚠️ Error: ${err.message}`);
  } finally {
    state.streaming = false;
    sendBtn.disabled = false;
  }
}

// ── History ──────────────────────────────────────────────────────────────────
function saveToHistory(title) {
  const id = Date.now().toString();
  state.history.unshift({ id, title: title.slice(0, 60), mode: state.mode });
  renderHistory();
  try {
    localStorage.setItem('arxiv-ai-history', JSON.stringify(state.history.slice(0, 50)));
  } catch {}
}

function renderHistory() {
  historyList.innerHTML = '';
  for (const item of state.history) {
    const btn = document.createElement('button');
    btn.className = 'history-item';
    btn.textContent = item.title;
    btn.title = item.title;
    historyList.appendChild(btn);
  }
}

function loadHistory() {
  try {
    const raw = localStorage.getItem('arxiv-ai-history');
    if (raw) state.history = JSON.parse(raw);
    renderHistory();
  } catch {}
}

$('clearHistoryBtn').addEventListener('click', () => {
  state.history = [];
  renderHistory();
  localStorage.removeItem('arxiv-ai-history');
});

// ── Paper Chat loading ───────────────────────────────────────────────────────
async function loadPaperByArxivId(arxivId) {
  if (!arxivId.trim()) return;
  sessionStatus.className = 'session-status loading';
  sessionStatus.textContent = `Loading ${arxivId}…`;
  loadArxivBtn.disabled = true;
  try {
    const data = await callApi('/chat/load-arxiv', { arxiv_id: arxivId.trim() });
    state.sessionId = data.session_id;
    state.sessionSource = data.source;
    sessionStatus.className = 'session-status ok';
    sessionStatus.textContent = `✓ ${data.cached ? 'Cached' : 'Loaded'}: ${data.chunk_count} chunks`;
    setMode('chat');
    showToast(`Paper loaded: ${arxivId}`);
  } catch (e) {
    sessionStatus.className = 'session-status err';
    sessionStatus.textContent = `Error: ${e.message}`;
  } finally {
    loadArxivBtn.disabled = false;
  }
}

loadArxivBtn.addEventListener('click', () => loadPaperByArxivId(arxivIdInput.value));
arxivIdInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') loadPaperByArxivId(arxivIdInput.value);
});

pdfUpload.addEventListener('change', async () => {
  const file = pdfUpload.files?.[0];
  if (!file) return;
  sessionStatus.className = 'session-status loading';
  sessionStatus.textContent = `Uploading ${file.name}…`;
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/chat/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.sessionId = data.session_id;
    state.sessionSource = data.source;
    sessionStatus.className = 'session-status ok';
    sessionStatus.textContent = `✓ Uploaded: ${data.chunk_count} chunks`;
    setMode('chat');
  } catch (e) {
    sessionStatus.className = 'session-status err';
    sessionStatus.textContent = `Error: ${e.message}`;
  }
  pdfUpload.value = '';
});

// Exposed for paper card buttons
window.loadPaperChat = id => {
  if (!id) return;
  arxivIdInput.value = id;
  loadPaperByArxivId(id);
};

window.summarisePaper = abstract => {
  if (!abstract) return;
  setMode('summarize');
  chatInput.value = abstract.slice(0, 1500);
  sendMessage(chatInput.value);
};

// ── Composer ─────────────────────────────────────────────────────────────────
function autoGrow() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
}

chatInput.addEventListener('input', autoGrow);
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(chatInput.value);
  }
});
sendBtn.addEventListener('click', () => sendMessage(chatInput.value));

// ── Quick chips ───────────────────────────────────────────────────────────────
quickChips.forEach(chip => {
  chip.addEventListener('click', () => {
    const text = chip.textContent;
    if (text.toLowerCase().startsWith('classify')) setMode('classify');
    else if (text.toLowerCase().startsWith('summarize') || text.toLowerCase().startsWith('find')) setMode('search');
    else setMode('ask');
    chatInput.value = text;
    autoGrow();
    sendMessage(text);
  });
});

// ── New chat ──────────────────────────────────────────────────────────────────
newChatBtn.addEventListener('click', showHero);

// ── Sidebar toggle ────────────────────────────────────────────────────────────
mobileSidebarToggle.addEventListener('click', () => sidebar.classList.toggle('open'));
document.addEventListener('click', e => {
  if (!sidebar.contains(e.target) && !mobileSidebarToggle.contains(e.target)) {
    sidebar.classList.remove('open');
  }
});

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  statusDot.className = 'status-dot loading';
  statusText.textContent = 'Connecting…';
  try {
    const data = await fetch('/health').then(r => r.json());
    const ready = data.components?.rag || data.components?.classifier;
    statusDot.className = `status-dot ${ready ? 'ok' : 'err'}`;
    const parts = [];
    if (data.components?.rag)        parts.push('Search');
    if (data.components?.classifier) parts.push('Classify');
    if (data.components?.summarizer) parts.push('Summarize');
    statusText.textContent = parts.length
      ? parts.join(' · ')
      : 'API up (no artifacts)';
  } catch {
    statusDot.className = 'status-dot err';
    statusText.textContent = 'API offline';
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
setMode('ask');
loadHistory();
showHero();
checkHealth();
setInterval(checkHealth, 60_000);
