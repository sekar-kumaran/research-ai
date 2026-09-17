'use strict';
// ── Chat module ──────────────────────────────────────────────────────────────
// Depends on: state.js, api.js, history.js

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

function createAssistantShell() {
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant';
  wrap.innerHTML = `
    <div class="msg-avatar ai-avatar">AI</div>
    <div class="msg-content">
      <div class="msg-bubble ai-bubble" id="streamTarget">
        <div class="typing-indicator"><span></span><span></span><span></span></div>
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
      <details class="debug-trace" style="display:none">
        <summary>Execution trace</summary>
        <pre></pre>
      </details>
    </div>`;
  chatArea.appendChild(wrap);
  chatArea.scrollTop = chatArea.scrollHeight;

  const bubble     = wrap.querySelector('#streamTarget');
  bubble.removeAttribute('id');
  const meta       = wrap.querySelector('.msg-meta');
  const timeEl     = wrap.querySelector('.msg-time');
  const confBadge  = wrap.querySelector('.confidence-badge');
  const intentBadge= wrap.querySelector('.intent-badge');
  const srcSection = wrap.querySelector('.sources-section');
  const srcToggle  = wrap.querySelector('.sources-toggle');
  const srcList    = wrap.querySelector('.sources-list');
  const debugTrace = wrap.querySelector('.debug-trace');
  const debugPre   = wrap.querySelector('.debug-trace pre');

  srcToggle.addEventListener('click', () => {
    const open = srcList.style.display !== 'none';
    srcList.style.display = open ? 'none' : '';
    srcToggle.classList.toggle('open', !open);
  });

  return { wrap, bubble, meta, timeEl, confBadge, intentBadge, srcSection, srcList, debugTrace, debugPre };
}

function finalizeAssistantBubble(shell, data) {
  const { bubble, meta, timeEl, confBadge, intentBadge, srcSection, srcList } = shell;
  const { sources = [], confidence = 0, intent = '', tools_used = [] } = data;
  timeEl.textContent = nowStr();
  const pct = Math.round(confidence * 100);
  const confClass = pct >= 70 ? 'conf-high' : pct >= 40 ? 'conf-mid' : 'conf-low';
  confBadge.textContent = `${pct}% confidence`;
  confBadge.className = `confidence-badge ${confClass}`;
  confBadge.title = `Evidence confidence: ${pct}% (tools: ${tools_used.join(', ')})`;
  if (intent && intent !== 'research_analysis') {
    intentBadge.textContent = intent.replace(/_/g, ' ');
    intentBadge.className = 'intent-badge';
  }
  meta.style.display = '';
  if (sources.length) {
    renderSources(srcList, sources);
    srcSection.querySelector('.sources-toggle').childNodes[1].textContent = ` Sources (${sources.length})`;
    srcSection.style.display = '';
    if (sources.length <= 3) {
      srcList.style.display = '';
      srcSection.querySelector('.sources-toggle').classList.add('open');
    }
  }
  if (data.debug_trace) renderDebugTrace(shell, data.debug_trace);
}

function renderDebugTrace(shell, trace) {
  if (!shell.debugTrace || !shell.debugPre) return;
  shell.debugPre.textContent = JSON.stringify({
    request_id: trace.request_id,
    intent: trace.mode,
    plan: trace.plan,
    tools: Object.keys(trace.executor_output || {}),
    evaluation: trace.evaluation,
    latency_ms: trace.latency_ms,
  }, null, 2);
  shell.debugTrace.style.display = '';
}

function appendAssistantMessage(text) {
  const shell = createAssistantShell();
  shell.bubble.innerHTML = mdToHtml(text);
  finalizeAssistantBubble(shell, { sources: [], confidence: 1.0, intent: '' });
}

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
        ${src.abstract_snippet ? `<div class="source-snippet">${esc(src.abstract_snippet)}</div>` : ''}
        <div class="source-actions">
          ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer" class="source-link">arXiv ↗</a>` : ''}
          ${pid ? `<button class="source-link chat-btn" data-arxiv="${esc(pid)}">Chat with paper</button>` : ''}
        </div>
      </div>`;
    const chatBtn = card.querySelector('.chat-btn');
    if (chatBtn) chatBtn.addEventListener('click', () => loadArxivPaper(chatBtn.dataset.arxiv));
    srcList.appendChild(card);
  });
}

// ── Core send ────────────────────────────────────────────────────────────────
async function sendMessage(query) {
  if (state.streaming) return;
  query = (query || '').trim();
  if (!query) return;

  showChat();
  appendUserMessage(query);
  chatInput.value = '';
  autoGrow();
  sendBtn.disabled = true;
  state.streaming = true;

  const shell = createAssistantShell();
  chatArea.scrollTop = chatArea.scrollHeight;

  const sessionId = state.loadedSessions.length
    ? state.loadedSessions.map(s => s.session_id).join(',')
    : null;

  let accumulated = '';
  let streamDone = false;
  const timeoutMs = 120_000;
  let timeoutHandle = null;

  function armTimeout() {
    if (timeoutHandle) clearTimeout(timeoutHandle);
    timeoutHandle = setTimeout(() => {
      if (!streamDone) {
        if (!accumulated) shell.bubble.innerHTML = '<em>Response timed out. Please try again.</em>';
        finishStreaming();
      }
    }, timeoutMs);
  }
  armTimeout();

  const msgSteps = document.createElement('div');
  msgSteps.className = 'msg-steps';
  shell.bubble.parentNode.insertBefore(msgSteps, shell.bubble);

  const stepMessages = [
    'Analyzing question intent...',
    'Selecting ML models...',
    'Retrieving vector embeddings...',
    'Extracting paper context...',
    'Processing with LLM...',
  ];
  let currentStepIdx = 0;

  function advanceStep() {
    if (accumulated || currentStepIdx >= stepMessages.length) {
      if (msgSteps.parentNode) msgSteps.style.display = 'none';
      return;
    }
    const prev = msgSteps.querySelector('.msg-step.active');
    if (prev) { prev.classList.remove('active'); prev.classList.add('completed'); }
    const stepEl = document.createElement('div');
    stepEl.className = 'msg-step active';
    stepEl.innerHTML = `<div class="step-icon"></div> <span>${stepMessages[currentStepIdx]}</span>`;
    msgSteps.appendChild(stepEl);
    currentStepIdx++;
    chatArea.scrollTop = chatArea.scrollHeight;
  }
  advanceStep();

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
    let pendingData = {};

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
        if (line === '[DONE]') break;
        if (rawLine === ': keepalive' || rawLine === ':keepalive') { advanceStep(); continue; }
        if (!line) continue;
        try {
          const obj = JSON.parse(line);
          if (obj.delta !== undefined) {
            if (!accumulated) {
              shell.bubble.innerHTML = '';
              if (msgSteps.parentNode) msgSteps.style.display = 'none';
            }
            accumulated += obj.delta;
            shell.bubble.innerHTML = mdToHtml(accumulated);
            chatArea.scrollTop = chatArea.scrollHeight;
          } else if (obj.event === 'start') {
            if (obj.conversation_id) state.conversationId = obj.conversation_id;
          } else if (obj.event === 'sources') {
            pendingData.sources = obj.sources || [];
          } else if (obj.event === 'debug') {
            pendingData.debug_trace = obj.debug_trace || {};
            renderDebugTrace(shell, pendingData.debug_trace);
          } else if (obj.event === 'done') {
            if (obj.conversation_id) state.conversationId = obj.conversation_id;
            pendingData.confidence = obj.confidence || 0;
          } else if (obj.event === 'error') {
            shell.bubble.innerHTML = `<span class="error-text">Error: ${esc(obj.message || 'Unknown error')}</span>`;
          }
        } catch (_) {}
      }
    }

    if (!accumulated && !shell.bubble.querySelector('.error-text')) {
      shell.bubble.innerHTML = '<em>No response received.</em>';
    }
    finalizeAssistantBubble(shell, pendingData);
    // Refresh history sidebar to show new conversation
    await refreshHistory();

  } catch (err) {
    shell.bubble.innerHTML = `<span class="error-text">${esc(err.message)}</span>`;
    shell.meta.style.display = '';
    shell.timeEl.textContent = nowStr();
  } finally {
    finishStreaming();
  }
}

// ── Composer ─────────────────────────────────────────────────────────────────
function autoGrow() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
}

chatInput.addEventListener('input', () => {
  autoGrow();
  sendBtn.disabled = !chatInput.value.trim() || state.streaming;
});
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(chatInput.value); }
});
sendBtn.addEventListener('click', () => sendMessage(chatInput.value));

// ── Settings ──────────────────────────────────────────────────────────────────
topKSlider.addEventListener('input', () => {
  state.topK = parseInt(topKSlider.value, 10);
  topKVal.textContent = state.topK;
});
debugToggle.addEventListener('change', () => { state.debug = debugToggle.checked; });

// ── Example chips ─────────────────────────────────────────────────────────────
document.querySelectorAll('.example-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    chatInput.value = chip.textContent.trim();
    autoGrow();
    sendBtn.disabled = false;
    sendMessage(chatInput.value);
  });
});

// ── New chat ──────────────────────────────────────────────────────────────────
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
