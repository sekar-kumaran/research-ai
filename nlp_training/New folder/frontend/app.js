const chatBox = document.getElementById('chatBox');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const topKInput = document.getElementById('topK');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const historyList = document.getElementById('historyList');
const chatMain = document.getElementById('chatMain');
const promptChips = Array.from(document.querySelectorAll('.prompt-chip'));

const STORAGE_KEY = 'research-ai-chat-sessions-v1';

let sessions = [];
let activeSessionId = null;

function autoGrowTextarea(el) {
  el.style.height = 'auto';
  const next = Math.min(el.scrollHeight, 140);
  el.style.height = `${next}px`;
  el.style.overflowY = el.scrollHeight > 140 ? 'auto' : 'hidden';
}

function pickAssistantText(payload) {
  if (!payload || typeof payload !== 'object') {
    return 'No response received.';
  }

  if (payload.error) {
    return `Error: ${payload.error}`;
  }

  if (payload.answer) {
    if (typeof payload.answer === 'string') {
      return payload.answer;
    }
    if (payload.answer.final_answer) {
      return payload.answer.final_answer;
    }
    if (payload.answer.answer) {
      return payload.answer.answer;
    }
    if (payload.answer.paper_answer && payload.answer.paper_answer.answer) {
      return payload.answer.paper_answer.answer;
    }
  }

  if (payload.agent_output) {
    return payload.agent_output;
  }

  if (payload.summary) {
    return payload.summary;
  }

  if (payload.predicted_category) {
    return `Predicted category: ${payload.predicted_category}`;
  }

  return JSON.stringify(payload, null, 2);
}

function escapeHtml(text) {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function markdownToHtml(text) {
  let out = escapeHtml(text || '');

  // Fenced code blocks.
  out = out.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

  // Inline code.
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Links.
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  // Bold and italic.
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // Basic lists line-by-line.
  const lines = out.split('\n');
  let html = '';
  let inUl = false;
  let inOl = false;

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (/^[-*]\s+/.test(line)) {
      if (inOl) {
        html += '</ol>';
        inOl = false;
      }
      if (!inUl) {
        html += '<ul>';
        inUl = true;
      }
      html += `<li>${line.replace(/^[-*]\s+/, '')}</li>`;
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      if (inUl) {
        html += '</ul>';
        inUl = false;
      }
      if (!inOl) {
        html += '<ol>';
        inOl = true;
      }
      html += `<li>${line.replace(/^\d+\.\s+/, '')}</li>`;
      continue;
    }

    if (inUl) {
      html += '</ul>';
      inUl = false;
    }
    if (inOl) {
      html += '</ol>';
      inOl = false;
    }

    if (!line) {
      continue;
    }

    html += `<p>${line}</p>`;
  }

  if (inUl) {
    html += '</ul>';
  }
  if (inOl) {
    html += '</ol>';
  }

  return html || '<p></p>';
}

function renderAssistantBubble(bubble, text) {
  bubble.classList.add('markdown');
  bubble.innerHTML = markdownToHtml(text);
}

async function runAssistantQuery(query, topK) {
  const res = await fetch('/agent/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: 'auto', query, top_k: topK }),
  });

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Request failed (${res.status}): ${msg}`);
  }

  return await res.json();
}

async function runAssistantQueryStream(query, topK, onText) {
  const res = await fetch('/agent/run/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: 'auto', query, top_k: topK }),
  });

  if (!res.ok || !res.body) {
    const msg = await res.text();
    throw new Error(`Stream failed (${res.status}): ${msg}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let full = '';

  while (true) {
    // eslint-disable-next-line no-await-in-loop
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const ev of events) {
      const lines = ev.split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) {
          continue;
        }
        const data = line.slice(6).trim();
        if (data === '[DONE]') {
          return full;
        }
        try {
          const obj = JSON.parse(data);
          const delta = typeof obj.delta === 'string' ? obj.delta : '';
          full += delta;
          onText(full);
        } catch {
          // Ignore malformed chunks and continue.
        }
      }
    }
  }

  return full;
}

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    sessions = Array.isArray(parsed) ? parsed : [];
  } catch {
    sessions = [];
  }

  if (!sessions.length) {
    createNewSession();
  } else {
    activeSessionId = sessions[0].id;
  }
}

function saveSessions() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function getActiveSession() {
  return sessions.find((s) => s.id === activeSessionId);
}

function createNewSession() {
  const session = {
    id: `chat_${Date.now()}`,
    title: 'New chat',
    createdAt: new Date().toISOString(),
    messages: [],
  };
  sessions.unshift(session);
  activeSessionId = session.id;
  saveSessions();
  renderHistory();
  renderChat();
}

function renameSession(sessionId) {
  const session = sessions.find((s) => s.id === sessionId);
  if (!session) {
    return;
  }
  const next = window.prompt('Rename conversation', session.title || 'New chat');
  if (!next) {
    return;
  }
  session.title = next.trim() || session.title;
  saveSessions();
  renderHistory();
}

function deleteSession(sessionId) {
  if (sessions.length <= 1) {
    return;
  }
  const ok = window.confirm('Delete this conversation?');
  if (!ok) {
    return;
  }
  sessions = sessions.filter((s) => s.id !== sessionId);
  if (activeSessionId === sessionId) {
    activeSessionId = sessions[0]?.id || null;
  }
  saveSessions();
  renderHistory();
  renderChat();
}

function setSessionTitleFromFirstUserMessage(session) {
  const firstUser = session.messages.find((m) => m.role === 'user');
  if (!firstUser) {
    return;
  }
  const base = firstUser.text.trim().replace(/\s+/g, ' ');
  session.title = base.length > 44 ? `${base.slice(0, 44)}...` : base;
}

function renderHistory() {
  historyList.innerHTML = '';
  sessions.forEach((session) => {
    const row = document.createElement('div');
    row.className = 'history-row';

    const item = document.createElement('button');
    item.type = 'button';
    item.className = `history-item${session.id === activeSessionId ? ' active' : ''}`;
    item.textContent = session.title || 'New chat';
    item.addEventListener('click', () => {
      activeSessionId = session.id;
      renderHistory();
      renderChat();
      chatInput.focus();
    });

    const actions = document.createElement('div');
    actions.className = 'history-actions';

    const renameBtn = document.createElement('button');
    renameBtn.type = 'button';
    renameBtn.title = 'Rename';
    renameBtn.textContent = 'Ren';
    renameBtn.addEventListener('click', () => renameSession(session.id));

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.title = 'Delete';
    deleteBtn.textContent = 'Del';
    deleteBtn.disabled = sessions.length <= 1;
    deleteBtn.addEventListener('click', () => deleteSession(session.id));

    actions.appendChild(renameBtn);
    actions.appendChild(deleteBtn);
    row.appendChild(item);
    row.appendChild(actions);
    historyList.appendChild(row);
  });
}

function findPreviousUserText(messages, assistantIndex) {
  for (let i = assistantIndex - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'user') {
      return messages[i].text;
    }
  }
  return null;
}

function copyText(text) {
  if (!navigator.clipboard) {
    return;
  }
  navigator.clipboard.writeText(text).catch(() => {
    // Best-effort clipboard copy only.
  });
}

async function animateAssistantText(bubble, finalText) {
  bubble.classList.add('streaming');
  bubble.textContent = '';

  const step = Math.max(1, Math.floor(finalText.length / 180));
  for (let i = 0; i < finalText.length; i += step) {
    bubble.textContent = finalText.slice(0, i + step);
    chatBox.scrollTop = chatBox.scrollHeight;
    // Small delay to simulate token streaming.
    // eslint-disable-next-line no-await-in-loop
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  bubble.textContent = finalText;
  bubble.classList.remove('streaming');
}

function renderChat() {
  const session = getActiveSession();
  if (!session) {
    return;
  }

  const hasUserMessage = session.messages.some((msg) => msg.role === 'user');
  chatMain.classList.toggle('has-messages', hasUserMessage);

  chatBox.innerHTML = '';
  session.messages.forEach((msg, index) => {
    const row = document.createElement('div');
    row.className = 'message';

    const bubble = document.createElement('div');
    bubble.className = msg.role === 'user' ? 'bubble user' : 'bubble assistant';
    if (msg.role === 'assistant') {
      renderAssistantBubble(bubble, msg.text);
    } else {
      bubble.textContent = msg.text;
    }
    row.appendChild(bubble);

    if (msg.role === 'assistant') {
      const actions = document.createElement('div');
      actions.className = 'msg-actions';

      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.textContent = 'Copy';
      copyBtn.addEventListener('click', () => copyText(msg.text));
      actions.appendChild(copyBtn);

      const regenBtn = document.createElement('button');
      regenBtn.type = 'button';
      regenBtn.textContent = 'Regenerate';
      regenBtn.addEventListener('click', async () => {
        const userText = findPreviousUserText(session.messages, index);
        if (!userText) {
          return;
        }
        await sendUserMessage(userText, true);
      });
      actions.appendChild(regenBtn);

      row.appendChild(actions);
    }

    chatBox.appendChild(row);
  });

  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendUserMessage(query, isRegenerate = false) {
  const session = getActiveSession();
  if (!session) {
    return;
  }

  const topK = Number(topKInput.value) || 5;

  if (!isRegenerate) {
    session.messages.push({ role: 'user', text: query });
  }

  const assistantMessage = { role: 'assistant', text: 'Thinking...' };
  session.messages.push(assistantMessage);

  setSessionTitleFromFirstUserMessage(session);
  saveSessions();
  renderHistory();
  renderChat();

  sendBtn.disabled = true;

  try {
    const allAssistantBubbles = chatBox.querySelectorAll('.bubble.assistant');
    const pendingBubble = allAssistantBubbles[allAssistantBubbles.length - 1];
    if (pendingBubble) {
      pendingBubble.classList.add('streaming');
    }

    const streamed = await runAssistantQueryStream(query, topK, (currentText) => {
      if (pendingBubble) {
        pendingBubble.textContent = currentText;
        chatBox.scrollTop = chatBox.scrollHeight;
      }
    });

    assistantMessage.text = streamed || 'No response received.';
  } catch (err) {
    try {
      const out = await runAssistantQuery(query, topK);
      assistantMessage.text = pickAssistantText(out);
    } catch (fallbackErr) {
      assistantMessage.text = `Error: ${fallbackErr.message || err.message}`;
    }
  }

  saveSessions();
  renderHistory();
  renderChat();

  sendBtn.disabled = false;
  chatInput.focus();
}

chatInput.addEventListener('input', () => autoGrowTextarea(chatInput));

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

newChatBtn.addEventListener('click', () => {
  createNewSession();
  chatInput.value = '';
  autoGrowTextarea(chatInput);
  chatInput.focus();
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) {
    return;
  }
  chatInput.value = '';
  autoGrowTextarea(chatInput);
  await sendUserMessage(query, false);
});

promptChips.forEach((chip) => {
  chip.addEventListener('click', async () => {
    const prompt = (chip.dataset.prompt || '').trim();
    if (!prompt) {
      return;
    }
    chatInput.value = '';
    autoGrowTextarea(chatInput);
    await sendUserMessage(prompt, false);
  });
});

loadSessions();
renderHistory();
renderChat();
autoGrowTextarea(chatInput);
