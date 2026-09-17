'use strict';
// ── History module ───────────────────────────────────────────────────────────
// Depends on: state.js, api.js
// Phase 3: loads history from server (/conversations), scoped to authenticated user

async function loadHistory() {
  try {
    const data = await callApi('/conversations', {}, 'GET');
    const convs = (data.conversations || []);
    historyList.innerHTML = '';
    if (!convs.length) {
      historyList.innerHTML = '<div class="history-empty">No conversations yet</div>';
      return;
    }
    for (const item of convs) {
      const btn = document.createElement('button');
      btn.className = 'history-item';
      btn.title = item.title || 'Untitled';
      btn.innerHTML = `
        <svg width="11" height="11" viewBox="0 0 11 11" fill="none" style="flex-shrink:0">
          <path d="M5.5 1a4.5 4.5 0 1 1 0 9 4.5 4.5 0 0 1 0-9zM5.5 3v3l2 1"
                stroke="currentColor" stroke-width="1.1" stroke-linecap="round"/>
        </svg>
        <span>${esc(item.title || 'Untitled')}</span>`;
      btn.addEventListener('click', () => loadConversation(item.conversation_id, item.title));
      historyList.appendChild(btn);
    }
  } catch (err) {
    // Not authenticated yet or network error — show empty
    historyList.innerHTML = '<div class="history-empty">No conversations yet</div>';
  }
}

async function loadConversation(id, title) {
  try {
    toast(`Loading: "${(title || '').slice(0, 40)}"`, 'info');
    const data = await callApi(`/conversations/${id}`, {}, 'GET');
    state.conversationId = id;
    chatArea.innerHTML = '';
    showChat();
    if (data.turns && data.turns.length > 0) {
      for (const turn of data.turns) {
        if (turn.role === 'user') appendUserMessage(turn.content);
        else if (turn.role === 'assistant') appendAssistantMessage(turn.content);
      }
    } else {
      chatArea.innerHTML = '<div class="history-empty">Conversation is empty.</div>';
    }
    chatArea.scrollTop = chatArea.scrollHeight;
    chatInput.focus();
  } catch (err) {
    if (err.message && err.message.includes('404')) {
      toast('Conversation expired. Starting fresh.', 'info');
      startNewChat();
    } else {
      toast(`Failed to load: ${err.message}`, 'error');
    }
  }
}

// ── Refresh history sidebar after a new message ──────────────────────────────
async function refreshHistory() {
  // Lightweight refresh: just reload from server
  await loadHistory();
}
