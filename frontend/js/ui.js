'use strict';
// ── UI module ────────────────────────────────────────────────────────────────
// Depends on: state.js, api.js
// Handles: health, models, sidebar, modal, export, knowledge graph

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  if (statusDot) statusDot.className = 'status-dot loading';
  if (statusText) statusText.textContent = 'Connecting…';
  try {
    const data = await fetch('/health').then(r => r.json());
    const c = data.components || {};
    const ready = c.hybrid_retrieval || c.classifier || c.paper_chat;
    if (statusDot) statusDot.className = `status-dot ${ready ? 'ok' : 'warn'}`;
    const parts = [];
    if (c.hybrid_retrieval) parts.push('Search');
    if (c.classifier) parts.push('Classify');
    if (c.summarizer) parts.push('Summarize');
    if (c.paper_chat) parts.push('Chat');
    if (statusText) statusText.textContent = parts.length ? parts.join(' · ') : data.version || 'Online';
  } catch (_) {
    if (statusDot) statusDot.className = 'status-dot err';
    if (statusText) statusText.textContent = 'API offline';
  }
}

// ── Model list ────────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const data = await callApi('/models/list', {}, 'GET').catch(() => null);
    if (!data || !data.available || !data.models.length) return;
    if (modelsSection) modelsSection.style.display = '';
    if (modelsList) {
      modelsList.innerHTML = data.models.map(m => `
        <div class="model-item">
          <span class="model-name">${esc(m.name)}</span>
          <span class="model-tier tier-${m.tier}">${esc(m.tier_label)}</span>
          ${m.size_gb ? `<span class="model-size">${m.size_gb}GB</span>` : ''}
        </div>`).join('');
    }
  } catch (_) {}
}

// ── Mobile sidebar ────────────────────────────────────────────────────────────
const sidebar   = document.querySelector('.sidebar');
const sidebarOv = $('sidebarOverlay');

function openSidebar()  { if (sidebar) sidebar.classList.add('open');    if (sidebarOv) sidebarOv.classList.add('visible'); }
function closeSidebar() { if (sidebar) sidebar.classList.remove('open'); if (sidebarOv) sidebarOv.classList.remove('visible'); }

const mobileSidebarToggle = $('mobileSidebarToggle');
if (mobileSidebarToggle) {
  mobileSidebarToggle.addEventListener('click', () =>
    sidebar && sidebar.classList.contains('open') ? closeSidebar() : openSidebar()
  );
}
if (sidebarOv) sidebarOv.addEventListener('click', closeSidebar);

// ── Paper modal ───────────────────────────────────────────────────────────────
if (modalClose)   modalClose.addEventListener('click', () => { if (modalOverlay) modalOverlay.style.display = 'none'; });
if (modalOverlay) modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) modalOverlay.style.display = 'none'; });

// ── Export ────────────────────────────────────────────────────────────────────
if (exportBtn) {
  exportBtn.addEventListener('click', () => {
    if (!state.conversationId) return toast('No active conversation to export.', 'info');
    toast('Preparing export...', 'info');
    callApi(`/conversations/${state.conversationId}`, {}, 'GET').then(data => {
      let md = `# Research AI Export\nDate: ${new Date().toLocaleString()}\n\n---\n\n`;
      if (data.turns) {
        data.turns.forEach(turn => {
          md += `### ${turn.role === 'user' ? 'User' : 'Research AI'}\n\n${turn.content}\n\n---\n\n`;
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
    }).catch(() => toast('Failed to export conversation.', 'error'));
  });
}

// ── Knowledge graph ───────────────────────────────────────────────────────────
if (kgBtn) {
  kgBtn.addEventListener('click', async () => {
    if (kgOverlay) kgOverlay.style.display = 'flex';
    if (kgContent) kgContent.innerHTML = 'Loading knowledge graph...';
    try {
      const data = await callApi('/knowledge-graph', {}, 'GET');
      if (data.concepts && Object.keys(data.concepts).length > 0) {
        let html = '<div style="display:flex;flex-wrap:wrap;gap:8px;padding-top:4px;">';
        const sorted = Object.entries(data.concepts).sort((a, b) => b[1] - a[1]);
        for (const [concept, count] of sorted) {
          html += `<span style="padding:6px 12px;background:var(--bg-3);border:1px solid var(--border);border-radius:16px;font-size:13px;color:var(--text);cursor:default;"
                         onmouseover="this.style.borderColor='var(--accent)'"
                         onmouseout="this.style.borderColor='var(--border)'">
                     <strong>${esc(concept)}</strong>
                     <span style="color:var(--text-3);font-size:11px;margin-left:4px;">${count}</span>
                   </span>`;
        }
        html += '</div>';
        if (kgContent) kgContent.innerHTML = html;
      } else {
        if (kgContent) kgContent.innerHTML = '<div style="text-align:center;padding:20px;color:var(--text-3);">No concepts yet. Start chatting!</div>';
      }
    } catch (_) {
      if (kgContent) kgContent.innerHTML = '<div style="color:var(--conf-low);">Failed to load knowledge graph.</div>';
    }
  });
}
if (kgClose) kgClose.addEventListener('click', () => { if (kgOverlay) kgOverlay.style.display = 'none'; });
if (kgOverlay) kgOverlay.addEventListener('click', e => { if (e.target === kgOverlay) kgOverlay.style.display = 'none'; });
