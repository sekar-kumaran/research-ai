'use strict';
// ── Shared application state ────────────────────────────────────────────────
const state = {
  conversationId: null,
  loadedSessions: [],
  topK: 5,
  debug: false,
  streaming: false,
  theme: localStorage.getItem('theme') || 'dark',
};

// ── DOM helpers ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── DOM refs ────────────────────────────────────────────────────────────────
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
const loginEmail     = $('loginEmail');
const loginPassword  = $('loginPassword');
const loginBtn       = $('loginBtn');
const loginError     = $('loginError');
const authLoginTab   = $('authLoginTab');
const authSignupTab  = $('authSignupTab');
const signupUsername = $('signupUsername');
const exportBtn      = $('exportBtn');
const kgBtn          = $('kgBtn');
const kgOverlay      = $('kgOverlay');
const kgClose        = $('kgClose');
const kgContent      = $('kgContent');
const loginHeaderBtn = $('loginHeaderBtn');

// ── Theme ───────────────────────────────────────────────────────────────────
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

// ── Helpers ─────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function mdToHtml(text) {
  let out = esc(text || '');
  out = out.replace(/```[\w]*\n?([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  out = out.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  out = out.replace(/(^|[\s>])(https?:\/\/[^\s<"&]+)/g,
    '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
  out = out.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  out = out.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  out = out.replace(/^## (.+)$/gm, '<h3>$1</h3>');
  out = out.replace(/^# (.+)$/gm, '<h3>$1</h3>');
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

function toast(msg, type = 'info', dur = 4000) {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.classList.add('toast-visible'), 10);
  setTimeout(() => { el.classList.remove('toast-visible'); setTimeout(() => el.remove(), 300); }, dur);
}
