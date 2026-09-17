'use strict';
// ── Main entry point ──────────────────────────────────────────────────────────
// Depends on: state.js, api.js, auth.js, chat.js, history.js, documents.js, ui.js
// Loaded LAST in index.html

async function init() {
  showWelcome();
  const authed = await checkAccountAuth();
  if (authed) {
    await loadHistory();
    checkHealth();
    loadModels();
  }
  setInterval(checkHealth, 60_000);
}

init();
