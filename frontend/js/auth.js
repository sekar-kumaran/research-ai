'use strict';
// ── Auth module ──────────────────────────────────────────────────────────────
// Depends on: state.js, api.js
// Handles: login, signup, logout, header state update, auth check on init

let authMode = 'login';

function setAuthMode(mode) {
  authMode = mode;
  if (authLoginTab) authLoginTab.classList.toggle('active', mode === 'login');
  if (authSignupTab) authSignupTab.classList.toggle('active', mode === 'signup');
  if (signupUsername) signupUsername.style.display = mode === 'signup' ? '' : 'none';
  if (loginBtn) loginBtn.textContent = mode === 'signup' ? 'Create account' : 'Login';
  if (loginError) loginError.style.display = 'none';
}

// ── Header button: shows "Login" or "username ✕" ───────────────────────────
function updateAuthHeader(user) {
  if (!loginHeaderBtn) return;
  const userIcon = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
    <circle cx="12" cy="7" r="4"></circle>
  </svg>`;
  if (user) {
    const label = user.username || user.email || 'Account';
    loginHeaderBtn.innerHTML = `${userIcon} ${esc(label)} ✕`;
    loginHeaderBtn.title = 'Logout';
    loginHeaderBtn.onclick = async () => {
      const token = getToken();
      if (token) {
        try {
          await fetch('/api/auth/logout', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` }
          });
        } catch (_) {}
      }
      localStorage.removeItem('rai-token');
      updateAuthHeader(null);
      if (loginOverlay) loginOverlay.style.display = 'flex';
      setTimeout(() => { if (loginEmail) loginEmail.focus(); }, 100);
    };
  } else {
    loginHeaderBtn.innerHTML = `${userIcon} Login`;
    loginHeaderBtn.title = 'Login / Settings';
    loginHeaderBtn.onclick = () => {
      if (loginOverlay) loginOverlay.style.display = 'flex';
      if (loginEmail) loginEmail.focus();
    };
  }
}

// ── Validate existing token via /api/auth/me ────────────────────────────────
async function checkAccountAuth() {
  const token = getToken();
  if (!token) {
    if (loginOverlay) loginOverlay.style.display = 'flex';
    setTimeout(() => { if (loginEmail) loginEmail.focus(); }, 100);
    return false;
  }
  try {
    const res = await fetch('/api/auth/me', {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (res.ok) {
      const user = await res.json();
      if (loginOverlay) loginOverlay.style.display = 'none';
      updateAuthHeader(user);
      return true;
    }
    localStorage.removeItem('rai-token');
    updateAuthHeader(null);
    if (loginOverlay) loginOverlay.style.display = 'flex';
    setTimeout(() => { if (loginEmail) loginEmail.focus(); }, 100);
    return false;
  } catch (_) {
    // Network error — let them in, will fail on API calls
    return true;
  }
}

// ── Login / Signup button ────────────────────────────────────────────────────
if (loginBtn) {
  loginBtn.addEventListener('click', async () => {
    const login = loginEmail ? loginEmail.value.trim() : '';
    const pwd   = loginPassword ? loginPassword.value.trim() : '';
    const uname = signupUsername ? signupUsername.value.trim() : '';
    if (!login || !pwd || (authMode === 'signup' && !uname)) return;

    const originalText = loginBtn.textContent;
    loginBtn.textContent = authMode === 'signup' ? 'Creating...' : 'Verifying...';
    loginBtn.disabled = true;
    if (loginError) loginError.style.display = 'none';

    try {
      const endpoint = authMode === 'signup' ? '/api/auth/signup' : '/api/auth/login';
      const body = authMode === 'signup'
        ? { email: login, username: uname, password: pwd }
        : { login, password: pwd };

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (res.ok) {
        const data = await res.json();
        localStorage.setItem('rai-token', data.access_token);
        if (loginOverlay) loginOverlay.style.display = 'none';
        updateAuthHeader(data.user);
        await loadHistory();
        checkHealth();
        loadModels();
      } else {
        let msg = authMode === 'signup' ? 'Could not create account' : 'Invalid login';
        try { const err = await res.json(); if (err.detail) msg = err.detail; } catch (_) {}
        if (loginError) { loginError.textContent = msg; loginError.style.display = 'block'; }
      }
    } catch (_) {
      if (loginError) { loginError.textContent = 'Connection error'; loginError.style.display = 'block'; }
    } finally {
      loginBtn.textContent = originalText;
      loginBtn.disabled = false;
    }
  });
}

// ── Enter key on inputs ──────────────────────────────────────────────────────
[loginEmail, signupUsername, loginPassword].forEach(input => {
  if (input) input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && loginBtn) loginBtn.click();
  });
});

// ── Auth mode tabs ───────────────────────────────────────────────────────────
if (authLoginTab) authLoginTab.addEventListener('click', () => setAuthMode('login'));
if (authSignupTab) authSignupTab.addEventListener('click', () => setAuthMode('signup'));
setAuthMode('login');
