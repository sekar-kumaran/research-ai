'use strict';
// ── API helpers ──────────────────────────────────────────────────────────────
// Depends on: state.js (getToken reads localStorage directly)

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
