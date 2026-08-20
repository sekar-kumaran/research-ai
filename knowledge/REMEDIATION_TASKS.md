# Remediation Pass - Task Tracker

## Phase 0 — Read Knowledge Base
- [x] Read all knowledge/ files

## Phase 1 — Root Causes
- [ ] Fix #1: Remote ML wiring self-referential HF_SPACE_ID (decide architecture)
- [ ] Fix #2: Git-LFS stub artifacts (faiss/parquet)
- [ ] Fix #3: Startup diagnostics never run in deployed environment

## Phase 2 — High-Priority Correctness & Security
- [ ] Fix #4: Middleware order breaks CORS preflight
- [ ] Fix #5: Authorization header parsing unhandled 500
- [ ] Fix #6: render.yaml free tier OOM
- [ ] Fix #7: CLOUD_LLM_PROVIDER default mismatch

## Phase 3 — Hygiene & Hardening
- [ ] Fix #8: Remove committed __pycache__ bytecode
- [ ] Fix #9: Add hf_microservice/README.md
- [ ] Fix #10: Clean pyflakes warnings
- [ ] Fix #11: Rate limiting / auth note
- [ ] Fix #12: Integration tests

## Post-fix
- [ ] Update knowledge/ after each phase
- [ ] Write CHANGES.md
