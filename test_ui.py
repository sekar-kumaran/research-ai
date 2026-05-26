"""End-to-end UI test — drives the ChatGPT-like interface with Playwright."""
from playwright.sync_api import sync_playwright
import time, sys

WAIT_STREAM = 90_000   # ms — allow long Ollama responses
BASE = "http://localhost:8000"

results = []

def ok(label):
    results.append(("PASS", label))
    print(f"  PASS  {label}")

def fail(label, reason=""):
    results.append(("FAIL", label))
    print(f"  FAIL  {label}" + (f" — {reason}" if reason else ""))

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=60)
    page = browser.new_page(viewport={"width": 1400, "height": 860})
    page.goto(BASE)
    page.wait_for_load_state("networkidle")

    # ── TEST 1: Welcome screen rendered ──────────────────────────────────────
    print("\n── Test 1: Welcome screen ──")
    title_el = page.locator(".welcome-title")
    if title_el.is_visible() and "Research" in title_el.inner_text():
        ok("Welcome title visible")
    else:
        fail("Welcome title", "not visible")

    chips = page.locator(".example-chip").all()
    if len(chips) >= 4:
        ok(f"Example chips rendered ({len(chips)})")
    else:
        fail("Example chips", f"only {len(chips)}")

    page.screenshot(path="screen_01_welcome.png")

    # ── TEST 2: Send a research query via example chip ────────────────────────
    print("\n── Test 2: Research query (example chip) ──")
    chips[0].click()

    # Typing indicator should appear
    try:
        page.wait_for_selector(".typing-indicator", timeout=8000)
        ok("Typing indicator appeared")
    except:
        fail("Typing indicator", "did not appear")

    # Wait for streaming to finish
    try:
        page.wait_for_function(
            "() => { const b = document.querySelector('.ai-bubble'); return b && !b.querySelector('.typing-indicator') && b.innerText.trim().length > 20; }",
            timeout=WAIT_STREAM
        )
        ok("Streaming response completed")
    except Exception as e:
        fail("Streaming response", str(e))

    time.sleep(0.5)
    page.screenshot(path="screen_02_response.png")

    # ── TEST 3: Confidence badge ──────────────────────────────────────────────
    print("\n── Test 3: Confidence badge ──")
    badge = page.locator(".confidence-badge").first
    if badge.is_visible():
        txt = badge.inner_text()
        ok(f"Confidence badge: {txt}")
    else:
        fail("Confidence badge", "not visible")

    # ── TEST 4: Sources panel ─────────────────────────────────────────────────
    print("\n── Test 4: Sources panel ──")
    src_toggle = page.locator(".sources-toggle").first
    if src_toggle.is_visible():
        ok(f"Sources toggle: {src_toggle.inner_text().strip()[:50]}")
        src_toggle.click()
        time.sleep(0.3)
        cards = page.locator(".source-card").all()
        if cards:
            ok(f"Source cards visible ({len(cards)})")
            title_txt = page.locator(".source-title").first.inner_text()
            ok(f"First source: {title_txt[:60]}")
        else:
            fail("Source cards", "none visible after toggle")
    else:
        fail("Sources toggle", "not visible")

    page.screenshot(path="screen_03_sources.png")

    # ── TEST 5: History entry added ────────────────────────────────────────────
    print("\n── Test 5: Conversation history ──")
    hist = page.locator(".history-item").all()
    if hist:
        ok(f"History entry added: {hist[0].inner_text().strip()[:50]}")
    else:
        fail("History entry", "none found")

    # ── TEST 6: Follow-up question (conversation continuity) ─────────────────
    print("\n── Test 6: Follow-up / multi-turn ──")
    page.locator("#chatInput").fill("Which of those used attention mechanisms?")
    page.locator("#chatInput").press("Enter")

    try:
        page.wait_for_selector(".typing-indicator", timeout=8000)
        ok("Follow-up: typing indicator appeared")
    except:
        fail("Follow-up: typing indicator")

    try:
        page.wait_for_function(
            "() => { const bubbles = document.querySelectorAll('.ai-bubble'); const last = bubbles[bubbles.length-1]; return last && !last.querySelector('.typing-indicator') && last.innerText.trim().length > 20; }",
            timeout=WAIT_STREAM
        )
        ok("Follow-up: response received")
    except Exception as e:
        fail("Follow-up response", str(e))

    time.sleep(0.4)
    page.screenshot(path="screen_04_followup.png")

    # ── TEST 7: Theme toggle ───────────────────────────────────────────────────
    print("\n── Test 7: Theme toggle (dark → light) ──")
    page.locator("#themeToggle").click()
    time.sleep(0.3)
    theme = page.locator("html").get_attribute("data-theme")
    if theme == "light":
        ok("Theme switched to light")
    else:
        fail("Theme toggle", f"data-theme={theme}")
    page.screenshot(path="screen_05_light_theme.png")

    # Switch back to dark
    page.locator("#themeToggle").click()
    time.sleep(0.2)

    # ── TEST 8: New chat resets to welcome ─────────────────────────────────────
    print("\n── Test 8: New chat button ──")
    page.locator("#newChatBtn").click()
    time.sleep(0.3)
    if page.locator(".welcome").is_visible():
        ok("New chat shows welcome screen")
    else:
        fail("New chat", "welcome screen not shown")
    page.screenshot(path="screen_06_new_chat.png")

    # ── TEST 9: arXiv load in sidebar ─────────────────────────────────────────
    print("\n── Test 9: arXiv paper load ──")
    page.locator("#arxivInput").fill("2303.08774")
    page.locator("#loadArxivBtn").click()
    try:
        page.wait_for_function(
            "() => document.querySelectorAll('.doc-item').length > 0",
            timeout=20000
        )
        doc = page.locator(".doc-item").first.inner_text()
        ok(f"arXiv paper loaded: {doc.strip()[:40]}")
    except Exception as e:
        fail("arXiv load", str(e))
    page.screenshot(path="screen_07_arxiv_loaded.png")

    # ── TEST 10: Direct classify endpoint still works ──────────────────────────
    print("\n── Test 10: /classify endpoint (internal tool still reachable) ──")
    import urllib.request, json
    req_data = json.dumps({"title":"attention is all you need","abstract":"transformer self-attention"}).encode()
    req = urllib.request.Request("http://localhost:8000/classify", data=req_data, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        ok(f"Classify: {data.get('predicted_category','?')} (confidence data present: {'confidence' in data})")
    except Exception as e:
        fail("Classify endpoint", str(e))

    # ── TEST 11: Debug mode toggle ─────────────────────────────────────────────
    print("\n── Test 11: Debug mode toggle ──")
    page.locator("#chatInput").fill("hello")
    page.locator("#debugToggle").click()   # enable debug
    time.sleep(0.1)
    page.locator("#chatInput").press("Enter")
    try:
        page.wait_for_function(
            "() => { const b = document.querySelector('.ai-bubble'); return b && !b.querySelector('.typing-indicator') && b.innerText.trim().length > 5; }",
            timeout=WAIT_STREAM
        )
        ok("Debug mode: response received")
    except Exception as e:
        fail("Debug mode response", str(e))
    page.locator("#debugToggle").click()   # disable debug
    page.screenshot(path="screen_08_debug.png")

    browser.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
passed = sum(1 for s,_ in results if s == "PASS")
failed = sum(1 for s,_ in results if s == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed")
for s, label in results:
    print(f"  [{s}] {label}")
sys.exit(0 if failed == 0 else 1)
