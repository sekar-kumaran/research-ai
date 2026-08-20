from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.on('console', lambda msg: print(f'CONSOLE: {msg.type}: {msg.text}'))
    page.on('pageerror', lambda exc: print(f'ERROR: {exc}'))
    
    print("Navigating to app...")
    page.goto('http://127.0.0.1:8000')
    page.wait_for_selector('#chatInput')
    
    print("Typing message...")
    page.fill('#chatInput', 'What is BERT?')
    page.click('#sendBtn')
    
    print("Waiting for response...")
    page.wait_for_timeout(10000)
    
    msgs = page.locator('.msg.assistant').count()
    print(f"Assistant messages found: {msgs}")
    
    browser.close()
