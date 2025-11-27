## Task snapshot – 2025-11-13

- **Objective:** Run the ONH Expert Connector FastAPI app locally, exercise the UI with Playwright in a real browser session, catalogue visible bugs, then iterate on fixes.
- **Progress so far:**
  - Confirmed repo layout, configs, and current FastAPI setup.
  - Attempts to start `uvicorn` under the provided sandbox fail because the embedding backend needs to download `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face, which is blocked when `network_access` is `restricted`. The server terminates after repeated `MaxRetryError` messages (see `/tmp/uvicorn_fail.log`).
  - Multiple Playwright runs (`npx playwright test`, custom `node scripts/playwright_smoke.js`) were attempted, but the sandbox disallows launching Chromium/Firefox headless shells. Each launch crashes immediately with `content/browser/sandbox_host_linux.cc:41` / `shutdown: Operation not permitted (1)` and sockets cannot be opened on 127.0.0.1.
- **Current blocker:** No outbound HTTPS and no permission to start Playwright browsers, so the app can’t download models or serve HTTP, and browsers can’t be launched to inspect the UI.
- **Next steps once access is granted:**
  1. Enable network + browser permissions in the Codex session (need `network_access=enabled` and sandbox rules that allow Chromium/Firefox headless sockets).
  2. Restart FastAPI with a `timeout … uvicorn app.main:app --host 127.0.0.1 --port 8000` guard and verify `/config` responds.
  3. Run Playwright (`npx playwright install --with-deps` if needed, then `npx playwright test` or a custom script) against the live app, capture screenshots, logs, and enumerate bugs for the user.
  4. Resume the bug-fix / verification loop requested by the user.
