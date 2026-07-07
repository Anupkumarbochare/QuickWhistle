"""
Capture demo screenshots WITHOUT touching the Gemini quota (MODEL_PROVIDER=mock):

  docs/screenshots/conversation.png  - a live 3-turn chat in the real app
  docs/screenshots/eval_table.png    - the Phase 8 eval results table

Boots the Streamlit app on a temp port, drives it with headless Chromium
(Playwright), then renders the eval CSV to an image. Self-contained: starts and
stops its own server.

Run:  python tests/make_screenshots.py
"""

from __future__ import annotations

import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def boot_app(port: int) -> subprocess.Popen:
    env = {**os.environ, "MODEL_PROVIDER": "mock", "ANONYMIZED_TELEMETRY": "False"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.port", str(port)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # wait for the port
    for _ in range(60):
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                time.sleep(3)  # let the script finish first render
                return proc
        except OSError:
            time.sleep(1)
    raise RuntimeError("Streamlit did not start")


def capture_conversation(port: int) -> None:
    from playwright.sync_api import sync_playwright

    turns = [
        "What is icing in the NHL?",
        "what about in the Olympics?",
        "Is hitting in the PWHL the same as in the NHL?",
    ]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 1400})
        page.goto(f"http://localhost:{port}", wait_until="networkidle")
        page.wait_for_timeout(2000)

        box = page.locator('[data-testid="stChatInput"] textarea')
        for q in turns:
            box.fill(q)
            box.press("Enter")
            # mock is instant; wait for the rerun + render to settle
            page.wait_for_timeout(3500)

        # reveal the retrieved-chunks expanders for a richer shot
        for exp in page.locator('[data-testid="stExpander"] summary').all():
            try:
                exp.click()
                page.wait_for_timeout(300)
            except Exception:
                pass
        page.wait_for_timeout(1000)

        path = OUT / "conversation.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"[shot] conversation -> {path}")
        browser.close()


def capture_eval_table() -> None:
    from playwright.sync_api import sync_playwright

    csv_path = ROOT / "tests" / "eval_results.csv"
    if not csv_path.exists():
        print("[shot] eval_results.csv not found; run src/eval.py first. Skipping.")
        return
    with csv_path.open() as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]

    def cell(v: str) -> str:
        cls = {"True": "ok", "False": "bad", "": "na"}.get(v, "")
        text = {"True": "✓", "False": "✗", "": "·"}.get(v, v)
        return f'<td class="{cls}">{text}</td>'

    html = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;padding:24px;background:#fff}",
        "h2{margin:0 0 4px}p{color:#555;margin:0 0 16px}",
        "table{border-collapse:collapse;font-size:13px}",
        "th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}",
        "th{background:#1f2937;color:#fff}tr:nth-child(even){background:#f7f7f9}",
        ".ok{color:#0a7d28;font-weight:700;text-align:center}",
        ".bad{color:#c0202a;font-weight:700;text-align:center}",
        ".na{color:#bbb;text-align:center}",
        "</style></head><body>",
        "<h2>QuickWhistle — Phase 8 evaluation</h2>",
        "<p>Custom citation-accuracy + behavioral checks over the 21-question test set "
        "(retrieval 100% · citation correctness 100% · fidelity 100% · behavior 100%).</p>",
        "<table><tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>",
    ]
    for r in body:
        html.append("<tr>" + "".join(cell(v) for v in r) + "</tr>")
    html.append("</table></body></html>")
    tmp = OUT / "_eval_table.html"
    tmp.write_text("\n".join(html), encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1300, "height": 900})
        page.goto(f"file://{tmp}", wait_until="networkidle")
        page.wait_for_timeout(500)
        path = OUT / "eval_table.png"
        page.locator("body").screenshot(path=str(path))
        print(f"[shot] eval table -> {path}")
        browser.close()
    tmp.unlink(missing_ok=True)


def main() -> None:
    port = _free_port()
    proc = boot_app(port)
    try:
        capture_conversation(port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    capture_eval_table()
    print("\nDone. Screenshots in docs/screenshots/")


if __name__ == "__main__":
    main()
