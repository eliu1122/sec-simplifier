"""Local browser UI for the SEC Simplifier demo.

Run with:
    python app.py
Then open http://127.0.0.1:8000 in a browser.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.grounded_qa import answer_question

HOST = "127.0.0.1"
PORT = 8000


def demo_corpus() -> list[dict[str, str]]:
    """Return a small local corpus so the UI works without an SEC network call."""
    return [
        {
            "ticker": "NVCT",
            "form": "10-K",
            "filing_date": "2024-12-31",
            "section": "Item 1. Business",
            "text": (
                "NVCT operates a cloud analytics platform that enables customers to "
                "monitor and analyze operational performance."
            ),
            "source_url": "https://example.com/filing",
            "document_name": "filing.htm",
        },
        {
            "ticker": "NVCT",
            "form": "10-K",
            "filing_date": "2024-12-31",
            "section": "Item 7A. Market Risk",
            "text": "The company is exposed to changes in interest rates and customer concentration risk.",
            "source_url": "https://example.com/filing",
            "document_name": "filing.htm",
        },
    ]


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SEC Simplifier</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, sans-serif; color: #172033; background: #f4f7fb; }
    body { margin: 0; }
    main { max-width: 780px; margin: 0 auto; padding: 64px 24px; }
    .eyebrow { color: #4766c5; font-size: .82rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
    h1 { font-size: clamp(2rem, 5vw, 3.25rem); margin: 9px 0 12px; letter-spacing: -.04em; }
    .intro { color: #536175; line-height: 1.6; margin: 0 0 28px; }
    .card { background: white; border: 1px solid #dce3ef; border-radius: 16px; box-shadow: 0 10px 30px #1927440d; padding: 20px; }
    label { display: block; font-weight: 700; margin-bottom: 9px; }
    textarea { box-sizing: border-box; border: 1px solid #b9c5d8; border-radius: 10px; font: inherit; min-height: 105px; padding: 12px; resize: vertical; width: 100%; }
    .actions { align-items: center; display: flex; gap: 14px; justify-content: space-between; margin-top: 14px; }
    button { background: #3157c6; border: 0; border-radius: 9px; color: white; cursor: pointer; font: inherit; font-weight: 700; padding: 11px 18px; }
    button:hover { background: #2448ae; } button:disabled { opacity: .65; cursor: wait; }
    .hint { color: #68768a; font-size: .86rem; }
    #result { margin-top: 24px; } .hidden { display: none; }
    .status { border-radius: 999px; display: inline-block; font-size: .78rem; font-weight: 800; padding: 5px 10px; }
    .supported { background: #e4f7eb; color: #146b38; } .unsupported { background: #fff0e8; color: #9b3b12; }
    h2 { font-size: 1.2rem; margin: 14px 0 7px; } #answer { line-height: 1.6; margin: 0; white-space: pre-wrap; }
    ul { padding-left: 20px; } li { margin: 8px 0; } a { color: #3157c6; }
    footer { color: #68768a; font-size: .85rem; margin-top: 23px; text-align: center; }
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Local demo</div>
    <h1>SEC Simplifier</h1>
    <p class="intro">Ask a question about the available filing text. Answers include citations when the information is supported.</p>
    <section class="card">
      <form id="question-form">
        <label for="question">Your question</label>
        <textarea id="question" placeholder="What is NVCT's business?" required></textarea>
        <div class="actions"><span class="hint">Current local corpus: NVCT 2024 10-K</span><button id="submit" type="submit">Ask question</button></div>
      </form>
    </section>
    <section class="card hidden" id="result" aria-live="polite">
      <span id="status" class="status"></span>
      <h2>Answer</h2><p id="answer"></p>
      <div id="citations-wrap"><h2>Citations</h2><ul id="citations"></ul></div>
    </section>
    <footer>This interface uses the local demo corpus and does not contact the SEC.</footer>
  </main>
  <script>
    const form = document.querySelector('#question-form');
    const question = document.querySelector('#question');
    const submit = document.querySelector('#submit');
    const result = document.querySelector('#result');
    form.addEventListener('submit', async (event) => {
      event.preventDefault(); submit.disabled = true; submit.textContent = 'Searching…';
      try {
        const response = await fetch('/api/answer', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: question.value})});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Unable to answer the question.');
        const supported = data.supported;
        const status = document.querySelector('#status');
        status.textContent = supported ? 'Supported by filing text' : 'Not found in available filings';
        status.className = 'status ' + (supported ? 'supported' : 'unsupported');
        document.querySelector('#answer').textContent = data.answer;
        const citations = document.querySelector('#citations'); citations.replaceChildren();
        data.citations.forEach((citation) => { const li = document.createElement('li'); li.textContent = citation; citations.append(li); });
        document.querySelector('#citations-wrap').classList.toggle('hidden', !data.citations.length);
        result.classList.remove('hidden');
      } catch (error) { window.alert(error.message); }
      finally { submit.disabled = false; submit.textContent = 'Ask question'; }
    });
  </script>
</body>
</html>"""


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - method name required by stdlib
        if urlparse(self.path).path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send(HTTPStatus.OK, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802 - method name required by stdlib
        if urlparse(self.path).path != "/api/answer":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("Enter a question before submitting.")
            response = answer_question(question, demo_corpus())
            self._send(HTTPStatus.OK, json.dumps(response).encode("utf-8"), "application/json")
        except (ValueError, json.JSONDecodeError) as error:
            self._send(HTTPStatus.BAD_REQUEST, json.dumps({"error": str(error)}).encode("utf-8"), "application/json")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep local terminal output focused on the startup message."""


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"SEC Simplifier is running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
