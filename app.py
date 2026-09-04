"""Run a local SEC Simplifier UI at http://127.0.0.1:8000."""

from __future__ import annotations

import json
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.corpus import load_local_corpus
from src.grounded_qa import answer_question

HOST, PORT = "127.0.0.1", 8000
DATA_DIR = Path(__file__).resolve().parent / "data"

# Candidates pulled from the vector store before fusion. Wider than the number
# of citations shown, so rank fusion has something to work with.
VECTOR_CANDIDATES = 10


def demo_corpus() -> list[dict[str, str]]:
    return [
        {"ticker": "NVCT", "form": "10-K", "filing_date": "2024-12-31", "section": "Item 1. Business", "text": "NVCT operates a cloud analytics platform that enables customers to monitor and analyze operational performance.", "source_url": "https://example.com/filing", "document_name": "filing.htm"},
        {"ticker": "NVCT", "form": "10-K", "filing_date": "2024-12-31", "section": "Item 7A. Market Risk", "text": "The company is exposed to changes in interest rates and customer concentration risk.", "source_url": "https://example.com/filing", "document_name": "filing.htm"},
    ]


@lru_cache(maxsize=1)
def active_corpus() -> tuple[tuple[dict, ...], str, bool]:
    """Load the answerable corpus once, preferring the persisted vector store.

    Parsing every filing's HTML takes seconds, so this is cached for the life of
    the process. Restart the server after re-ingesting or rebuilding the index.
    """
    try:
        from src.vector_store import load_all_chunks

        indexed = load_all_chunks()
    except ImportError:
        indexed = []

    if indexed:
        tickers = ", ".join(sorted({chunk["ticker"] for chunk in indexed}))
        label = f"Hybrid search over {tickers} ({len(indexed)} indexed chunks)"
        return tuple(indexed), label, True

    corpus = load_local_corpus(DATA_DIR)
    if corpus:
        tickers = ", ".join(sorted({chunk["ticker"] for chunk in corpus}))
        label = f"Keyword search over {tickers} ({len(corpus)} sections) - run src.build_index for hybrid"
        return tuple(corpus), label, False

    return tuple(demo_corpus()), "Demo corpus: NVCT 2024 10-K", False


def search_vector_store(question: str, enabled: bool) -> list[dict]:
    """Similarity hits for the question, or nothing if the store is unavailable."""
    if not enabled:
        return []
    try:
        from src.vector_store import query_vector_store

        return query_vector_store(question, limit=VECTOR_CANDIDATES)
    except ImportError:
        return []


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SEC Simplifier</title><style>
:root{font-family:system-ui,sans-serif;color:#172033;background:#f4f7fb}body{margin:0}main{max-width:780px;margin:auto;padding:56px 24px}.eyebrow{color:#4766c5;font-size:.8rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase}h1{font-size:clamp(2rem,5vw,3.2rem);margin:8px 0}.intro,.hint,.metadata,footer{color:#647287;line-height:1.5}.card{background:#fff;border:1px solid #dce3ef;border-radius:16px;box-shadow:0 10px 30px #1927440d;padding:20px;margin-top:22px}label{display:block;font-weight:700;margin-bottom:8px}textarea{box-sizing:border-box;border:1px solid #b9c5d8;border-radius:10px;font:inherit;min-height:100px;padding:12px;width:100%}.starters{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.actions{display:flex;justify-content:space-between;align-items:center;gap:12px}button{background:#3157c6;border:0;border-radius:9px;color:#fff;cursor:pointer;font:inherit;font-weight:700;padding:10px 15px}.starter{background:#eff3ff;color:#294ba9;font-size:.86rem}.hidden{display:none}.status{border-radius:99px;display:inline-block;font-size:.78rem;font-weight:800;padding:5px 10px}.supported{background:#e4f7eb;color:#146b38}.unsupported{background:#fff0e8;color:#9b3b12}.insight{background:#f3f6ff;border-left:4px solid #6382df;border-radius:6px;line-height:1.5;margin-top:16px;padding:12px}.evidence{border-top:1px solid #e4e9f2;margin-top:18px}.evidence-card{background:#fafbfd;border:1px solid #e2e7f0;border-radius:10px;margin-top:10px;padding:14px}.evidence-card p{line-height:1.55;margin:8px 0}a{color:#3157c6}footer{font-size:.85rem;margin-top:22px;text-align:center}</style></head><body><main><div class="eyebrow">Local research workspace</div><h1>SEC Simplifier</h1><p class="intro">Plain-English due diligence with evidence from company filings.</p><section class="card"><form id="question-form"><label for="question">Your question</label><textarea id="question" placeholder="What is NVCT's business?" required></textarea><div class="starters"><button class="starter" type="button" data-question="What is NVCT's business?">How does it make money?</button><button class="starter" type="button" data-question="What market risks does NVCT disclose?">What are the risks?</button></div><div class="actions"><span class="hint" id="corpus-label"></span><button id="submit" type="submit">Ask question</button></div></form></section><section class="card hidden" id="result"><span id="status" class="status"></span><h2>Answer</h2><p id="answer"></p><div class="insight"><strong>Why it matters</strong><div id="why"></div></div><div class="evidence" id="evidence-wrap"><h2>Evidence from the filing</h2><div id="evidence"></div></div></section><footer>Verify answers against the cited filing before making an investment decision.</footer></main><script>
const question=document.querySelector('#question'),submit=document.querySelector('#submit'),result=document.querySelector('#result');
fetch('/api/status').then(r=>r.json()).then(d=>document.querySelector('#corpus-label').textContent=d.label);
document.querySelectorAll('.starter').forEach(b=>b.onclick=()=>{question.value=b.dataset.question;question.focus()});
document.querySelector('#question-form').onsubmit=async e=>{e.preventDefault();submit.disabled=true;submit.textContent='Searching...';try{const r=await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:question.value})}),d=await r.json();if(!r.ok)throw Error(d.error||'Unable to answer.');const tag=document.querySelector('#status');tag.textContent=d.supported?'Supported by filing text':'Not found in loaded filings';tag.className='status '+(d.supported?'supported':'unsupported');document.querySelector('#answer').textContent=d.answer;document.querySelector('#why').textContent=d.why_it_matters;const evidence=document.querySelector('#evidence');evidence.replaceChildren();d.evidence.forEach(item=>{const card=document.createElement('article');card.className='evidence-card';const title=document.createElement('strong');title.textContent=item.section;const meta=document.createElement('div');meta.className='metadata';meta.textContent=item.form+' | filed '+item.filing_date;const excerpt=document.createElement('p');excerpt.textContent='"'+item.excerpt+'"';const link=document.createElement('a');link.href=item.source_url;link.target='_blank';link.rel='noreferrer';link.textContent='Open source filing';card.append(title,meta,excerpt,link);evidence.append(card)});document.querySelector('#evidence-wrap').classList.toggle('hidden',!d.evidence.length);result.classList.remove('hidden')}catch(err){window.alert(err.message)}finally{submit.disabled=false;submit.textContent='Ask question'}};
</script></body></html>"""


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/status":
            _, label, _ = active_corpus()
            self._json(HTTPStatus.OK, {"label": label})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/answer":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            question = str(json.loads(body.decode())["question"]).strip()
            if not question:
                raise ValueError("Enter a question before submitting.")
            corpus, _, vector_ready = active_corpus()
            vector_hits = search_vector_store(question, vector_ready)
            self._json(
                HTTPStatus.OK, answer_question(question, list(corpus), vector_hits)
            )
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"SEC Simplifier is running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
