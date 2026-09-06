"""Run a local SEC Simplifier UI at http://127.0.0.1:8000."""

from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.grounded_qa import answer_question

HOST, PORT = "127.0.0.1", 8000
DATA_DIR = Path(__file__).resolve().parent / "data"

# Candidates pulled from the vector store before fusion. Wider than the number
# of citations shown, so rank fusion has something to work with.
VECTOR_CANDIDATES = 10

# Companies whose tokenized corpus is kept in memory. Matches the lexical
# index cache in grounded_qa, so the two do not evict each other's work.
CORPUS_CACHE_SIZE = 8


@lru_cache(maxsize=CORPUS_CACHE_SIZE)
def company_corpus(ticker: str) -> tuple[dict, ...]:
    """Every indexed chunk for one company.

    Scoped by ticker so a question about one company can never be answered from
    another's filings. Cached because the lexical half of retrieval scores the
    whole corpus on every question.
    """
    from src.vector_store import load_all_chunks

    return tuple(load_all_chunks(ticker=ticker))


def forget_company(ticker: str) -> None:
    """Drop cached state for one company after re-indexing it."""
    company_corpus.cache_clear()


def known_companies() -> dict[str, dict]:
    try:
        from src.vector_store import indexed_companies

        return indexed_companies()
    except ImportError:
        return {}


def search_vector_store(question: str, ticker: str) -> list[dict]:
    """Similarity hits for the question within one company's filings."""
    try:
        from src.vector_store import query_vector_store

        return query_vector_store(question, limit=VECTOR_CANDIDATES, ticker=ticker)
    except ImportError:
        return []


@lru_cache(maxsize=1)
def active_generator():
    """The Claude generator, or None when the SDK or credentials are missing."""
    try:
        from src import generate

        if not generate.is_available():
            return None
        return lambda question, evidence: generate.generate_grounded_answer(
            question, evidence
        )
    except ImportError:
        return None


# --- On-demand company loading ---------------------------------------------
#
# Ingesting a company takes 30-90 seconds - SEC download time plus embedding
# several hundred chunks. That is too long for one request, so the work runs on
# a thread and the page polls for progress.

_load_lock = threading.Lock()
_load_state: dict = {"ticker": None, "message": "", "percent": 0, "done": True, "error": None}


def load_state() -> dict:
    with _load_lock:
        return dict(_load_state)


def _set_load_state(**fields) -> None:
    with _load_lock:
        _load_state.update(fields)


def start_company_load(raw_ticker: str) -> dict:
    """Begin ingesting a company on a worker thread.

    Returns the initial state. Refuses to start a second load while one is
    running, since both would be writing the same Chroma collection.
    """
    from src import company

    with _load_lock:
        if not _load_state["done"]:
            return {"error": f"Still loading {_load_state['ticker']}. One at a time."}

    ticker = company.normalize_ticker(raw_ticker)
    _set_load_state(ticker=ticker, message="Starting", percent=0, done=False, error=None)

    def run() -> None:
        try:
            company.ingest(
                ticker,
                progress=lambda message, percent: _set_load_state(
                    message=message, percent=percent
                ),
            )
            forget_company(ticker)
            _set_load_state(message=f"{ticker} is ready", percent=100, done=True)
        except Exception as error:
            _set_load_state(done=True, error=str(error), message="", percent=0)

    threading.Thread(target=run, daemon=True).start()
    return load_state()


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SEC Simplifier</title><style>
:root{font-family:system-ui,sans-serif;color:#172033;background:#f4f7fb}body{margin:0}main{max-width:780px;margin:auto;padding:56px 24px}.eyebrow{color:#4766c5;font-size:.8rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase}h1{font-size:clamp(2rem,5vw,3.2rem);margin:8px 0}.intro,.hint,.metadata,footer{color:#647287;line-height:1.5}.card{background:#fff;border:1px solid #dce3ef;border-radius:16px;box-shadow:0 10px 30px #1927440d;padding:20px;margin-top:22px}label{display:block;font-weight:700;margin-bottom:8px}textarea{box-sizing:border-box;border:1px solid #b9c5d8;border-radius:10px;font:inherit;min-height:100px;padding:12px;width:100%}.starters{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.actions{display:flex;justify-content:space-between;align-items:center;gap:12px}button{background:#3157c6;border:0;border-radius:9px;color:#fff;cursor:pointer;font:inherit;font-weight:700;padding:10px 15px}.starter{background:#eff3ff;color:#294ba9;font-size:.86rem}.hidden{display:none}.status{border-radius:99px;display:inline-block;font-size:.78rem;font-weight:800;padding:5px 10px}.supported{background:#e4f7eb;color:#146b38}.unsupported{background:#fff0e8;color:#9b3b12}.excerpt{background:#eef1f7;color:#4a5568}.evidence-card .status{margin-bottom:8px}.company-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}select,input#ticker{border:1px solid #b9c5d8;border-radius:9px;font:inherit;padding:9px 10px}select{min-width:190px}input#ticker{width:210px;text-transform:uppercase}.warn{background:#fff7ed;border-left:3px solid #d97706;border-radius:5px;color:#92400e;margin-top:10px;padding:9px 11px}.progress{margin-top:12px}.bar{background:#e6ebf4;border-radius:99px;height:8px;overflow:hidden}.fill{background:#3157c6;height:100%;width:0;transition:width .35s}.insight{background:#f3f6ff;border-left:4px solid #6382df;border-radius:6px;line-height:1.5;margin-top:16px;padding:12px}.evidence{border-top:1px solid #e4e9f2;margin-top:18px}.evidence-card{background:#fafbfd;border:1px solid #e2e7f0;border-radius:10px;margin-top:10px;padding:14px}.evidence-card p{line-height:1.55;margin:8px 0}a{color:#3157c6}footer{font-size:.85rem;margin-top:22px;text-align:center}</style></head><body><main><div class="eyebrow">Local research workspace</div><h1>SEC Simplifier</h1><p class="intro">Plain-English due diligence with evidence from company filings.</p><section class="card"><label for="ticker">Company</label><div class="company-row"><select id="company-picker"></select><input id="ticker" placeholder="or add a ticker, e.g. AAPL" maxlength="10" autocomplete="off"><button id="load" type="button">Add</button></div><div class="hint" id="company-hint"></div><div class="progress hidden" id="progress"><div class="bar"><div class="fill" id="progress-fill"></div></div><div class="hint" id="progress-text"></div></div></section><section class="card"><form id="question-form"><label for="question">Your question</label><textarea id="question" placeholder="Ask about the selected company's filings, e.g. how much cash does it have?" required></textarea><div class="starters"><button class="starter" type="button" data-question="How much cash does the company have on hand?">How much cash does it have?</button><button class="starter" type="button" data-question="What could go wrong with the business?">What are the risks?</button><button class="starter" type="button" data-question="What is the CEO annual salary?">What is the CEO paid?</button><button class="starter" type="button" data-question="Were there any related party transactions?">Any related-party deals?</button><button class="starter" type="button" data-question="What is the price of Bitcoin?">Ask something not disclosed</button></div><div class="actions"><span class="hint" id="corpus-label"></span><button id="submit" type="submit">Ask question</button></div></form></section><section class="card hidden" id="result"><span id="status" class="status"></span><h2>Answer</h2><p id="answer"></p><div class="insight"><strong>Why it matters</strong><div id="why"></div></div><div class="evidence" id="evidence-wrap"><h2 id="evidence-heading">Evidence from the filing</h2><div id="evidence"></div></div></section><footer>Verify answers against the cited filing before making an investment decision.</footer></main><script>
const question=document.querySelector('#question'),submit=document.querySelector('#submit'),result=document.querySelector('#result');
const picker=document.querySelector('#company-picker'),tickerBox=document.querySelector('#ticker'),loadBtn=document.querySelector('#load');
const progress=document.querySelector('#progress'),fill=document.querySelector('#progress-fill'),progressText=document.querySelector('#progress-text');
let mode='';
function currentTicker(){return picker.value}
async function refreshCompanies(select){
  const d=await(await fetch('/api/companies')).json();mode=d.mode;
  picker.replaceChildren();
  d.companies.forEach(c=>{const o=document.createElement('option');o.value=c.ticker;o.textContent=c.ticker+'  ('+c.filings+' filings, '+c.chunks+' sections)';picker.append(o)});
  if(select)picker.value=select;
  const has=d.companies.length>0;
  submit.disabled=!has;question.disabled=!has;
  picker.style.display=has?'':'none';
  // The ticker box stays usable even without a User-Agent: clicking Add then
  // explains what to set, which beats a dead input and a hint you might miss.
  const hint=document.querySelector('#company-hint');
  hint.classList.toggle('warn',!d.can_load);
  hint.textContent=d.can_load
    ?(has?'Pick a company, or add another by ticker.':'No companies loaded yet - add a ticker to begin.')
    :'To add companies, set SEC_USER_AGENT to your name and email, then restart the app. The SEC requires it.';
  document.querySelector('#corpus-label').textContent=has?(currentTicker()+' - '+mode):'';
}
picker.onchange=()=>{document.querySelector('#corpus-label').textContent=currentTicker()+' - '+mode;result.classList.add('hidden')};
async function pollLoad(){
  const s=await(await fetch('/api/company/status')).json();
  fill.style.width=(s.percent||0)+'%';progressText.textContent=s.message||'';
  if(!s.done){setTimeout(pollLoad,700);return}
  loadBtn.disabled=false;loadBtn.textContent='Add';
  if(s.error){progress.classList.add('hidden');window.alert(s.error);return}
  progressText.textContent=s.message;await refreshCompanies(s.ticker);
  document.querySelector('#corpus-label').textContent=currentTicker()+' - '+mode;
  setTimeout(()=>progress.classList.add('hidden'),1500);
}
loadBtn.onclick=async()=>{
  const t=tickerBox.value.trim().toUpperCase();if(!t)return;
  loadBtn.disabled=true;loadBtn.textContent='Loading...';
  progress.classList.remove('hidden');fill.style.width='0%';progressText.textContent='Starting';
  try{const r=await fetch('/api/company',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker:t})});
    const d=await r.json();if(!r.ok)throw Error(d.error||'Could not load that company.');
    tickerBox.value='';pollLoad();
  }catch(err){loadBtn.disabled=false;loadBtn.textContent='Add';progress.classList.add('hidden');window.alert(err.message)}
};
tickerBox.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();loadBtn.click()}};
refreshCompanies();
document.querySelectorAll('.starter').forEach(b=>b.onclick=()=>{question.value=b.dataset.question;question.focus()});
document.querySelector('#question-form').onsubmit=async e=>{e.preventDefault();submit.disabled=true;submit.textContent='Searching...';try{const r=await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:question.value,ticker:currentTicker()})}),d=await r.json();if(!r.ok)throw Error(d.error||'Unable to answer.');const tag=document.querySelector('#status');tag.textContent=d.supported?'Supported by filing text':'Not found in loaded filings';tag.className='status '+(d.supported?'supported':'unsupported');document.querySelector('#answer').textContent=d.answer;document.querySelector('#why').textContent=d.why_it_matters;const evidence=document.querySelector('#evidence');evidence.replaceChildren();document.querySelector('#evidence-heading').textContent=d.generated?'Quotes from the filing':'Evidence from the filing';d.evidence.forEach(item=>{const card=document.createElement('article');card.className='evidence-card';const title=document.createElement('strong');title.textContent=item.section;const meta=document.createElement('div');meta.className='metadata';meta.textContent=item.form+' | filed '+item.filing_date;const excerpt=document.createElement('p');excerpt.textContent='"'+item.excerpt+'"';const badge=document.createElement('span');badge.className='status '+(d.generated?'supported':'excerpt');badge.textContent=d.generated?'Verified quote':'Filing excerpt';badge.title=d.generated?'Checked to appear word for word in the filing section above.':'A passage from the cited section, shown as-is.';const link=document.createElement('a');link.href=item.source_url;link.target='_blank';link.rel='noreferrer';link.textContent='Open source filing';card.append(title,meta,excerpt,badge,document.createElement('br'),link);evidence.append(card)});document.querySelector('#evidence-wrap').classList.toggle('hidden',!d.evidence.length);result.classList.remove('hidden')}catch(err){window.alert(err.message)}finally{submit.disabled=false;submit.textContent='Ask question'}};
</script></body></html>"""


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/companies":
            companies = known_companies()
            mode = (
                "answers written from cited evidence"
                if active_generator()
                else "answers excerpted from filings"
            )
            self._json(
                HTTPStatus.OK,
                {
                    "companies": sorted(companies.values(), key=lambda c: c["ticker"]),
                    "mode": mode,
                    "can_load": bool(os.environ.get("SEC_USER_AGENT")),
                },
            )
        elif path == "/api/company/status":
            self._json(HTTPStatus.OK, load_state())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/company":
            self._load_company()
        elif path == "/api/answer":
            self._answer()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _body(self) -> dict:
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        return json.loads(raw.decode())

    def _load_company(self) -> None:
        from src.company import CompanyError

        try:
            ticker = str(self._body().get("ticker", "")).strip()
            if not ticker:
                raise ValueError("Enter a ticker symbol.")
            state = start_company_load(ticker)
            if state.get("error"):
                self._json(HTTPStatus.CONFLICT, state)
            else:
                self._json(HTTPStatus.ACCEPTED, state)
        except (CompanyError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _answer(self) -> None:
        try:
            payload = self._body()
            question = str(payload.get("question", "")).strip()
            ticker = str(payload.get("ticker", "")).strip().upper()
            if not question:
                raise ValueError("Enter a question before submitting.")
            if not ticker:
                raise ValueError("Choose a company first.")

            corpus = company_corpus(ticker)
            if not corpus:
                raise ValueError(f"{ticker} is not loaded yet.")

            vector_hits = search_vector_store(question, ticker)
            self._json(
                HTTPStatus.OK,
                answer_question(
                    question, list(corpus), vector_hits, generator=active_generator()
                ),
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
