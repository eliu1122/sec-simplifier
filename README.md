# SEC Simplifier — Single-Company Due-Diligence MVP

This project is intentionally scoped to a single company, single session.
The goal is to answer natural-language questions using filing text that is
actually grounded in EDGAR documents, with visible citations and explicit
"I don't see this disclosed" behavior when a claim is unsupported.

## Scope

- One ticker only: NVCT
- Recent 10-K / 10-Q / 8-K / DEF 14A filings
- Local ingestion and metadata capture
- Section-aware chunking from filing HTML
- grounded answers with citation back to the filing section
- honest failure when the answer is not disclosed

This cuts out:

- multi-company comparison
- broad ticker coverage
- automated alerts on new filings
- heavy orchestration or generic RAG framework complexity

## Current state

The repo now has a minimal working MVP around the grounded Q&A loop:

- [src/ingest_filings.py](src/ingest_filings.py): fetches filings for the chosen ticker
- [src/sec_client.py](src/sec_client.py): SEC API client and filing metadata
- [src/grounded_qa.py](src/grounded_qa.py): section-aware chunking and grounded answer logic
- [tests/test_grounded_qa.py](tests/test_grounded_qa.py): contract tests for supported vs unsupported answers

## Run tests

```powershell
cd "C:\Users\yceri\Desktop\SEC Simplifier\sec-simplifier"
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m pytest -q tests/test_grounded_qa.py
```

## Run the demo

From the project directory, run:

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" .\run_demo.py
```

On this machine, `python` currently resolves to the Microsoft Store launcher,
so use the full interpreter path above until Python is added to your `PATH`.
After adding `C:\Users\yceri\AppData\Local\Programs\Python\Python310\` to
your user `PATH` and opening a new PowerShell window, the shorter command will
work:

```powershell
python .\run_demo.py
```

## Run the local UI

The browser UI runs entirely on your computer and uses the same local demo
corpus as `run_demo.py`; it does not make an SEC network request.

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" .\app.py
```

Then visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
Use `Ctrl+C` in PowerShell to stop the server.

## Expected behavior

- Supported question: returns answer + filing citations
- Unsupported question: returns "I don't see this disclosed in the filings available for this company."
- This is the trust signal that separates a toy demo from a real analyst tool

## Next step

Use this as the default MVP for interview demos and then expand only if needed
into multi-company compare or filing-diff workflows later. Refer to previous iterations
