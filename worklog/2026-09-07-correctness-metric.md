# 2026-09-07 — Honest reporting, and the correctness metric

Commits: `cfcaecd`, `6d10847`, `388f293`

## The status line was reporting configuration, not reality

It said "answers written by gemini" whenever a key was present — and kept saying
it after the daily quota was exhausted, while serving extractive excerpts. In a
project whose premise is not overstating what it knows, that was the wrong bug to
leave in.

`answer_question` now returns `generated` and `generation_error` on every
response, and the page reports what actually happened:

| situation | shown |
| --- | --- |
| on load | `generation enabled (gemini gemini-3.6-flash)` |
| after a generated answer | `last answer written by gemini gemini-3.6-flash` |
| quota exhausted | `last answer excerpted - daily free-tier quota reached` |
| bad key | `last answer excerpted - the API key was rejected` |
| model overloaded | `last answer excerpted - the model is temporarily unavailable` |

Degraded states render amber. The load-time wording is deliberately *intent*
("generation enabled") rather than fact, since a configured backend can still be
out of quota.

## The correctness metric

Ten cases now record `expect_answer_contains` — facts, any one of which must
appear in the answer. Each was checked against NVCT's filings before being
asserted. The CEO-salary case was dropped: only RSA grants are disclosed, not a
salary figure, so the check would have tested nothing.

Fixing the metric **inverted the comparison**:

| | answered | stated the fact |
| --- | --- | --- |
| extractive | 10/10 · 100% | 4/10 · 40% |
| generation | 7/10 · 70% | 6/7 · 86% |

Counted as what a reader receives:

| | correct | confidently wrong | honestly declined |
| --- | --- | --- | --- |
| extractive | 4 | **6** | 0 |
| generation | **6** | 1 | 3 |

The old scoring preferred extractive because it only counted whether an answer
came back. Six extractive "successes" never contained what was asked — no cash
figure, no mention of losses, never naming NXP900, never saying NASDAQ.

All three questions generation declines turned out to be **retrieval failures**,
checked against the evidence actually passed to the model. The board-of-directors
case is the sharpest: retrieval returned governance *policy* text plus a
552-character fragment, with exactly one director's name in the whole of it — yet
the case asserts `expect_section_contains: CORPORATE GOVERNANCE` and retrieval
**hit that section**. A section match is not correctness.

## The 7-hour run

A 30-case evaluation ran for **7.3 hours** and produced 22 bytes. My fault, and
introduced in the same change where I thought I was fixing reliability.

The `google-genai` SDK already retries 408/429/5xx **five times** with up to 60s
delays. Layered under a second retry, that is up to 25 requests per question with
compounding backoff — and with no timeout set, one stuck connection blocked
indefinitely.

Fixed: `REQUEST_TIMEOUT_MS` bounds a single call, and `HttpRetryOptions(attempts=1)`
turns the SDK's retry off so exactly one layer owns the policy — the local one,
because it reads the delay the API asks for out of the error body rather than
guessing.

Both pinned by a test that explains why, so a second layer is not re-added.

## `PARTIAL RUN`

`evaluate.py` now refuses to publish a run where some calls never reached the
model:

```
PARTIAL RUN: generation reached 13/30 cases;
17 fell back to extractive answers. These figures mix both modes
and measure neither. Re-run before trusting them.
  cause: the model is temporarily unavailable
```

This warning was written a commit earlier and **never fired** — the patch matched
a string containing a newline escape and silently did not apply. Which is itself
an instance of the pattern it exists to catch.

## Still not done

A clean full 30-case generation run. Three attempts, three different failures:
daily quota, a 503 storm, then the hang above.
