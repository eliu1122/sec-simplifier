# 2026-09-06 — Week 6: evaluation, multi-company, and generation runs for real

Commits: `4b45468`, `0c30c2d`, `2f524d0`, `8e0d057`, `715a66a`, `ec0bfe4`, `f93f58e`

The longest session, and the one that changed the project's direction twice.

## The evaluation harness

`evaluate.py` scores the golden set and tracks a baseline, so a change that helps
one metric and quietly costs another is visible. Answering and declining are
reported separately, because they fail independently — and here they fail very
differently.

## What evaluation immediately found: 98% of both proxies discarded

Writing DEF 14A cases exposed silent data loss. The chunker recognised only
`Item N.` headings, and a proxy statement contains none, so both filings fell
through to a fallback that truncated at 2,000 characters.

| | before | after |
| --- | --- | --- |
| DEF 14A chunks | 2 | 110 |
| text indexed | ~2% | all of it |

Executive compensation, director biographies, auditor fees and related-party
transactions all live in the proxy. None had been reachable.

## Any company, on demand

The MVP was one hard-coded ticker. Opening it up needed scoped reads
(`where={"ticker": ...}`), two caches that would have broken, and on-demand
ingestion on a worker thread with progress polling.

Scoping matters beyond tidiness: BM25 judges term rarity *within the corpus it is
given*, so scoping keeps "rare" meaning rare in one filer's filings.

## The generalization finding

Six companies, 23 shared questions. Recall 100% everywhere. Specificity
62 / 25 / 38 / 25 / 38 / 25%.

Correlation with corpus size: **r = −0.28**. Size does not explain it. NVCT is
simply the company the thresholds were tuned on — a small clinical-stage pharma
with little surface area to match against. Two questions are correctly declined
on NVCT alone and answered by every real operating company.

Taking best vector distance per question, the should-answer and should-abstain
ranges **overlap** on five of six companies. No distance floor separates them.
Retrieval thresholding cannot deliver abstention — not "does not yet", *cannot*.

## One fix that data made possible

Diagnosing "What is the weather forecast for New Jersey?" showed the vector half
was never at fault (0.78–0.82, correctly past the floor). The **lexical gate was
the leak**: "forecast" is financial vocabulary and the state name sits in the
address block. Because the gates are OR'd, the weaker one sets the floor.

Vetoing lexical matches the embedding places far away: specificity 35% → 48%
across six companies, no recall cost, flat from 0.80 down to 0.65.

Probing question types the golden set did not cover then found the veto broke
`What does the DEF 14A say about voting?` — an exact form reference, exactly what
the lexical half exists for. Fixed with an identifier exemption, and four
`exact_identifier` cases added so the class is measured rather than remembered.

## Generation, at last

A Gemini backend was added so generation could be tested for free, sharing the
prompt, schema and quote verification with the Claude backend so the grounding
contract cannot drift.

Three bugs only running it could find, all masked by a fallback that swallowed
exceptions:

1. The SDK's `Models` helper does not keep its parent `Client` alive — a client
   left as a temporary was garbage-collected mid-request.
2. `gemini-2.5-flash` appears in `models.list()` but 404s for new keys. Listing
   is not access.
3. Free tier: 5 requests/minute, **20 per day per model**.

Adding one log line to the fallback surfaced (2) and (3) within seconds.

## The result

Three recorded gaps closed, including **both** `subject_mismatch` cases. On
Apple, the carbon-emissions question at cosine distance 0.530 — closer than
several questions that should be answered — came back:

> "the evidence mentions progress toward environmental and climate goals, but
> does not state what the specific target is."

Specificity 54% → 77%. Zero fabricated quotes.

But recall fell 17/17 → 13/17, and inspecting the failures found something more
useful than a regression: on the stock-exchange question the model abstained
because the evidence does not mention a stock exchange — **and it was right**.
Extractive mode had scored that as a success by citing ownership sections.

The metric was rewarding bluffing and penalising honesty.
