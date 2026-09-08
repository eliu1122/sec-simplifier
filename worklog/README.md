# Worklog

A session-by-session account of how this project was built, and what changed its
direction. [PROGRESS.md](../PROGRESS.md) records the *state* of the system;
these entries record the *work* — including the mistakes, because several of them
taught more than the successes did.

| Session | Focus | Entry |
| --- | --- | --- |
| 2026-08-30 | Initial MVP: EDGAR ingestion, chunking, lexical grounding | (pre-dates this log) |
| 2026-09-03 | Week 3 — vector store and metadata | [2026-09-03-week3-vector-store.md](2026-09-03-week3-vector-store.md) |
| 2026-09-04 | Week 4 — hybrid retrieval, table-aware chunking | [2026-09-04-week4-hybrid-retrieval.md](2026-09-04-week4-hybrid-retrieval.md) |
| 2026-09-05 | Week 5 — grounded generation, built but unrun | [2026-09-05-week5-generation.md](2026-09-05-week5-generation.md) |
| 2026-09-06 | Week 6 — evaluation, multi-company, generation runs for real | [2026-09-06-week6-evaluation-and-generation.md](2026-09-06-week6-evaluation-and-generation.md) |
| 2026-09-07 | Honest reporting, and the correctness metric | [2026-09-07-correctness-metric.md](2026-09-07-correctness-metric.md) |

## The through-line

Four separate times, a silent fallback turned a failure into a plausible-looking
result: a broken API client, an exhausted quota, a 503 storm, and a retry layer
stacked on a retry layer. Each cost a full evaluation run, and two nearly went
into the documentation as findings.

The project's thesis is that a system should say when it does not know. It turned
out the **measurement harness** needed that discipline at least as much as the
app did. That is why `PARTIAL RUN` exists, why the status line reports what ran
rather than what was configured, and why generation failures are logged rather
than swallowed.
