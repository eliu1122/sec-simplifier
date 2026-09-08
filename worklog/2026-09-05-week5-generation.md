# 2026-09-05 — Week 5: grounded generation, built but unrun

Commit: `4ddd69a` · 10 files, +1144/−78

## Built

A generation step that reads the retrieved evidence and either writes the answer
or declines — replacing the Week 4 placeholder, which returned the top chunk
truncated to 300 characters behind a template sentence.

- **Structured output** via `client.messages.parse()` with a Pydantic
  `GroundedAnswer` model, so abstention is a boolean rather than a phrase to
  pattern-match.
- **The model never handles citation metadata.** It returns an evidence block id
  and a quote; section, form, dates and source URL are looked up locally. A
  citation cannot be wrong about its own provenance.
- **Every quote is verified before display.** A quote that does not appear
  verbatim in the chunk it cites is dropped, and an answer left with no verified
  citation is downgraded to an abstention. A fabricated quote cannot reach the
  user.
- **The generator can only narrow.** It runs after retrieval and sees only
  chunks retrieval already approved.

Also deleted the hard-coded keyword guard that special-cased "lawsuit",
"settlement" and "exact" — a placeholder for judgement the model should make.
Removing it did not break the abstention test that motivated it.

## The honest caveat

**None of this ran.** There was no API key, so every test used an injected fake
or a stub client. The wiring was proven; the model's judgement was not.

This was recorded as the project's largest open risk rather than glossed, and the
commit message said so in its second paragraph — so anyone reading the history
sees the caveat before the feature.

## Lesson banked

Designing the generator as an injectable callable — `answer_question(...,
generator=...)` — made it testable without credentials, and later made adding a
second backend a 60-line change rather than a rewrite.
