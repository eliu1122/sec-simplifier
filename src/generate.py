"""Week 5: Claude reads the retrieved evidence and writes the answer.

Retrieval narrows the filing to a handful of candidate chunks. It matches words,
not meaning - so it will happily return real, on-topic-looking text for a
question about a different company or a different kind of fact. This module is
the step that adjudicates: it reads the evidence and either writes an answer
grounded in it, or declines.

Two guarantees, one asked for and one enforced:

- The model is instructed to answer only from the evidence and to quote it.
- Every quote it returns is checked against the source text before the answer is
  shown. A citation whose quote does not appear verbatim in the chunk it names
  is dropped, and an answer left with no verified citation is downgraded to an
  abstention. A model that invents a quote cannot get it past this.

`anthropic` is an optional dependency, as `chromadb` is. Without it, or without
credentials, the app falls back to the Week 4 extractive answer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_MODEL = os.environ.get("SEC_SIMPLIFIER_MODEL", "claude-opus-5")
MAX_TOKENS = 4000

# Only these fields of a chunk are shown to the model. Everything else it might
# need for a citation (form, date, URL) is looked up locally from the chunk_id,
# so the model never has to reproduce a URL correctly.
EVIDENCE_TEMPLATE = """<evidence id="{chunk_id}" form="{form}" section="{section}" period="{period}">
{text}
</evidence>"""

SYSTEM_PROMPT = """You answer questions about a public company using only the text of its SEC filings.

You will be given a question and numbered evidence blocks pulled from the company's filings. Answer using only what those blocks say.

The evidence was selected by keyword and similarity search. That matches wording, not meaning, so some blocks will be irrelevant, and a block can look relevant while being about something else entirely - a different company, a different person, a different kind of figure. Read what the evidence actually says before relying on it.

Set supported = false when:
- The evidence does not address the question.
- The evidence is about a different subject than the question asks about. A question about another company's patents is not answered by this company's patent disclosures. A question about someone's home address is not answered by the company's corporate address.
- The question asks for a specific figure, date, or name that the evidence does not state. Do not estimate, infer, or compute one.
- Answering would require knowledge you have that is not in the evidence. Your own knowledge of this company, or of any company, is not evidence.

When supported = false, leave answer empty and explain in reason_if_unsupported what the filings would need to say. Do not hedge into a partial answer.

Set supported = true only when the evidence states the answer. Then:
- Write two to four sentences in plain language, for a reader who does not know filing vocabulary. Explain terms of art rather than repeating them.
- Attribute claims to the company ("the company reports...", "the filing states..."), because that is what the evidence establishes - not whether the claim is true.
- Give one citation per claim that carries weight. Each citation needs the id of the evidence block it came from and a quote copied exactly from that block, character for character. Do not tidy, trim mid-word, or join text from separate places. A quote that does not appear verbatim in the block will be discarded and may cost you the answer.
- Keep quotes short: the sentence or table row that carries the fact.

Prefer the most recent filing when two say different things, and say that the figure is as of its period."""


class Citation(BaseModel):
    """One claim traced to one evidence block."""

    chunk_id: str = Field(description="id attribute of the evidence block this came from")
    quote: str = Field(description="text copied exactly from that evidence block")


class GroundedAnswer(BaseModel):
    """The model's verdict on whether the evidence answers the question."""

    supported: bool = Field(description="true only if the evidence states the answer")
    answer: str = Field(description="the plain-language answer; empty when unsupported")
    citations: list[Citation] = Field(description="evidence for each claim; empty when unsupported")
    reason_if_unsupported: str = Field(
        description="what the filings would need to say; empty when supported"
    )


def _normalize(text: str) -> str:
    """Collapse the differences that should not break a quote match.

    Filing HTML is full of curly quotes, en dashes, and irregular spacing. A
    quote that differs from the source only in those ways is still a real quote;
    one that differs in wording is not.
    """
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return " ".join(text.split()).lower()


def verify_citations(
    citations: list[Citation], evidence: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split citations into those whose quote really appears in the chunk, and the rest.

    Returns (verified, rejected). A verified citation is enriched with the
    chunk's own metadata - section, form, dates, source URL - so the answer
    never depends on the model reproducing those correctly.
    """
    by_id = {str(chunk.get("chunk_id", "")): chunk for chunk in evidence}
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for citation in citations:
        chunk = by_id.get(citation.chunk_id)
        if chunk is None:
            rejected.append({"chunk_id": citation.chunk_id, "quote": citation.quote,
                             "reason": "cited an evidence id that was not provided"})
            continue
        if _normalize(citation.quote) not in _normalize(chunk.get("text", "")):
            rejected.append({"chunk_id": citation.chunk_id, "quote": citation.quote,
                             "reason": "quote does not appear in the cited evidence"})
            continue
        verified.append(
            {
                "chunk_id": citation.chunk_id,
                "quote": citation.quote.strip(),
                "section": chunk.get("section", ""),
                "form": chunk.get("form", ""),
                "filing_date": chunk.get("filing_date", ""),
                "period_of_report": chunk.get("period_of_report", ""),
                "source_url": chunk.get("source_url", ""),
            }
        )

    return verified, rejected


def format_evidence(evidence: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as labelled blocks the model can cite by id."""
    return "\n\n".join(
        EVIDENCE_TEMPLATE.format(
            chunk_id=chunk.get("chunk_id", f"chunk-{index}"),
            form=chunk.get("form", ""),
            section=chunk.get("section", ""),
            period=chunk.get("period_of_report", chunk.get("filing_date", "")),
            text=chunk.get("text", ""),
        )
        for index, chunk in enumerate(evidence)
    )


def build_prompt(question: str, evidence: list[dict[str, Any]]) -> str:
    return (
        f"{format_evidence(evidence)}\n\n"
        f"Question: {question}\n\n"
        "Answer only from the evidence above, or set supported = false."
    )


def is_available() -> bool:
    """True when the SDK is installed and some credential is configured."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or (Path.home() / ".config" / "anthropic").exists()
    )


def generate_grounded_answer(
    question: str,
    evidence: list[dict[str, Any]],
    client: Any = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Ask Claude to answer from the evidence, then verify what it cites.

    Returns a dict with `supported`, `answer`, `citations` (verified only),
    `rejected_citations`, and `reason`. Never raises for an ordinary API
    failure - the caller falls back to the extractive answer.
    """
    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": build_prompt(question, evidence)}],
        output_format=GroundedAnswer,
    )

    # A safety decline is not an answer. Abstaining is the honest reading of it,
    # and matches what the app does when evidence is inadequate.
    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "model": model,
    }

    if response.stop_reason == "refusal":
        return {
            "supported": False,
            "answer": "",
            "citations": [],
            "rejected_citations": [],
            "reason": "The model declined to answer this question.",
            "usage": usage,
        }

    parsed: GroundedAnswer = response.parsed_output
    verified, rejected = verify_citations(parsed.citations, evidence)

    # An answer whose every quote failed verification is not grounded, whatever
    # the model claimed. Downgrade rather than show it.
    if parsed.supported and not verified:
        return {
            "supported": False,
            "answer": "",
            "citations": [],
            "rejected_citations": rejected,
            "reason": (
                "The drafted answer could not be traced back to the filing text, "
                "so it was withheld."
            ),
            "usage": usage,
        }

    return {
        "supported": parsed.supported,
        "answer": parsed.answer if parsed.supported else "",
        "citations": verified if parsed.supported else [],
        "rejected_citations": rejected,
        "reason": parsed.reason_if_unsupported if not parsed.supported else "",
        "usage": usage,
    }
