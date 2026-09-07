"""Gemini backend for grounded generation - the free-tier alternative to Claude.

Google AI Studio has a free tier that comfortably covers evaluating the golden
set, which makes it the cheapest way to answer the question the project is stuck
on: does a model *reading* the evidence fix abstention, where distance
thresholding provably cannot?

Only the API call lives here. The prompt, the answer schema, and - most
importantly - the quote verification are shared with [generate.py](generate.py),
because those are the parts that define the grounding contract and they must not
drift between backends. A quote is checked against the filing the same way
whoever wrote it.

Get a key at https://aistudio.google.com/apikey and set GEMINI_API_KEY.
"""

from __future__ import annotations

import os
from typing import Any

from .generate import (
    SYSTEM_PROMPT,
    GroundedAnswer,
    build_prompt,
    verify_citations,
)

# Overridable because Google's model line-up moves. `list_models()` below shows
# what the key actually has access to.
DEFAULT_MODEL = os.environ.get("SEC_SIMPLIFIER_GEMINI_MODEL", "gemini-2.5-flash")
MAX_OUTPUT_TOKENS = 4000


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_available() -> bool:
    """True when the SDK is installed and a key is configured."""
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return bool(api_key())


def _client(client: Any = None):
    if client is not None:
        return client
    from google import genai

    key = api_key()
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY to a key from https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=key)


def list_models(client: Any = None) -> list[str]:
    """Model names this key can generate with - to confirm the default exists."""
    models = []
    for model in _client(client).models.list():
        actions = getattr(model, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            models.append(model.name.replace("models/", ""))
    return sorted(models)


def generate_grounded_answer(
    question: str,
    evidence: list[dict[str, Any]],
    client: Any = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Ask Gemini to answer from the evidence, then verify what it cites.

    Returns the same shape as the Claude backend, so the two are
    interchangeable wherever a generator is accepted.
    """
    from google.genai import types

    response = _client(client).models.generate_content(
        model=model,
        contents=build_prompt(question, evidence),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            # Gemini's structured output takes the same Pydantic model Claude
            # does, so both backends are constrained to an identical schema.
            response_mime_type="application/json",
            response_schema=GroundedAnswer,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            # Grounded extraction should not be creative.
            temperature=0.0,
        ),
    )

    usage = getattr(response, "usage_metadata", None)
    reported = {
        "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        "cache_read_tokens": getattr(usage, "cached_content_token_count", 0) or 0,
        "cache_write_tokens": 0,
        "model": model,
    }

    # A safety block or a truncated response is not an answer. Abstaining is the
    # honest reading of both, and matches what the app does when evidence is
    # inadequate.
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    if finish is not None and getattr(finish, "name", str(finish)) not in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
        return {
            "supported": False,
            "answer": "",
            "citations": [],
            "rejected_citations": [],
            "reason": f"The model stopped early ({getattr(finish, 'name', finish)}).",
            "usage": reported,
        }

    parsed = response.parsed
    if parsed is None:
        return {
            "supported": False,
            "answer": "",
            "citations": [],
            "rejected_citations": [],
            "reason": "The model did not return a usable answer.",
            "usage": reported,
        }

    verified, rejected = verify_citations(parsed.citations, evidence)

    # Same rule as the Claude backend: an answer whose every quote failed
    # verification is not grounded, whatever the model claimed.
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
            "usage": reported,
        }

    return {
        "supported": parsed.supported,
        "answer": parsed.answer if parsed.supported else "",
        "citations": verified if parsed.supported else [],
        "rejected_citations": rejected,
        "reason": parsed.reason_if_unsupported if not parsed.supported else "",
        "usage": reported,
    }
