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

import logging
import os
import re
import threading
import time
from typing import Any

from .generate import (
    SYSTEM_PROMPT,
    GroundedAnswer,
    build_prompt,
    verify_citations,
)

# Overridable because Google's model line-up moves quickly. `list_models()`
# below shows what the key can see - though note that listing is not the same as
# access: gemini-2.5-flash still appears there but returns 404 for keys created
# after it was retired to existing users.
DEFAULT_MODEL = os.environ.get("SEC_SIMPLIFIER_GEMINI_MODEL", "gemini-3.6-flash")
MAX_OUTPUT_TOKENS = 4000

# AI Studio's free tier is what this backend exists for, so a question costs
# nothing. Reporting Claude's rates here - as the shared cost line originally
# did - would invent a charge that was never made.
COST_PER_INPUT_TOKEN = 0.0
COST_PER_OUTPUT_TOKEN = 0.0

# The free tier allows a handful of requests per minute per model, so an
# evaluation run of 30 questions will hit the limit within seconds if left
# unthrottled. A rate-limited request is not a failed one - falling back to an
# extractive answer would quietly corrupt a whole evaluation with results that
# never reached the model. So: pace requests, and wait when told to.
MIN_SECONDS_BETWEEN_REQUESTS = float(os.environ.get("SEC_SIMPLIFIER_GEMINI_INTERVAL", "13"))
MAX_RETRIES = 4
FALLBACK_RETRY_SECONDS = 30.0

# A single call must not be able to hang. Without this the SDK waits
# indefinitely, and one stuck connection turned a 30-case run into a 7-hour one.
REQUEST_TIMEOUT_MS = 25_000

# Ceiling on the wall-clock time one question may spend retrying. Without it a
# degraded service costs ~7 minutes per question - five attempts each waiting
# out the full request timeout, plus backoff - and a 30-case run grinds for an
# hour producing nothing.
RETRY_BUDGET_SECONDS = 90.0

# The SDK retries 408/429/5xx five times on its own, with up to 60s between
# attempts. Layered under the retry below that is 25 requests per question with
# compounding backoff. Exactly one layer should own the policy, and it is the
# one here, because it reads the delay the API asks for out of the error body
# rather than guessing at exponential backoff.
SDK_ATTEMPTS = 1

_throttle = threading.Lock()
_last_request_at = 0.0

_RETRY_DELAY = re.compile(r"retry in ([0-9.]+)s", re.I)
logger = logging.getLogger(__name__)


def _wait_turn() -> None:
    """Space requests far enough apart to stay inside the free-tier limit."""
    global _last_request_at
    with _throttle:
        gap = time.monotonic() - _last_request_at
        if _last_request_at and gap < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - gap)
        _last_request_at = time.monotonic()


def _retry_after(error: Exception) -> float:
    """Seconds the API asked us to wait, or a safe default."""
    match = _RETRY_DELAY.search(str(error))
    if match:
        return min(float(match.group(1)) + 1.0, 120.0)
    # 503s carry no retry hint and usually clear quickly; a rate limit needs
    # longer. Back off further on each attempt either way.
    return 8.0 if "503" in str(error) or "UNAVAILABLE" in str(error) else FALLBACK_RETRY_SECONDS


def _is_exhausted_for_today(error: Exception) -> bool:
    """A per-day quota cannot come back during a run.

    Retrying it burns minutes of backoff to fail identically. The run should
    fall back immediately and let the harness report how far it got.
    """
    text = str(error)
    return "PerDay" in text or "per day" in text.lower()


def _is_retryable(error: Exception) -> bool:
    """Transient conditions worth waiting out rather than failing on.

    503 UNAVAILABLE means the model is busy, not that anything is wrong with
    the request - it accounted for 17 of 30 cases in one evaluation run, every
    one of which silently became an extractive answer.
    """
    if _is_exhausted_for_today(error):
        return False
    text = str(error)
    return any(
        marker in text
        # 504 / DEADLINE_EXCEEDED is the gateway giving up on a slow model, and
        # is what a request that would otherwise hang now surfaces as.
        for marker in (
            "429", "RESOURCE_EXHAUSTED",
            "503", "UNAVAILABLE", "overloaded",
            "504", "DEADLINE_EXCEEDED",
        )
    )


def _generate_with_retry(active: Any, throttle: bool = True, **kwargs):
    """Call the model, waiting out rate limits rather than treating them as errors.

    `throttle` is off when the caller supplied its own client - a stub in tests
    has no quota to respect, and pacing it would make the suite sleep for
    minutes.
    """
    deadline = time.monotonic() + RETRY_BUDGET_SECONDS
    for attempt in range(MAX_RETRIES + 1):
        if throttle:
            _wait_turn()
        try:
            return active.models.generate_content(**kwargs)
        except Exception as error:
            if not _is_retryable(error) or attempt == MAX_RETRIES:
                raise
            delay = _retry_after(error) * (attempt + 1)
            if time.monotonic() + delay > deadline:
                logger.warning("Retry budget spent on this question; giving up")
                raise
            logger.info("Transient failure; waiting %.0fs before retry %d", delay, attempt + 1)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_available() -> bool:
    """True when the SDK is installed and a key is configured."""
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return bool(api_key())


_DEFAULT_CLIENT: Any = None


def _client(client: Any = None):
    """Return the shared client, creating it once.

    It must be held somewhere: the SDK's `Models` helper does not keep the
    parent client alive, so a client left as a temporary is garbage-collected -
    closing its HTTP connection - before the request completes. Caching it also
    reuses the connection across questions.
    """
    if client is not None:
        return client

    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        from google import genai

        key = api_key()
        if not key:
            raise RuntimeError(
                "Set GEMINI_API_KEY to a key from https://aistudio.google.com/apikey"
            )
        from google.genai import types

        _DEFAULT_CLIENT = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=SDK_ATTEMPTS),
            ),
        )
    return _DEFAULT_CLIENT


def list_models(client: Any = None) -> list[str]:
    """Model names this key can generate with - to confirm the default exists."""
    # Bind the client to a name: `.list()` returns a lazy pager, and a client
    # left as a temporary is closed before the pages are fetched.
    active = _client(client)
    models = []
    for model in active.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            models.append((model.name or "").replace("models/", ""))
    return sorted(name for name in models if name)


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

    active = _client(client)
    response = _generate_with_retry(
        active,
        throttle=client is None,
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
