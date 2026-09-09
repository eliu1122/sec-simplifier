"""Tests for the Gemini backend and the backend selector.

No network. The Gemini client is stubbed, so the request shape, the shared
verification, and the failure paths are all exercised offline - the same
treatment the Claude backend gets.

The point of most of these is that the two backends must stay interchangeable:
same answer shape, same abstention rules, same quote checking.
"""

from __future__ import annotations

import pytest

from src.generate import Citation, GroundedAnswer
from src.generate_gemini import generate_grounded_answer
from tests.test_generate import EVIDENCE


class _StubUsage:
    prompt_token_count = 3100
    candidates_token_count = 240
    cached_content_token_count = 0


class _StubCandidate:
    def __init__(self, finish="STOP"):
        class Reason:
            name = finish

        self.finish_reason = Reason()


class _StubResponse:
    def __init__(self, parsed, finish="STOP"):
        self.parsed = parsed
        self.candidates = [_StubCandidate(finish)]
        self.usage_metadata = _StubUsage()


class _StubModels:
    def __init__(self, parsed, finish="STOP"):
        self._parsed = parsed
        self._finish = finish
        self.request = None

    def generate_content(self, **kwargs):
        self.request = kwargs
        return _StubResponse(self._parsed, self._finish)


class _StubClient:
    def __init__(self, parsed, finish="STOP"):
        self.models = _StubModels(parsed, finish)


def _answer(**kwargs):
    return GroundedAnswer(**kwargs)


REAL_QUOTE = "fund operations into the second quarter of 2027"


# --- Request shape ----------------------------------------------------------


def test_request_constrains_output_to_the_shared_schema():
    from src.generate_gemini import DEFAULT_MODEL

    client = _StubClient(_answer(supported=False, answer="", citations=[], reason_if_unsupported="no"))
    generate_grounded_answer("What is the cash runway?", EVIDENCE, client=client)

    request = client.models.request
    assert request["model"] == DEFAULT_MODEL
    config = request["config"]
    # The same Pydantic model the Claude backend uses, so both are constrained
    # identically rather than merely similarly.
    assert config.response_schema is GroundedAnswer
    assert config.response_mime_type == "application/json"
    assert "SEC filings" in config.system_instruction
    # Grounded extraction should not be creative.
    assert config.temperature == 0.0
    assert "acc:liquidity:0" in request["contents"]


# --- Verification is shared, not reimplemented ------------------------------


def test_fabricated_quotes_are_rejected_here_too():
    client = _StubClient(
        _answer(
            supported=True,
            answer="The company reports cash into Q2 2027.",
            citations=[
                Citation(chunk_id="acc:liquidity:0", quote=REAL_QUOTE),
                Citation(chunk_id="acc:liquidity:0", quote="a sentence never written"),
            ],
            reason_if_unsupported="",
        )
    )

    result = generate_grounded_answer("q", EVIDENCE, client=client)

    assert result["supported"] is True
    assert len(result["citations"]) == 1
    assert len(result["rejected_citations"]) == 1


def test_an_answer_with_no_surviving_quote_is_downgraded():
    client = _StubClient(
        _answer(
            supported=True,
            answer="The company earned $50 million.",
            citations=[Citation(chunk_id="acc:liquidity:0", quote="revenue of $50 million")],
            reason_if_unsupported="",
        )
    )

    result = generate_grounded_answer("q", EVIDENCE, client=client)

    assert result["supported"] is False
    assert result["citations"] == []
    assert "could not be traced" in result["reason"]


# --- Failure paths ----------------------------------------------------------


@pytest.mark.parametrize("finish", ["SAFETY", "MAX_TOKENS", "PROHIBITED_CONTENT", "RECITATION"])
def test_stopping_early_becomes_an_abstention(finish):
    """A blocked or truncated response is not an answer."""
    client = _StubClient(
        _answer(supported=True, answer="x", citations=[], reason_if_unsupported=""),
        finish=finish,
    )

    result = generate_grounded_answer("q", EVIDENCE, client=client)

    assert result["supported"] is False
    assert result["citations"] == []


def test_unparseable_output_becomes_an_abstention():
    client = _StubClient(None)

    result = generate_grounded_answer("q", EVIDENCE, client=client)

    assert result["supported"] is False
    assert "usable answer" in result["reason"]


def test_usage_is_reported_in_the_shared_shape():
    client = _StubClient(_answer(supported=False, answer="", citations=[], reason_if_unsupported="no"))

    usage = generate_grounded_answer("q", EVIDENCE, client=client)["usage"]

    assert usage["input_tokens"] == 3100
    assert usage["output_tokens"] == 240
    assert "model" in usage


# --- Rate limiting ----------------------------------------------------------
#
# The free tier allows a few requests per minute. A rate-limited request is not
# a failed one: treating it as failure silently corrupted an entire evaluation
# run with extractive answers that never reached the model.


class _RateLimited(Exception):
    def __init__(self, retry_seconds=None):
        message = "429 RESOURCE_EXHAUSTED. Quota exceeded"
        if retry_seconds is not None:
            message += f". Please retry in {retry_seconds}s"
        super().__init__(message)


class _FlakyModels:
    """Fails with 429 a set number of times, then succeeds."""

    def __init__(self, failures, parsed, retry_seconds=None):
        self.remaining = failures
        self._parsed = parsed
        self._retry_seconds = retry_seconds
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise _RateLimited(self._retry_seconds)
        return _StubResponse(self._parsed)


class _FlakyClient:
    def __init__(self, failures, parsed, retry_seconds=None):
        self.models = _FlakyModels(failures, parsed, retry_seconds)


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr("src.generate_gemini.time.sleep", lambda s: slept.append(s))
    return slept


def test_a_rate_limited_request_is_retried_not_failed(no_sleep):
    parsed = _answer(supported=False, answer="", citations=[], reason_if_unsupported="no")
    client = _FlakyClient(failures=2, parsed=parsed)

    result = generate_grounded_answer("q", EVIDENCE, client=client)

    assert client.models.calls == 3
    assert result["supported"] is False
    assert "usable answer" not in result["reason"]


def test_the_delay_the_api_asks_for_is_honoured(no_sleep):
    parsed = _answer(supported=False, answer="", citations=[], reason_if_unsupported="no")
    client = _FlakyClient(failures=1, parsed=parsed, retry_seconds="32.3")

    generate_grounded_answer("q", EVIDENCE, client=client)

    assert no_sleep and 32 < no_sleep[0] < 35


def test_persistent_rate_limiting_eventually_raises(no_sleep):
    """Better a visible error than a run full of silent extractive fallbacks."""
    from src.generate_gemini import MAX_RETRIES

    parsed = _answer(supported=False, answer="", citations=[], reason_if_unsupported="no")
    client = _FlakyClient(failures=MAX_RETRIES + 1, parsed=parsed)

    with pytest.raises(Exception, match="429"):
        generate_grounded_answer("q", EVIDENCE, client=client)


def test_a_non_rate_limit_error_is_not_retried(no_sleep):
    class _Boom:
        def generate_content(self, **kwargs):
            raise RuntimeError("400 INVALID_ARGUMENT")

    class _Client:
        models = _Boom()

    with pytest.raises(RuntimeError, match="400"):
        generate_grounded_answer("q", EVIDENCE, client=_Client())
    assert no_sleep == []


def test_an_injected_client_is_not_throttled(monkeypatch):
    """A stub has no quota; pacing it would make the suite sleep for minutes."""
    waits = []
    monkeypatch.setattr("src.generate_gemini._wait_turn", lambda: waits.append(1))

    client = _StubClient(_answer(supported=False, answer="", citations=[], reason_if_unsupported="no"))
    generate_grounded_answer("q", EVIDENCE, client=client)

    assert waits == []


# --- The selector -----------------------------------------------------------


def test_no_keys_means_no_generator(monkeypatch):
    from src import backend

    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "SEC_SIMPLIFIER_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)

    assert backend.get_generator() is None
    assert "excerpted" in backend.describe()


def test_gemini_is_selected_when_only_its_key_is_set(monkeypatch):
    from src import backend

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "SEC_SIMPLIFIER_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    name, module = backend.active_backend()

    assert name == "gemini"
    assert backend.get_generator() is not None
    assert "gemini" in backend.describe()


def test_the_backend_can_be_forced_off(monkeypatch):
    from src import backend

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("SEC_SIMPLIFIER_BACKEND", "none")

    assert backend.get_generator() is None


def test_forcing_a_backend_ignores_the_other(monkeypatch):
    from src import backend

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("SEC_SIMPLIFIER_BACKEND", "gemini")

    name, _ = backend.active_backend()
    assert name == "gemini"


def test_both_backends_return_the_same_keys(monkeypatch):
    """They are interchangeable, so callers must not need to know which ran."""
    from src.generate import generate_grounded_answer as claude_generate
    from tests.test_generate import _StubClient as ClaudeStub

    parsed = _answer(
        supported=True,
        answer="The company reports cash into Q2 2027.",
        citations=[Citation(chunk_id="acc:liquidity:0", quote=REAL_QUOTE)],
        reason_if_unsupported="",
    )

    claude = claude_generate("q", EVIDENCE, client=ClaudeStub(parsed))
    gemini = generate_grounded_answer("q", EVIDENCE, client=_StubClient(parsed))

    assert set(claude) == set(gemini)
    assert claude["supported"] == gemini["supported"] is True
    assert claude["citations"][0]["quote"] == gemini["citations"][0]["quote"]


# --- Transient failures are waited out, not counted as answers --------------


class _Overloaded(Exception):
    def __init__(self):
        super().__init__("503 UNAVAILABLE. This model is currently experiencing high demand.")


def test_a_503_is_retried_like_a_rate_limit(no_sleep):
    """17 of 30 cases in one run were 503s, each silently becoming an excerpt."""
    from src.generate_gemini import _is_retryable

    assert _is_retryable(_Overloaded())

    parsed = _answer(supported=False, answer="", citations=[], reason_if_unsupported="no")
    result = generate_grounded_answer("q", EVIDENCE, client=_FlakyClient(failures=2, parsed=parsed))

    assert result["supported"] is False


def test_retry_delays_grow_with_each_attempt(no_sleep):
    parsed = _answer(supported=False, answer="", citations=[], reason_if_unsupported="no")
    generate_grounded_answer("q", EVIDENCE, client=_FlakyClient(failures=3, parsed=parsed))

    assert no_sleep == sorted(no_sleep), f"delays should not shrink: {no_sleep}"
    assert len(no_sleep) == 3


def test_a_request_error_is_still_not_retried(no_sleep):
    class _Bad:
        def generate_content(self, **kwargs):
            raise RuntimeError("400 INVALID_ARGUMENT: bad schema")

    class _Client:
        models = _Bad()

    with pytest.raises(RuntimeError, match="400"):
        generate_grounded_answer("q", EVIDENCE, client=_Client())
    assert no_sleep == []


def test_the_client_bounds_a_single_request_and_owns_retry_policy(monkeypatch):
    """One retry layer, and no request that can hang.

    The SDK retries 408/429/5xx five times by itself with up to 60s delays.
    Under this module's own retry that is 25 requests per question, and with no
    timeout a stuck connection blocks forever - which turned one 30-case
    evaluation into a 7-hour run producing nothing.
    """
    import google.genai

    import src.generate_gemini as gg

    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(google.genai, "Client", fake_client)
    monkeypatch.setattr(gg, "_DEFAULT_CLIENT", None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    gg._client()

    options = captured["http_options"]
    assert options.timeout == gg.REQUEST_TIMEOUT_MS
    assert options.retry_options.attempts == 1, "the SDK must not retry underneath us"


@pytest.mark.parametrize("message", [
    "503 UNAVAILABLE. This model is currently experiencing high demand.",
    "504 DEADLINE_EXCEEDED. The request timed out.",
    "429 RESOURCE_EXHAUSTED. Quota exceeded",
])
def test_transient_server_conditions_are_retryable(message):
    """A slow or busy model is not a bad request."""
    from src.generate_gemini import _is_retryable

    assert _is_retryable(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "400 INVALID_ARGUMENT: bad schema",
    "401 Unauthorized",
    "404 NOT_FOUND: no such model",
])
def test_request_errors_are_not_retryable(message):
    """Retrying these just wastes quota - they will fail identically."""
    from src.generate_gemini import _is_retryable

    assert not _is_retryable(RuntimeError(message))


def test_a_daily_quota_error_is_not_retried(no_sleep):
    """A per-day cap cannot recover mid-run; retrying only burns minutes."""
    from src.generate_gemini import _is_retryable

    daily = RuntimeError(
        "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    )
    assert not _is_retryable(daily)

    class _Capped:
        def generate_content(self, **kwargs):
            raise daily

    class _Client:
        models = _Capped()

    with pytest.raises(RuntimeError, match="429"):
        generate_grounded_answer("q", EVIDENCE, client=_Client())
    assert no_sleep == [], "a daily cap should fail fast, not back off"


def test_a_per_minute_rate_limit_is_still_retried(no_sleep):
    """Only the per-day cap is unrecoverable; per-minute limits pass."""
    from src.generate_gemini import _is_retryable

    assert _is_retryable(RuntimeError(
        "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
        "GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 32s"
    ))


def test_retrying_stops_at_the_budget_rather_than_the_attempt_count(monkeypatch):
    """A degraded service cost ~7 minutes per question before this."""
    import src.generate_gemini as gg

    clock = {"now": 0.0}
    monkeypatch.setattr(gg.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(gg.time, "sleep", lambda s: clock.__setitem__("now", clock["now"] + s))
    monkeypatch.setattr(gg, "_wait_turn", lambda: None)

    parsed = _answer(supported=False, answer="", citations=[], reason_if_unsupported="no")
    # More failures than the budget allows time for.
    client = _FlakyClient(failures=99, parsed=parsed)

    with pytest.raises(Exception):
        generate_grounded_answer("q", EVIDENCE, client=client)

    assert clock["now"] <= gg.RETRY_BUDGET_SECONDS, (
        f"spent {clock['now']}s, budget is {gg.RETRY_BUDGET_SECONDS}s"
    )
