"""Pick whichever generation backend is configured.

Two are supported and they are interchangeable: both return the same answer
shape, share the same prompt and answer schema, and have their quotes verified
against the filing by the same code. Only the API call differs.

    SEC_SIMPLIFIER_BACKEND=claude|gemini|none   force one (optional)
    ANTHROPIC_API_KEY                           enables Claude
    GEMINI_API_KEY                              enables Gemini

With neither key set there is no generator, and the app falls back to
extractive answers - which is the mode every measurement in PROGRESS.md was
taken in.
"""

from __future__ import annotations

import os
from typing import Any, Callable

# (name, module path) in preference order when nothing is forced. Claude first
# only because it is what the code was written against; Gemini's free tier is
# the cheaper way to run the same evaluation.
BACKENDS = (("claude", "src.generate"), ("gemini", "src.generate_gemini"))


def _load(module_path: str):
    import importlib

    return importlib.import_module(module_path)


def available_backends() -> list[str]:
    """Backends whose SDK is installed and whose key is set."""
    ready = []
    for name, module_path in BACKENDS:
        try:
            if _load(module_path).is_available():
                ready.append(name)
        except ImportError:
            continue
    return ready


def active_backend() -> tuple[str, Any] | tuple[None, None]:
    """The chosen backend's name and module, or (None, None) for extractive mode."""
    forced = os.environ.get("SEC_SIMPLIFIER_BACKEND", "").strip().lower()
    if forced in {"none", "off", "extractive"}:
        return None, None

    for name, module_path in BACKENDS:
        if forced and forced != name:
            continue
        try:
            module = _load(module_path)
        except ImportError:
            continue
        if module.is_available():
            return name, module

    return None, None


def get_generator() -> Callable[[str, list[dict]], dict] | None:
    """A generator callable for `answer_question`, or None if none is configured."""
    _, module = active_backend()
    if module is None:
        return None
    return lambda question, evidence: module.generate_grounded_answer(question, evidence)


def format_cost(usage: dict) -> str:
    """A cost line for one call, priced by whichever backend actually ran."""
    tokens = f"{usage.get('input_tokens', 0):,} in / {usage.get('output_tokens', 0):,} out"
    model = usage.get("model", "?")

    module = None
    for name, module_path in BACKENDS:
        try:
            candidate = _load(module_path)
        except ImportError:
            continue
        if model.startswith(name) or model.startswith(candidate.DEFAULT_MODEL.split("-")[0]):
            module = candidate
            break

    if module is None:
        return f"{tokens} tokens on {model}"

    cost = (
        usage.get("input_tokens", 0) * module.COST_PER_INPUT_TOKEN
        + usage.get("output_tokens", 0) * module.COST_PER_OUTPUT_TOKEN
    )
    priced = "free tier" if cost == 0 else f"about ${cost:.4f}"
    return f"{tokens} tokens on {model} - {priced}"


def describe() -> str:
    """What is *configured*, for a status line before any question is asked.

    Deliberately phrased as intent rather than fact. A configured backend can
    still be out of quota or unreachable, so claiming answers "are written by"
    it would overstate - which is precisely the failure mode this project
    exists to avoid. What actually happened is reported per answer, from the
    `generated` and `generation_error` fields on the response.
    """
    name, module = active_backend()
    if module is None:
        return "generation off - answers excerpted from filings"
    return f"generation enabled ({name} {module.DEFAULT_MODEL})"
