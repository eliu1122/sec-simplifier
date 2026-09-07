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


def describe() -> str:
    """One line naming the backend and model, for status lines and reports."""
    name, module = active_backend()
    if module is None:
        return "answers excerpted from filings (no generation backend configured)"
    return f"answers written by {name} ({module.DEFAULT_MODEL}) from cited evidence"
