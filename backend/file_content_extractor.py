"""
Centralized content extraction helpers (avoid duplicated heuristics).
"""

from __future__ import annotations

from typing import Optional
import re


_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\s*\n([\s\S]*?)\n```", re.MULTILINE)


def extract_code_fence_content(text: str) -> Optional[str]:
    """
    Extracts the first fenced code block content from text.
    Returns None if no code fence is found or content is empty.
    """
    if not text:
        return None
    m = _CODE_FENCE_RE.search(text)
    if not m:
        return None
    content = (m.group(1) or "").rstrip("\n")
    return content if content.strip() else None


def extract_after_marker(text: str) -> Optional[str]:
    """
    Extracts content after common "content markers" in prompts.
    """
    if not text:
        return None
    markers = [
        "mit folgendem inhalt:",
        "folgenden inhalt:",
        "inhalt:",
        "content:",
    ]
    low = text.lower()
    for mk in markers:
        idx = low.find(mk)
        if idx != -1:
            tail = text[idx + len(mk) :].lstrip()
            return tail if tail.strip() else None
    return None


def extract_write_file_content(prompt: str) -> Optional[str]:
    """
    Best-effort extraction for write_file content from user prompt.
    Prefers fenced code, then marker-based tail.
    """
    if not prompt:
        return None
    c = extract_code_fence_content(prompt)
    if c:
        return c
    return extract_after_marker(prompt)

