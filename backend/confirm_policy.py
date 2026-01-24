"""
Central confirmation policy helpers.
"""

from __future__ import annotations

from typing import Optional
import re


_CONFIRM_DELETE_RE = re.compile(r"^\s*CONFIRM\s+DELETE\s+(.+?)\s*$", re.IGNORECASE)


def parse_confirm_delete(message: str) -> Optional[str]:
    """
    Parses `CONFIRM DELETE <path>` and returns <path> if present.
    """
    if not message:
        return None
    m = _CONFIRM_DELETE_RE.match(message)
    if not m:
        return None
    path = (m.group(1) or "").strip().strip('"\''"`")
    return path or None

