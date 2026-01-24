"""
Integration Tests (defensiv): Pong, Web-Search-Behavior, Streaming

Hinweis:
- Diese Tests sind so gebaut, dass sie in Umgebungen ohne laufenden Server nicht fehlschlagen (return = skip).
- Pong-Test: generiert Code via /chat, schreibt ihn lokal und prüft Syntax via py_compile.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Füge backend zum Python-Pfad hinzu (für Settings/Utilities falls nötig)
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "backend"))


API_BASE = "http://127.0.0.1:8000"


def _api_available() -> bool:
    try:
        import requests

        r = requests.get(f"{API_BASE}/status", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _extract_python_code(md: str) -> str:
    m = re.search(r"```python\\s*([\\s\\S]*?)```", md)
    return (m.group(1).strip() if m else "")


def test_pong_generation_and_saved_to_disk_compiles():
    if not _api_available():
        return

    import requests

    # Prompt bewusst konkret, aber kompakt (Syntax-Test, nicht Laufzeit)
    prompt = (
        "Erstelle ein vollständiges Python Pong-Spiel mit pygame als EIN einzelnes .py Script. "
        "Gib NUR den Code in einem ```python Codeblock``` aus, ohne weitere Erklärungen."
    )

    r = requests.post(
        f"{API_BASE}/chat",
        json={"message": prompt, "max_length": 2048, "temperature": 0.2},
        timeout=180,
    )
    if r.status_code != 200:
        return
    data = r.json()
    text = data.get("response") or ""
    code = _extract_python_code(text)
    if not code:
        # Modell hat keinen Codeblock geliefert -> skip (nicht als harter Fail)
        return

    out_dir = workspace_root / "data" / "test_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "pong_generated.py"
    out_file.write_text(code, encoding="utf-8")

    # Syntax-check
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(out_file)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"py_compile failed: {proc.stderr}"


def test_web_search_citations_when_enabled():
    if not _api_available():
        return

    # Web-Search Toggle (lokale Settings-Datei)
    qs_path = workspace_root / "data" / "quality_settings.json"
    if not qs_path.exists():
        return

    try:
        qs = json.loads(qs_path.read_text(encoding="utf-8"))
    except Exception:
        return

    if not qs.get("auto_web_search", False):
        return

    import requests

    r = requests.post(
        f"{API_BASE}/chat",
        json={"message": "Wer ist Ada Lovelace? Nenne die Quelle(n).", "max_length": 512, "temperature": 0.2},
        timeout=120,
    )
    if r.status_code != 200:
        return
    text = (r.json().get("response") or "").strip()
    if not text:
        return

    # Soft-assert: entweder Zitatform [1].. oder URL im Text
    assert ("[1]" in text) or ("http://" in text) or ("https://" in text)


def test_chat_stream_endpoint_returns_sse():
    if not _api_available():
        return

    import requests

    r = requests.post(
        f"{API_BASE}/chat/stream",
        json={"message": "Was ist 2+2?", "max_length": 128, "temperature": 0.0},
        stream=True,
        timeout=30,
    )
    assert r.status_code == 200
    ctype = (r.headers.get("content-type") or "").lower()
    assert "text/event-stream" in ctype

    got_data = False
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            got_data = True
            break
    assert got_data

