"""
Integration-ish Tests: Streaming /chat im Model Service (Port 8001)

Diese Tests sind defensiv:
- Wenn Model-Service nicht erreichbar oder kein Textmodell geladen ist -> skip (return).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Füge backend zum Python-Pfad hinzu
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "backend"))


def _model_service_available() -> bool:
    try:
        import requests

        r = requests.get("http://127.0.0.1:8001/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _text_model_loaded() -> bool:
    try:
        import requests

        r = requests.get("http://127.0.0.1:8001/models/text/status", timeout=3)
        if r.status_code != 200:
            return False
        data = r.json()
        return bool(data.get("loaded"))
    except Exception:
        return False


def test_model_service_chat_stream_false_returns_json():
    # Skip wenn Service/Model nicht da
    if not _model_service_available() or not _text_model_loaded():
        return

    from model_service_client import ModelServiceClient

    client = ModelServiceClient(host="127.0.0.1", port=8001)
    res = client.chat(
        message="Was ist 2+2?",
        max_length=64,
        temperature=0.0,
        stream=False,
    )
    assert isinstance(res, dict)
    assert "response" in res and isinstance(res["response"], str) and res["response"].strip()
    assert "model_id" in res


def test_model_service_chat_stream_true_is_event_stream():
    # Skip wenn Service/Model nicht da
    if not _model_service_available() or not _text_model_loaded():
        return

    import requests

    r = requests.post(
        "http://127.0.0.1:8001/chat",
        json={
            "message": "Was ist 2+2?",
            "max_length": 64,
            "temperature": 0.0,
            "stream": True,
        },
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=20,
    )
    assert r.status_code == 200
    ctype = (r.headers.get("content-type") or "").lower()
    assert "text/event-stream" in ctype

    # Lies ein paar Events an, um sicherzustellen, dass Daten kommen
    got_data = False
    for _line in r.iter_lines(decode_unicode=True):
        if not _line:
            continue
        if _line.startswith("data:"):
            got_data = True
            break
    assert got_data, "Kein SSE 'data:' Event empfangen"

