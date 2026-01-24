import json
import os
import time
from typing import Any, Dict

import requests


BASE = "http://127.0.0.1:8000"
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def wait_ready(timeout_s: int = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server not ready")


def stream(message: str, max_length: int = 512, temperature: float = 0.2) -> Dict[str, Any]:
    r = requests.post(
        f"{BASE}/chat/stream",
        json={"message": message, "max_length": max_length, "temperature": temperature},
        stream=True,
        timeout=300,
    )
    r.raise_for_status()
    seen = {"meta": 0, "tool_call": 0, "tool_result": 0, "chunk": 0, "done": 0, "error": 0, "replace": 0}
    full = ""
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        evt = json.loads(line[len("data:") :].strip())
        for k in list(seen.keys()):
            if k in evt:
                seen[k] += 1
        if isinstance(evt.get("chunk"), str):
            full += evt["chunk"]
        if evt.get("error"):
            return {"seen": seen, "full": full, "error": evt.get("error")}
        if evt.get("done"):
            break
    return {"seen": seen, "full": full}


def main() -> None:
    wait_ready()

    # tool read_file
    res = None
    last_err = None
    for _ in range(5):
        res = stream("Lies die Datei README.md", max_length=256, temperature=0.2)
        if res.get("error"):
            last_err = res.get("error")
            time.sleep(2)
            continue
        break
    assert res is not None
    if res.get("error"):
        raise AssertionError(f"stream error: {last_err}")
    print("read_file seen", res["seen"])
    assert res["seen"]["tool_call"] >= 1, "expected tool_call in stream"
    assert res["seen"]["tool_result"] >= 1, "expected tool_result in stream"
    assert res["seen"]["chunk"] >= 1, "expected chunks"

    # delete confirmation flow via /chat (stream should ask for confirm)
    test_dir = os.path.join(WORKSPACE, "data", "tool_tests")
    os.makedirs(test_dir, exist_ok=True)
    test_rel = "data/tool_tests/stream_delete_test.txt"
    test_abs = os.path.join(WORKSPACE, test_rel.replace("/", os.sep))
    with open(test_abs, "w", encoding="utf-8") as f:
        f.write("delete me")

    res2 = stream(f"Lösche die Datei {test_rel}", max_length=256, temperature=0.2)
    print("delete_first seen", res2["seen"])
    assert "CONFIRM DELETE" in res2["full"], "expected confirmation request"
    assert os.path.exists(test_abs), "file should still exist before confirm"

    # confirm deletion via /chat (non-stream) is fine here
    r = requests.post(f"{BASE}/chat", json={"message": f"CONFIRM DELETE {test_rel}", "max_length": 256, "temperature": 0.2}, timeout=120)
    r.raise_for_status()
    time.sleep(0.5)
    assert not os.path.exists(test_abs), "file should be deleted after confirm"

    print("ok")


if __name__ == "__main__":
    main()

