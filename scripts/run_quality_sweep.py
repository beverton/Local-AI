import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


BASE = "http://127.0.0.1:8000"
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def wait_for_text_model_loaded(timeout_sec: int = 90) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            r = requests.get("http://127.0.0.1:8001/models/text/status", timeout=3)
            if r.status_code == 200 and r.json().get("loaded"):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def chat_stream(message: str, max_length: int = 384, temperature: float = 0.2) -> Tuple[Optional[str], str, bool, float, Optional[str]]:
    t0 = time.time()
    r = requests.post(
        f"{BASE}/chat/stream",
        json={"message": message, "max_length": max_length, "temperature": temperature},
        stream=True,
        timeout=300,
    )
    trace_id = None
    full = ""
    replaced = False
    error = None

    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        evt = json.loads(line[5:].strip())
        meta = evt.get("meta")
        if isinstance(meta, dict) and meta.get("trace_id"):
            trace_id = meta["trace_id"]
        if evt.get("error"):
            error = str(evt.get("error"))
            break
        if evt.get("replace") and isinstance(evt.get("content"), str):
            replaced = True
            full = evt["content"]
            continue
        if isinstance(evt.get("chunk"), str):
            full += evt["chunk"]
        if evt.get("done"):
            break

    return trace_id, full, replaced, (time.time() - t0), error


def chat(message: str, conversation_id: Optional[str] = None, max_length: int = 512, temperature: float = 0.2) -> Dict[str, Any]:
    r = requests.post(
        f"{BASE}/chat",
        json={"message": message, "conversation_id": conversation_id, "max_length": max_length, "temperature": temperature},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def compile_python(path: str) -> str:
    import py_compile

    py_compile.compile(path, doraise=True)
    return "ok"


def main() -> None:
    print("waiting for text model ...", wait_for_text_model_loaded())
    stream_prompts: List[Tuple[str, int, float]] = [
        ("Schreibe eine kurze Antwort (3 Saetze): Was ist der Unterschied zwischen TCP und UDP?", 256, 0.2),
        ("Erklaere mir in 6 Bulletpoints was ein Reverse Proxy ist.", 384, 0.2),
        ("Gib mir eine kurze Schritt-fuer-Schritt Anleitung (max 6 Schritte) um Windows neu aufzusetzen.", 320, 0.2),
        ("Schreibe eine kreative, aber kurze Mini-Geschichte (6 Saetze) ueber einen Roboter.", 384, 0.6),
    ]

    print("STREAM SWEEP", len(stream_prompts))
    for msg, ml, temp in stream_prompts:
        tid, text, repl, sec, err = chat_stream(msg, max_length=ml, temperature=temp)
        preview = text[:500].encode("unicode_escape", "backslashreplace").decode("ascii")
        print("\n--- stream", "sec", round(sec, 2), "trace_id", tid, "replace", repl, "temp", temp, "max_length", ml, "error", bool(err))
        if err:
            print(("ERROR: " + err)[:500].encode("unicode_escape", "backslashreplace").decode("ascii"))
        else:
            print(preview)

    # /chat tool + coding sweep (ChatAgent)
    print("\nCHAT/TOOLS SWEEP")
    cid = None
    test_file = "data/tool_tests/sweep_hello.py"
    abs_test_file = os.path.join(WORKSPACE, test_file)
    code = "print('hello from tool write')\n"
    # tool write (multiline pattern)
    res = chat(f"Erstelle eine Datei {test_file} mit folgendem Inhalt:\n{code}", conversation_id=cid, max_length=512, temperature=0.2)
    cid = res.get("conversation_id") or cid
    print("\n--- chat write_file", "conversation_id", cid)
    print((res.get("response") or "")[:300].encode("unicode_escape", "backslashreplace").decode("ascii"))
    print("fs_exists", os.path.exists(abs_test_file))
    if os.path.exists(abs_test_file):
        try:
            print("compile", compile_python(abs_test_file))
        except Exception as e:
            print("compile_error", str(e).encode("unicode_escape", "backslashreplace").decode("ascii")[:500])
            with open(abs_test_file, "r", encoding="utf-8") as f:
                head = f.read(120)
            print("file_head", head.encode("unicode_escape", "backslashreplace").decode("ascii"))

    # tool read
    res = chat(f"Lies die Datei {test_file}", conversation_id=cid, max_length=256, temperature=0.2)
    print("\n--- chat read_file")
    print((res.get("response") or "")[:300].encode("unicode_escape", "backslashreplace").decode("ascii"))

    # tool delete
    res = chat(f"Lösche die Datei {test_file}", conversation_id=cid, max_length=256, temperature=0.2)
    print("\n--- chat delete_file")
    print((res.get("response") or "")[:300].encode("unicode_escape", "backslashreplace").decode("ascii"))
    print("fs_exists_after_delete", os.path.exists(abs_test_file))


if __name__ == "__main__":
    main()

