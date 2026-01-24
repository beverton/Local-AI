import json
import os
import time
from typing import Any, Dict, Optional

import requests


BASE = "http://127.0.0.1:8000"
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_quality_settings() -> Dict[str, Any]:
    return requests.get(f"{BASE}/quality/settings", timeout=10).json()


def set_quality_settings(settings: Dict[str, Any]) -> None:
    requests.post(f"{BASE}/quality/settings", json=settings, timeout=10).raise_for_status()


def chat(message: str, conversation_id: Optional[str] = None, max_length: int = 512, temperature: float = 0.2) -> Dict[str, Any]:
    r = requests.post(
        f"{BASE}/chat",
        json={
            "message": message,
            "conversation_id": conversation_id,
            "max_length": max_length,
            "temperature": temperature,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    # Use a dedicated temp path inside workspace
    test_dir = "data/tool_tests"
    test_file = f"{test_dir}/e2e_tool_test.txt"
    abs_test_file = os.path.join(WORKSPACE, test_file)

    cid = None

    tests = [
        ("write_file", f"Schreibe in die Datei {test_file} den Inhalt Hallo Welt"),
        ("file_exists", f"Existiert die Datei {test_file}?"),
        ("read_file", f"Lies die Datei {test_file}"),
        ("list_directory", f"Liste das Verzeichnis {test_dir}"),
        ("delete_file_needs_confirm", f"Lösche die Datei {test_file}"),
        ("file_exists_after_delete", f"Existiert die Datei {test_file}?"),
        ("web_search_disabled", "Suche nach aktuellen Infos zu Qwen 2.5 7B Instruct und gib mir 2 Quellen mit URLs."),
    ]

    print("running", len(tests), "tests")
    for name, prompt in tests:
        t0 = time.time()
        res = chat(prompt, conversation_id=cid, max_length=512, temperature=0.2)
        cid = res.get("conversation_id") or cid
        dt = time.time() - t0
        text = (res.get("response") or "").strip()
        print("\n---", name, "sec", round(dt, 2), "conversation_id", cid)
        print(text[:700].encode("unicode_escape", "backslashreplace").decode("ascii"))

        # Ground-truth checks for file tools (local FS)
        if name == "write_file":
            print("fs_exists_after_write", os.path.exists(abs_test_file))
            if os.path.exists(abs_test_file):
                with open(abs_test_file, "r", encoding="utf-8") as f:
                    print("fs_content", f.read().encode("unicode_escape", "backslashreplace").decode("ascii"))
        if name == "delete_file":
            print("fs_exists_after_delete", os.path.exists(abs_test_file))

        if name == "delete_file_needs_confirm":
            # should not have deleted yet
            exists_before = os.path.exists(abs_test_file)
            print("fs_exists_before_confirm", exists_before)
            # confirm
            res2 = chat(f"CONFIRM DELETE {test_file}", conversation_id=cid, max_length=256, temperature=0.2)
            cid = res2.get("conversation_id") or cid
            text2 = (res2.get("response") or "").strip()
            print("--- delete_file_confirm_response")
            print(text2[:400].encode("unicode_escape", "backslashreplace").decode("ascii"))
            print("fs_exists_after_confirm", os.path.exists(abs_test_file))

    # Web search tool test (enabled) – toggle on temporarily
    s = get_quality_settings()
    orig = bool(s.get("auto_web_search", False))
    try:
        s["auto_web_search"] = True
        set_quality_settings(s)
        t0 = time.time()
        res = chat("Suche nach Qwen 2.5 7B Instruct Wikipedia und gib mir die Top 2 Ergebnisse.", conversation_id=cid, max_length=512, temperature=0.2)
        dt = time.time() - t0
        cid = res.get("conversation_id") or cid
        text = (res.get("response") or "").strip()
        print("\n--- web_search_enabled sec", round(dt, 2), "conversation_id", cid)
        print(text[:900].encode("unicode_escape", "backslashreplace").decode("ascii"))
    finally:
        s["auto_web_search"] = orig
        try:
            set_quality_settings(s)
        except Exception:
            pass


if __name__ == "__main__":
    main()

