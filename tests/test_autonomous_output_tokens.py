"""
Autonomer Testlauf fuer Ausgabeformatierung und Token-Budgets.
Fuehrt gestufte Prompts aus, prueft Tool-Use (write_file) und speichert Report.
"""
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

import requests


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")
BACKEND_CONVERSATIONS_DIR = os.path.join(PROJECT_ROOT, "backend", "data", "conversations")
TEST_OUTPUT_DIR = os.path.join(DATA_DIR, "tool_tests")
REPORT_DIR = os.path.join(DATA_DIR, "test_runs")
REQUEST_TIMEOUT_SECONDS = 60
MAX_RUN_SECONDS = 300


def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_base_url() -> str:
    config = load_config()
    server = config.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = server.get("port", 8000)
    return f"http://{host}:{port}"


def health_check(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def list_conversations(paths: List[str], limit: int = 3) -> Tuple[List[Dict[str, Any]], str]:
    all_items = []
    source_path = ""
    for path in paths:
        if os.path.isdir(path):
            source_path = path
            for name in os.listdir(path):
                if not name.endswith(".json"):
                    continue
                full_path = os.path.join(path, name)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    all_items.append({
                        "id": data.get("id", name[:-5]),
                        "title": data.get("title", ""),
                        "updated_at": data.get("updated_at"),
                        "path": full_path,
                    })
                except Exception:
                    # Ignore unreadable files
                    continue
            if all_items:
                break
    if not all_items:
        return [], source_path
    # Sort by updated_at or mtime fallback
    def sort_key(item: Dict[str, Any]) -> float:
        if item.get("updated_at"):
            try:
                return datetime.fromisoformat(item["updated_at"]).timestamp()
            except Exception:
                pass
        try:
            return os.path.getmtime(item["path"])
        except Exception:
            return 0.0
    all_items.sort(key=sort_key, reverse=True)
    return all_items[:limit], source_path


def create_conversation(base_url: str) -> str:
    resp = requests.post(f"{base_url}/conversations", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["conversation_id"]


def send_chat(base_url: str, conversation_id: str, message: str, max_length: int, temperature: float, timeout: int) -> str:
    payload = {
        "message": message,
        "conversation_id": conversation_id,
        "max_length": max_length,
        "temperature": temperature
    }
    resp = requests.post(f"{base_url}/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def analyze_response(text: str, requested_max_length: int) -> Dict[str, Any]:
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    ends_incomplete = False
    if text:
        last = text.strip()[-1:]
        if last in [",", ";", ":"]:
            ends_incomplete = True
        if words:
            last_word = words[-1]
            if len(last_word) < 2 and last not in [".", "!", "?"]:
                ends_incomplete = True
    warnings = []
    if ends_incomplete:
        warnings.append("possible_truncation")
    if requested_max_length <= 64 and word_count > 160:
        warnings.append("too_long_for_budget")
    if requested_max_length >= 256 and word_count < 30:
        warnings.append("too_short_for_budget")
    return {
        "char_count": char_count,
        "word_count": word_count,
        "warnings": warnings
    }


def check_file(path: str, expected_content: str) -> Dict[str, Any]:
    abs_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(abs_path):
        return {"exists": False, "content_matches": False}
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"exists": True, "content_matches": content.strip() == expected_content.strip()}
    except Exception:
        return {"exists": True, "content_matches": False}


def find_conversation_file(conversation_id: str) -> str:
    root_path = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
    backend_path = os.path.join(BACKEND_CONVERSATIONS_DIR, f"{conversation_id}.json")
    if os.path.exists(root_path):
        return root_path
    if os.path.exists(backend_path):
        return backend_path
    return ""


def main() -> int:
    base_url = get_base_url()
    print(f"[INFO] Base URL: {base_url}")
    if not health_check(base_url):
        print("[ERROR] Server nicht erreichbar. Bitte Local AI Server starten.")
        return 1

    # Log last conversations (before new test run)
    last_convs, source = list_conversations([CONVERSATIONS_DIR, BACKEND_CONVERSATIONS_DIR], limit=3)
    if last_convs:
        print(f"[INFO] Letzte Conversations ({source}):")
        for item in last_convs:
            print(f"  - {item.get('id')} | {item.get('title')} | {item.get('updated_at')}")
    else:
        print("[INFO] Keine bestehenden Conversations gefunden.")

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    conversation_id = create_conversation(base_url)
    print(f"[INFO] Neue Conversation: {conversation_id}")

    tests = [
        {
            "id": "short_ok",
            "message": "Antworte nur mit dem Wort OK.",
            "max_length": 32
        },
        {
            "id": "file_write_short",
            "message": "Schreibe in die Datei data/tool_tests/test_short.txt den Inhalt Hallo",
            "max_length": 64,
            "timeout": 120,
            "expect_file": {"path": "data/tool_tests/test_short.txt", "content": "Hallo"}
        },
        {
            "id": "simple_points",
            "message": "Gib drei kurze Stichpunkte, warum klare Ausgabeformatierung wichtig ist.",
            "max_length": 128
        },
        {
            "id": "medium_explain",
            "message": "Erklaere in 5-7 Saetzen, wie Token-Budgets und Kontextlaenge zusammenhaengen.",
            "max_length": 192
        },
        {
            "id": "long_structured",
            "message": "Schreibe eine strukturierte Antwort mit Ueberschriften und einer kurzen Beispiel-Liste (100-150 Woerter) ueber Strategien zur Vermeidung von Antwort-Kuerzung in lokalen LLM-Setups.",
            "max_length": 256
        }
    ]

    results = {
        "timestamp": datetime.now().isoformat(),
        "base_url": base_url,
        "conversation_id": conversation_id,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "max_run_seconds": MAX_RUN_SECONDS,
        "tests": []
    }

    start_time = time.time()
    for item in tests:
        elapsed = time.time() - start_time
        if elapsed > MAX_RUN_SECONDS:
            results["tests"].append({
                "id": "run_timeout",
                "error": f"max_run_seconds_exceeded_{MAX_RUN_SECONDS}s"
            })
            print(f"[ERROR] Gesamtlaufzeit ueberschritten ({MAX_RUN_SECONDS}s)")
            break
        test_id = item["id"]
        message = item["message"]
        max_length = item["max_length"]
        timeout = item.get("timeout", REQUEST_TIMEOUT_SECONDS)
        print(f"[RUN] {test_id} (max_length={max_length})")
        try:
            response_text = send_chat(base_url, conversation_id, message, max_length, temperature=0.3, timeout=timeout)
            analysis = analyze_response(response_text, max_length)
            test_result = {
                "id": test_id,
                "max_length": max_length,
                "timeout_seconds": timeout,
                "response_preview": response_text[:200],
                "analysis": analysis
            }
            if "expect_file" in item:
                file_check = check_file(item["expect_file"]["path"], item["expect_file"]["content"])
                test_result["file_check"] = file_check
                if not file_check.get("exists"):
                    test_result.setdefault("analysis", {}).setdefault("warnings", []).append("file_not_written")
                elif not file_check.get("content_matches"):
                    test_result.setdefault("analysis", {}).setdefault("warnings", []).append("file_content_mismatch")
            results["tests"].append(test_result)
        except requests.exceptions.Timeout:
            timeout_result = {
                "id": test_id,
                "error": f"timeout_after_{timeout}s",
                "timeout": True
            }
            if "expect_file" in item:
                file_check = check_file(item["expect_file"]["path"], item["expect_file"]["content"])
                timeout_result["file_check"] = file_check
            results["tests"].append(timeout_result)
            print(f"[ERROR] Test {test_id} Timeout nach {timeout}s")
        except Exception as e:
            results["tests"].append({
                "id": test_id,
                "error": str(e)
            })
            print(f"[ERROR] Test {test_id} fehlgeschlagen: {e}")
        time.sleep(0.2)

    conv_file = find_conversation_file(conversation_id)
    results["conversation_file"] = conv_file
    if conv_file:
        print(f"[INFO] Conversation gespeichert: {conv_file}")
    else:
        print("[WARN] Conversation-Datei nicht gefunden (weder data/conversations noch backend/data/conversations)")

    report_name = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(REPORT_DIR, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Report gespeichert: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
