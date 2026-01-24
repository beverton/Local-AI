import json
import re
import time
from typing import Tuple

import requests


BASE = "http://127.0.0.1:8000"


def run_stream(message: str, max_length: int = 384, temperature: float = 0.2) -> Tuple[str | None, str, bool]:
    r = requests.post(
        f"{BASE}/chat/stream",
        json={"message": message, "max_length": max_length, "temperature": temperature},
        stream=True,
        timeout=300,
    )
    trace_id = None
    full = str()
    replaced = False

    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        evt = json.loads(line[5:].strip())
        meta = evt.get("meta")
        if isinstance(meta, dict) and meta.get("trace_id"):
            trace_id = meta["trace_id"]

        if evt.get("replace") and isinstance(evt.get("content"), str):
            replaced = True
            full = evt["content"]
            continue

        chunk = evt.get("chunk")
        if isinstance(chunk, str):
            full += chunk

        if evt.get("done"):
            break

    return trace_id, full, replaced


def weird_case_score(text: str) -> tuple[int, int]:
    words = re.findall(r"[A-Za-z]{4,}", text)
    bad = 0
    for w in words:
        inner = w[1:]
        if (not w.isupper()) and any(c.isupper() for c in inner):
            bad += 1
    return bad, len(words)


def main() -> None:
    prompts = [
        "Schreibe eine kurze Antwort (3 Saetze): Was ist der Unterschied zwischen TCP und UDP?",
        "Gib mir eine Schritt-fuer-Schritt Anleitung, wie ich einen PC sicher neu aufsetze.",
        "Wie macht man marmelade? gib mir ein rezept.",
        "Nenne 5 Gruende, warum regelmaessiger Schlaf wichtig ist (kurz).",
        "Erklaere mir in 6 Bulletpoints was ein Reverse Proxy ist.",
    ]

    results = []
    for p in prompts:
        t0 = time.time()
        tid, full, repl = run_stream(p)
        dt = time.time() - t0
        bad, total = weird_case_score(full)
        results.append((p, tid, dt, repl, bad, total, ("\ufffd" in full), full))

    print("count", len(results))
    for i, (p, tid, dt, repl, bad, total, has_ufffd, full) in enumerate(results, 1):
        print("\n---", i)
        print("trace_id", tid, "sec", round(dt, 2), "replace", repl, "weird_case", f"{bad}/{total}", "has_ufffd", has_ufffd)
        prev = full[:500].encode("unicode_escape", "backslashreplace").decode("ascii")
        print("preview", prev)

    trs = requests.get(f"{BASE}/debug/generations?limit=10", timeout=10).json().get("traces", [])
    print("\nrecent_traces", len(trs))
    if trs:
        print("top_trace", trs[0].get("trace_id"), "chunks", trs[0].get("chunks_total"), "chars", trs[0].get("chars_total"))


if __name__ == "__main__":
    main()

