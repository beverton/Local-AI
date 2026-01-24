"""
Generation Trace - lightweight diagnostics for streaming generation.

Goals:
- Observe server-side chunks during generation (timing + sizes)
- Preserve the final end result (and optional polished/replace result)
- Provide debug endpoints to fetch recent traces
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import os
import json
import time
import uuid
import threading


@dataclass
class GenerationTrace:
    trace_id: str
    route: str
    started_at: float

    # Request context
    conversation_id: Optional[str] = None
    model_id: Optional[str] = None
    use_model_service: Optional[bool] = None
    language: Optional[str] = None
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    message: Optional[str] = None

    # Streaming stats
    first_chunk_at: Optional[float] = None
    chunks_total: int = 0
    chars_total: int = 0
    chunks_stored: int = 0
    chunk_previews: List[str] = field(default_factory=list)

    # Tool loop observability
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # Final result
    full_response: Optional[str] = None
    replaced_response: Optional[str] = None

    # Errors
    error: Optional[str] = None
    ended_at: Optional[float] = None

    def add_chunk(self, chunk: str, store_limit: int = 120, max_chunks_store: int = 200):
        now = time.time()
        if self.first_chunk_at is None:
            self.first_chunk_at = now

        self.chunks_total += 1
        self.chars_total += len(chunk or "")

        # Store only a limited preview to avoid huge memory usage
        if self.chunks_stored < max_chunks_store and chunk:
            preview = chunk if len(chunk) <= store_limit else (chunk[:store_limit] + "…")
            self.chunk_previews.append(preview)
            self.chunks_stored += 1

    def add_tool_call(self, *, name: str, arguments: Any, call_id: Optional[str] = None):
        try:
            self.tool_calls.append(
                {
                    "t": time.time(),
                    "name": name,
                    "call_id": call_id,
                    "arguments": arguments,
                }
            )
        except Exception:
            return

    def add_tool_result(self, *, name: str, ok: bool, call_id: Optional[str] = None, preview: Optional[str] = None, error: Optional[str] = None):
        try:
            self.tool_results.append(
                {
                    "t": time.time(),
                    "name": name,
                    "call_id": call_id,
                    "ok": bool(ok),
                    "preview": preview,
                    "error": error,
                }
            )
        except Exception:
            return

    def finish(self, *, full_response: Optional[str] = None, replaced_response: Optional[str] = None, error: Optional[str] = None):
        self.ended_at = time.time()
        if full_response is not None:
            self.full_response = full_response
        if replaced_response is not None:
            self.replaced_response = replaced_response
        if error is not None:
            self.error = error

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Keep payloads reasonable in debug endpoint
        if d.get("full_response") and isinstance(d["full_response"], str) and len(d["full_response"]) > 8000:
            d["full_response"] = d["full_response"][:8000] + "\n…(truncated)"
        if d.get("replaced_response") and isinstance(d["replaced_response"], str) and len(d["replaced_response"]) > 8000:
            d["replaced_response"] = d["replaced_response"][:8000] + "\n…(truncated)"
        # tool arguments can get large; keep endpoint responsive
        if isinstance(d.get("tool_calls"), list) and len(d["tool_calls"]) > 50:
            d["tool_calls"] = d["tool_calls"][:50] + [{"truncated": True, "count": len(d["tool_calls"])}]
        if isinstance(d.get("tool_results"), list) and len(d["tool_results"]) > 50:
            d["tool_results"] = d["tool_results"][:50] + [{"truncated": True, "count": len(d["tool_results"])}]
        return d


class GenerationTraceBuffer:
    def __init__(self, max_items: int = 30, log_file: Optional[str] = None):
        self.max_items = max_items
        self._items: List[GenerationTrace] = []
        self._lock = threading.Lock()
        self.log_file = log_file

    def start(self, *, route: str, conversation_id: Optional[str], model_id: Optional[str], use_model_service: Optional[bool],
              language: Optional[str], max_length: Optional[int], temperature: Optional[float], message: Optional[str]) -> GenerationTrace:
        trace = GenerationTrace(
            trace_id=str(uuid.uuid4()),
            route=route,
            started_at=time.time(),
            conversation_id=conversation_id,
            model_id=model_id,
            use_model_service=use_model_service,
            language=language,
            max_length=max_length,
            temperature=temperature,
            message=message,
        )
        with self._lock:
            self._items.insert(0, trace)
            if len(self._items) > self.max_items:
                self._items = self._items[: self.max_items]
        return trace

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._items[: max(1, min(int(limit), self.max_items))])
        return [t.to_dict() for t in items]

    def get_by_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for t in self._items:
                if t.trace_id == trace_id:
                    return t.to_dict()
        return None

    def persist(self, trace: GenerationTrace):
        if not self.log_file:
            return
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            payload = trace.to_dict()
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Diagnostics must never break the server
            return


def default_trace_buffer(workspace_root: str) -> GenerationTraceBuffer:
    log_path = os.path.join(workspace_root, "logs", "generation_trace.jsonl")
    return GenerationTraceBuffer(max_items=30, log_file=log_path)

