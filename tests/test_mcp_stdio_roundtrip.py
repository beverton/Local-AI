import json
import os
import subprocess
import sys
import threading
import time
from queue import Queue, Empty


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MCP_SERVER = os.path.join(REPO_ROOT, "backend", "mcp_server.py")
DEBUG_LOG = os.path.join(REPO_ROOT, ".cursor", "debug.log")


def _log_debug(message, data):
    payload = {
        "sessionId": "debug-session",
        "runId": "pre-fix",
        "hypothesisId": "TEST",
        "location": "tests/test_mcp_stdio_roundtrip.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _read_stream_lines(stream, queue, label):
    for line in iter(stream.readline, ""):
        queue.put((label, line.rstrip()))


def _send(proc, payload):
    line = json.dumps(payload, separators=(",", ":"))
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def _assert_json_rpc(response, expected_id=None):
    data = json.loads(response)
    assert data.get("jsonrpc") == "2.0", f"Invalid jsonrpc: {data}"
    if expected_id is not None:
        assert data.get("id") == expected_id, f"Unexpected id: {data}"
    return data


def test_stdio_roundtrip():
    """Basic JSON-RPC roundtrip test for MCP stdio server."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "backend")

    proc = subprocess.Popen(
        [sys.executable, MCP_SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        bufsize=1,
    )
    _log_debug("process_started", {"pid": proc.pid, "server": MCP_SERVER})

    stdout_queue = Queue()
    stderr_queue = Queue()
    stdout_thread = threading.Thread(
        target=_read_stream_lines, args=(proc.stdout, stdout_queue, "stdout"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_read_stream_lines, args=(proc.stderr, stderr_queue, "stderr"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        # initialize
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"roots": {"listChanged": False}},
                    "clientInfo": {"name": "test", "version": "0.0.1"},
                },
            },
        )
        init_line = None
        end_time = time.time() + 5
        while time.time() < end_time and init_line is None:
            try:
                label, line = stdout_queue.get_nowait()
                if label == "stdout":
                    init_line = line
            except Empty:
                time.sleep(0.05)
        if not init_line:
            stderr_dump = []
            while not stderr_queue.empty():
                stderr_dump.append(stderr_queue.get_nowait()[1])
            _log_debug("initialize_timeout", {"stderr": stderr_dump})
            raise AssertionError("No initialize response (stdout empty)")
        init_data = _assert_json_rpc(init_line, expected_id=1)
        assert "result" in init_data, f"Initialize missing result: {init_data}"

        # tools/list
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools_line = None
        end_time = time.time() + 5
        while time.time() < end_time and tools_line is None:
            try:
                label, line = stdout_queue.get_nowait()
                if label == "stdout":
                    tools_line = line
            except Empty:
                time.sleep(0.05)
        if not tools_line:
            stderr_dump = []
            while not stderr_queue.empty():
                stderr_dump.append(stderr_queue.get_nowait()[1])
            _log_debug("tools_list_timeout", {"stderr": stderr_dump})
            raise AssertionError("No tools/list response (stdout empty)")
        tools_data = _assert_json_rpc(tools_line, expected_id=2)
        tools = tools_data.get("result", {}).get("tools", [])
        assert any(t.get("name") == "chat" for t in tools), "chat tool missing"

        # tools/call chat
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "chat", "arguments": {"message": "test roundtrip"}},
            },
        )
        chat_line = None
        end_time = time.time() + 30
        while time.time() < end_time and chat_line is None:
            try:
                label, line = stdout_queue.get_nowait()
                if label == "stdout":
                    chat_line = line
            except Empty:
                time.sleep(0.05)
        if not chat_line:
            stderr_dump = []
            while not stderr_queue.empty():
                stderr_dump.append(stderr_queue.get_nowait()[1])
            _log_debug("tools_call_timeout", {"stderr": stderr_dump})
            raise AssertionError("No tools/call response (stdout empty)")
        chat_data = _assert_json_rpc(chat_line, expected_id=3)
        content = chat_data.get("result", {}).get("content", [])
        assert content, f"Missing content in chat response: {chat_data}"
        assert content[0].get("type") == "text", f"Unexpected content: {content}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
