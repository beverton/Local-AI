"""
Test-Script für MCP-Server
Simuliert einen Request von Cursor
"""
import json
import subprocess
import sys
import os

# Pfad zum MCP-Server
mcp_server_path = os.path.join("backend", "mcp_server.py")
if not os.path.exists(mcp_server_path):
    print(f"Fehler: {mcp_server_path} nicht gefunden")
    sys.exit(1)

# Test-Request (simuliert Cursor's Tool-Call)
test_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "chat",
        "arguments": {
            "message": "Test: Was ist 2+2?"
        }
    }
}

print("Sende Test-Request an MCP-Server...")
print(f"Request: {json.dumps(test_request, indent=2)}")
print("\n" + "="*50 + "\n")

# Starte MCP-Server und sende Request
try:
    # Zuerst initialize Request
    initialize_request = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "workspaceRoot": os.getcwd()
        }
    }
    
    # Bestimme korrekten cwd
    project_root = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(project_root, "backend")
    
    process = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=backend_dir
    )
    
    # Sende initialize
    print("Sende initialize Request...")
    process.stdin.write(json.dumps(initialize_request) + "\n")
    process.stdin.flush()
    
    # Warte auf Response
    response_line = process.stdout.readline()
    if response_line:
        response = json.loads(response_line.strip())
        print(f"Initialize Response: {json.dumps(response, indent=2)}")
    
    # Sende tools/call Request
    print("\nSende tools/call Request...")
    process.stdin.write(json.dumps(test_request) + "\n")
    process.stdin.flush()
    
    # Warte auf Response
    response_line = process.stdout.readline()
    if response_line:
        response = json.loads(response_line.strip())
        print(f"Tools/Call Response: {json.dumps(response, indent=2)}")
    else:
        print("Keine Response erhalten!")
    
    process.terminate()
    process.wait(timeout=5)
    
except Exception as e:
    print(f"Fehler: {e}")
    import traceback
    traceback.print_exc()
