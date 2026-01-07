"""
Test Script für MCP Server
Testet die grundlegende Funktionalität des MCP Servers
"""
import json
import sys
import os

# Füge backend zum Path hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import MCPServer


def test_initialize():
    """Test initialize Request"""
    print("Testing initialize...")
    server = MCPServer()
    
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "workspaceRoot": os.getcwd()
        }
    }
    
    # Simuliere Response (normalerweise würde das nach stdout gehen)
    # Hier prüfen wir nur ob keine Exception geworfen wird
    try:
        result = server.handle_initialize(request["params"])
        print(f"✓ Initialize erfolgreich: {result.get('serverInfo', {}).get('name')}")
        return True
    except Exception as e:
        print(f"✗ Initialize fehlgeschlagen: {e}")
        return False


def test_tools_list():
    """Test tools/list Request"""
    print("\nTesting tools/list...")
    server = MCPServer()
    server.initialized = True  # Simuliere initialisierten Server
    
    try:
        result = server.handle_tools_list()
        tools = result.get("tools", [])
        print(f"✓ Tools-Liste erfolgreich: {len(tools)} Tools gefunden")
        for tool in tools:
            print(f"  - {tool.get('name')}: {tool.get('description', '')[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Tools-Liste fehlgeschlagen: {e}")
        return False


def test_model_service_connection():
    """Test Model Service Verbindung"""
    print("\nTesting Model Service connection...")
    server = MCPServer()
    
    try:
        available = server.model_service.is_available()
        if available:
            print("✓ Model Service ist erreichbar")
            status = server.model_service.get_status()
            if status:
                print(f"  Text Model: {status.get('text_model', {}).get('loaded', False)}")
                print(f"  Audio Model: {status.get('audio_model', {}).get('loaded', False)}")
                print(f"  Image Model: {status.get('image_model', {}).get('loaded', False)}")
            return True
        else:
            print("⚠ Model Service ist nicht erreichbar (muss auf Port 8001 laufen)")
            return False
    except Exception as e:
        print(f"✗ Model Service Test fehlgeschlagen: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("MCP Server Test")
    print("=" * 50)
    
    results = []
    results.append(test_initialize())
    results.append(test_tools_list())
    results.append(test_model_service_connection())
    
    print("\n" + "=" * 50)
    print(f"Tests: {sum(results)}/{len(results)} erfolgreich")
    print("=" * 50)
    
    if all(results):
        print("\n✓ Alle Tests erfolgreich!")
        sys.exit(0)
    else:
        print("\n⚠ Einige Tests fehlgeschlagen")
        sys.exit(1)







