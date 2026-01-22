"""
Test-Script für die Optimierungen: Sonderzeichen-Fix und Tool-Aufrufe
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from model_manager import ModelManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_special_char_validation():
    """Testet die Sonderzeichen-Validierung"""
    print("\n=== Test 1: Sonderzeichen-Validierung ===")
    
    manager = ModelManager()
    
    # Test mit hauptsächlich Sonderzeichen
    bad_response = ".      .   ." * 20  # 155 Zeichen, hauptsächlich Sonderzeichen
    result = manager._validate_response_quality(bad_response)
    print(f"Response mit Sonderzeichen: {result} (sollte False sein)")
    assert result == False, "Sonderzeichen-Response sollte als ungültig erkannt werden"
    
    # Test mit gültiger Response
    good_response = "Das ist eine gültige Antwort mit vielen Wörtern."
    result = manager._validate_response_quality(good_response)
    print(f"Gültige Response: {result} (sollte True sein)")
    assert result == True, "Gültige Response sollte akzeptiert werden"
    
    print("✓ Sonderzeichen-Validierung funktioniert!")

def test_tool_descriptions():
    """Testet ob Tool-Beschreibungen vorhanden sind"""
    print("\n=== Test 2: Tool-Beschreibungen ===")
    
    from mcp_server import MCPServer
    server = MCPServer()
    
    if "write_file" in server.tools:
        tool = server.tools["write_file"]
        description = tool.get("description", "")
        print(f"write_file Beschreibung: {description[:50]}...")
        assert "erstellt" in description.lower() or "erstellt" in description.lower(), "Beschreibung sollte 'erstellt' enthalten"
        print("✓ Tool-Beschreibungen sind vorhanden!")
    else:
        print("⚠ write_file Tool nicht gefunden")

if __name__ == "__main__":
    print("Teste Optimierungen: Sonderzeichen-Fix und Tool-Aufrufe")
    
    try:
        test_special_char_validation()
        test_tool_descriptions()
        print("\n✅ Alle Tests bestanden!")
    except Exception as e:
        print(f"\n❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
