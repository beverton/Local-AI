"""
MCP Server - Model Context Protocol Server für Cursor Integration
Implementiert MCP über stdio (Standard Input/Output) für Cursor
"""
import json
import sys
import logging
import os
from typing import Dict, Any, List, Optional
from io import TextIOWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Model Service Client
from model_service_client import ModelServiceClient
from agent_tools import (
    web_search, read_file, write_file, list_directory, 
    delete_file, file_exists, initialize_tools
)


class MCPServer:
    """MCP Server für Cursor Integration"""
    
    def __init__(self, model_service_host: str = "127.0.0.1", model_service_port: int = 8001):
        self.model_service = ModelServiceClient(model_service_host, model_service_port)
        self.request_id = 0
        self.initialized = False
        
        # Tool Registry
        self.tools = {
            "web_search": {
                "name": "web_search",
                "description": "Führt eine Websuche durch und gibt strukturierte Ergebnisse zurück",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Die Suchanfrage"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximale Anzahl der Ergebnisse (Standard: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            "read_file": {
                "name": "read_file",
                "description": "Liest eine Datei und gibt den Inhalt zurück",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Der Pfad zur Datei (relativ zum Workspace oder absolut)"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            "write_file": {
                "name": "write_file",
                "description": "Schreibt eine Datei",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Der Pfad zur Datei"
                        },
                        "content": {
                            "type": "string",
                            "description": "Der Inhalt der Datei"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            },
            "list_directory": {
                "name": "list_directory",
                "description": "Listet den Inhalt eines Verzeichnisses auf",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Der Pfad zum Verzeichnis (Standard: aktuelles Verzeichnis)",
                            "default": "."
                        }
                    }
                }
            },
            "delete_file": {
                "name": "delete_file",
                "description": "Löscht eine Datei",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Der Pfad zur Datei"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            "file_exists": {
                "name": "file_exists",
                "description": "Prüft ob eine Datei existiert",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Der Pfad zur Datei"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            "list_models": {
                "name": "list_models",
                "description": "Listet alle verfügbaren Modelle auf",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "Modell-Typ: 'text', 'image', 'audio' oder None für alle",
                            "enum": ["text", "image", "audio"]
                        }
                    }
                }
            },
            "load_model": {
                "name": "load_model",
                "description": "Lädt ein Modell im Model Service",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "Die ID des Modells (z.B. 'qwen-2.5-7b-instruct')"
                        },
                        "model_type": {
                            "type": "string",
                            "description": "Modell-Typ: 'text', 'image', 'audio'",
                            "enum": ["text", "image", "audio"]
                        }
                    },
                    "required": ["model_id", "model_type"]
                }
            },
            "unload_model": {
                "name": "unload_model",
                "description": "Entlädt ein Modell im Model Service",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "Modell-Typ: 'text', 'image', 'audio'",
                            "enum": ["text", "image", "audio"]
                        }
                    },
                    "required": ["model_type"]
                }
            },
            "model_status": {
                "name": "model_status",
                "description": "Gibt den Status des aktuell geladenen Modells zurück",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "Modell-Typ: 'text', 'image', 'audio'",
                            "enum": ["text", "image", "audio"]
                        }
                    },
                    "required": ["model_type"]
                }
            },
            "chat": {
                "name": "chat",
                "description": "Chat mit lokalem Modell über Model Service",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Die Nachricht"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximale Antwort-Länge",
                            "default": 2048
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Temperature für Generierung (0.0-1.0)",
                            "default": 0.3
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    
    def _send_response(self, response: Dict[str, Any]):
        """Sendet eine JSON-RPC Response"""
        json_str = json.dumps(response, ensure_ascii=False)
        print(json_str, flush=True)
        logger.debug(f"Sent: {json_str}")
    
    def _send_error(self, request_id: Optional[int], code: int, message: str, data: Any = None):
        """Sendet eine JSON-RPC Error Response"""
        error = {
            "code": code,
            "message": message
        }
        if data is not None:
            error["data"] = data
        
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error
        }
        self._send_response(response)
    
    def _send_result(self, request_id: Optional[int], result: Any):
        """Sendet eine JSON-RPC Success Response"""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
        self._send_response(response)
    
    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Behandelt initialize Request"""
        self.initialized = True
        
        # Initialisiere Tools mit Workspace Root
        workspace_root = params.get("workspaceRoot") or os.getcwd()
        try:
            # Initialisiere Tools mit Workspace Root (Manager können None sein)
            initialize_tools(None, None, None, workspace_root)
            logger.info(f"Tools initialisiert mit Workspace Root: {workspace_root}")
        except Exception as e:
            logger.warning(f"Tools konnten nicht initialisiert werden: {e}")
        
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": True
                }
            },
            "serverInfo": {
                "name": "local-ai-mcp-server",
                "version": "1.0.0"
            }
        }
    
    def handle_tools_list(self) -> Dict[str, Any]:
        """Gibt Liste aller verfügbaren Tools zurück"""
        tools_list = list(self.tools.values())
        return {"tools": tools_list}
    
    def handle_tools_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Führt ein Tool aus"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' nicht gefunden")
        
        try:
            # Rufe entsprechendes Tool auf
            if name == "web_search":
                query = arguments.get("query")
                max_results = arguments.get("max_results", 5)
                result = web_search(query, max_results)
                return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]}
            
            elif name == "read_file":
                file_path = arguments.get("file_path")
                content = read_file(file_path)
                return {"content": [{"type": "text", "text": content}]}
            
            elif name == "write_file":
                file_path = arguments.get("file_path")
                content = arguments.get("content")
                success = write_file(file_path, content)
                return {"content": [{"type": "text", "text": f"Datei erfolgreich geschrieben: {file_path}" if success else "Fehler beim Schreiben"}]}
            
            elif name == "list_directory":
                directory_path = arguments.get("directory_path", ".")
                items = list_directory(directory_path)
                items_str = json.dumps(items, ensure_ascii=False, indent=2)
                return {"content": [{"type": "text", "text": items_str}]}
            
            elif name == "delete_file":
                file_path = arguments.get("file_path")
                success = delete_file(file_path)
                return {"content": [{"type": "text", "text": f"Datei erfolgreich gelöscht: {file_path}" if success else "Fehler beim Löschen"}]}
            
            elif name == "file_exists":
                file_path = arguments.get("file_path")
                exists = file_exists(file_path)
                return {"content": [{"type": "text", "text": json.dumps({"exists": exists}, ensure_ascii=False)}]}
            
            elif name == "list_models":
                if not self.model_service.is_available():
                    raise RuntimeError("Model Service nicht verfügbar")
                model_type = arguments.get("model_type")
                if model_type == "text":
                    models = self.model_service.list_text_models()
                elif model_type == "image":
                    models = self.model_service.list_image_models()
                elif model_type == "audio":
                    models = self.model_service.list_audio_models()
                else:
                    # Alle Modelle
                    all_models = {
                        "text": self.model_service.list_text_models() if self.model_service.is_available() else [],
                        "image": self.model_service.list_image_models() if self.model_service.is_available() else [],
                        "audio": self.model_service.list_audio_models() if self.model_service.is_available() else []
                    }
                    return {"content": [{"type": "text", "text": json.dumps(all_models, ensure_ascii=False, indent=2)}]}
                return {"content": [{"type": "text", "text": json.dumps(models, ensure_ascii=False, indent=2)}]}
            
            elif name == "load_model":
                if not self.model_service.is_available():
                    raise RuntimeError("Model Service nicht verfügbar")
                model_id = arguments.get("model_id")
                model_type = arguments.get("model_type")
                if model_type == "text":
                    success = self.model_service.load_text_model(model_id)
                elif model_type == "image":
                    success = self.model_service.load_image_model(model_id)
                elif model_type == "audio":
                    success = self.model_service.load_audio_model(model_id)
                else:
                    raise ValueError(f"Unbekannter Modell-Typ: {model_type}")
                return {"content": [{"type": "text", "text": json.dumps({"success": success, "model_id": model_id, "model_type": model_type}, ensure_ascii=False)}]}
            
            elif name == "unload_model":
                if not self.model_service.is_available():
                    raise RuntimeError("Model Service nicht verfügbar")
                model_type = arguments.get("model_type")
                if model_type == "text":
                    success = self.model_service.unload_text_model()
                elif model_type == "image":
                    success = self.model_service.unload_image_model()
                elif model_type == "audio":
                    success = self.model_service.unload_audio_model()
                else:
                    raise ValueError(f"Unbekannter Modell-Typ: {model_type}")
                return {"content": [{"type": "text", "text": json.dumps({"success": success, "model_type": model_type}, ensure_ascii=False)}]}
            
            elif name == "model_status":
                if not self.model_service.is_available():
                    raise RuntimeError("Model Service nicht verfügbar")
                model_type = arguments.get("model_type")
                if model_type == "text":
                    status = self.model_service.get_text_model_status()
                elif model_type == "image":
                    status = self.model_service.get_image_model_status()
                elif model_type == "audio":
                    status = self.model_service.get_audio_model_status()
                else:
                    raise ValueError(f"Unbekannter Modell-Typ: {model_type}")
                return {"content": [{"type": "text", "text": json.dumps(status, ensure_ascii=False, indent=2)}]}
            
            elif name == "chat":
                if not self.model_service.is_available():
                    raise RuntimeError("Model Service nicht verfügbar")
                message = arguments.get("message")
                max_length = arguments.get("max_length", 2048)
                temperature = arguments.get("temperature", 0.3)
                result = self.model_service.chat(
                    message=message,
                    messages=[{"role": "user", "content": message}],
                    max_length=max_length,
                    temperature=temperature
                )
                if not result:
                    raise RuntimeError("Model Service hat keine Antwort zurückgegeben")
                response_text = result.get("response", "")
                return {"content": [{"type": "text", "text": response_text}]}
            
            else:
                raise ValueError(f"Tool '{name}' nicht implementiert")
                
        except Exception as e:
            logger.error(f"Fehler bei Tool-Ausführung {name}: {e}")
            raise
    
    def handle_chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Behandelt Chat-Request (delegiert an Model Service)"""
        if not self.model_service.is_available():
            raise RuntimeError("Model Service nicht verfügbar")
        
        # Extrahiere letzte User-Nachricht
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content")
                break
        
        if not user_message:
            raise ValueError("Keine User-Nachricht gefunden")
        
        # Rufe Model Service auf
        result = self.model_service.chat(
            message=user_message,
            messages=messages,
            max_length=1024,
            temperature=0.7
        )
        
        if not result:
            raise RuntimeError("Model Service hat keine Antwort zurückgegeben")
        
        response_text = result.get("response", "")
        
        return {
            "content": [{"type": "text", "text": response_text}]
        }
    
    def process_request(self, request: Dict[str, Any]):
        """Verarbeitet eine JSON-RPC Request"""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        
        logger.debug(f"Processing request: {method} (id: {request_id})")
        
        try:
            if method == "initialize":
                if self.initialized:
                    self._send_error(request_id, -32000, "Server bereits initialisiert")
                    return
                result = self.handle_initialize(params)
                self._send_result(request_id, result)
            
            elif method == "tools/list":
                if not self.initialized:
                    self._send_error(request_id, -32002, "Server nicht initialisiert")
                    return
                result = self.handle_tools_list()
                self._send_result(request_id, result)
            
            elif method == "tools/call":
                if not self.initialized:
                    self._send_error(request_id, -32002, "Server nicht initialisiert")
                    return
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = self.handle_tools_call(tool_name, arguments)
                self._send_result(request_id, result)
            
            elif method == "chat":
                if not self.initialized:
                    self._send_error(request_id, -32002, "Server nicht initialisiert")
                    return
                messages = params.get("messages", [])
                result = self.handle_chat(messages)
                self._send_result(request_id, result)
            
            else:
                self._send_error(request_id, -32601, f"Method '{method}' nicht gefunden")
        
        except ValueError as e:
            self._send_error(request_id, -32602, f"Invalid params: {str(e)}")
        except RuntimeError as e:
            self._send_error(request_id, -32000, f"Runtime error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            self._send_error(request_id, -32603, f"Internal error: {str(e)}")
    
    def run(self):
        """Startet den MCP Server (liest von stdin, schreibt nach stdout)"""
        logger.info("MCP Server gestartet (stdio mode)")
        
        # Wrappe stdin für besseres Handling
        stdin = TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
        
        try:
            for line in stdin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    self.process_request(request)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {line[:100]}...")
                    self._send_error(None, -32700, f"Parse error: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing request: {e}", exc_info=True)
                    self._send_error(None, -32603, f"Internal error: {str(e)}")
        
        except KeyboardInterrupt:
            logger.info("MCP Server beendet")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            stdin.close()


if __name__ == "__main__":
    server = MCPServer()
    server.run()

