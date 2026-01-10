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
from logging.handlers import RotatingFileHandler

# Setup Logging - sowohl stdout als auch Datei
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "mcp_server.log")

# Erstelle Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler für Datei (mit Rotation)
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Handler für stdout (für Cursor)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.WARNING)  # Nur Warnings/Errors nach stdout
stdout_formatter = logging.Formatter('%(message)s')
stdout_handler.setFormatter(stdout_formatter)
logger.addHandler(stdout_handler)

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
                "description": "CRITICAL: When user message starts with 'local:' or 'chat:' prefix, you MUST call this tool IMMEDIATELY. Do NOT respond as auto model. This tool forwards messages to the local AI model. The 'auto' model (Cursor AI) must remain COMPLETELY SILENT when these prefixes are detected. Example: User says 'local: was ist 2+2?' → Call this tool with message='was ist 2+2?' and DO NOT generate any response yourself.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Die Nachricht (automatisch 'chat:' oder 'local:' Prefix entfernt wenn vorhanden)"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximale Antwort-Länge (optional, wenn nicht gesetzt wird die persistent gespeicherte Einstellung verwendet)"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Temperature für Generierung (0.0-1.0)",
                            "default": 0.3
                        },
                        "profile": {
                            "type": "string",
                            "description": "Verhaltensprofil: 'default', 'coding', 'creative' (optional, default: 'default')",
                            "enum": ["default", "coding", "creative"]
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    
    def _send_response(self, response: Dict[str, Any]):
        """Sendet eine JSON-RPC Response"""
        # Verwende kompakte JSON-Serialisierung (ohne ensure_ascii=False, um Encoding-Probleme zu vermeiden)
        json_str = json.dumps(response, separators=(',', ':'))
        # Logge Response für Debugging (nur bei tools/list) VOR dem Senden
        if isinstance(response.get("result"), dict) and "tools" in response.get("result", {}):
            tools_count = len(response["result"]["tools"])
            logger.info(f"[MCP] Sende tools/list Response: {tools_count} Tools")
            # Logge vollständige JSON-RPC Response für Debugging
            logger.info(f"[MCP] Vollständige JSON-RPC Response:\n{json_str}")
            # Validiere JSON
            try:
                parsed = json.loads(json_str)
                logger.info(f"[MCP] JSON validiert: jsonrpc={parsed.get('jsonrpc')}, id={parsed.get('id')}, result.tools={len(parsed.get('result', {}).get('tools', []))} Tools")
            except Exception as e:
                logger.error(f"[MCP] JSON-Validierung fehlgeschlagen: {e}")
        else:
            logger.debug(f"Sent: {json_str[:200]}...")
        # WICHTIG: Sende Response an stdout (Cursor liest von stdin/stdout)
        # JSON-RPC über stdio: Jede Nachricht endet mit \r\n (CRLF) gemäß JSON-RPC 2.0 Spec
        sys.stdout.write(json_str + "\r\n")
        sys.stdout.flush()
    
    def _send_error(self, request_id: Optional[Any], code: int, message: str, data: Any = None):
        """Sendet eine JSON-RPC Error Response"""
        # JSON-RPC 2.0: id kann string, number oder null sein
        # Für Notifications (id=null) senden wir keine Response
        if request_id is None:
            # Notification - keine Response senden
            return
        
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
    
    def _send_result(self, request_id: Optional[Any], result: Any):
        """Sendet eine JSON-RPC Success Response"""
        # JSON-RPC 2.0: id kann string, number oder null sein
        # Für Notifications (id=null) senden wir keine Response
        if request_id is None:
            # Notification - keine Response senden
            return
        
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
        # Logge Response für Debugging (nur bei tools/list)
        if isinstance(result, dict) and "tools" in result:
            logger.info(f"[MCP] Sende tools/list Response: {len(result.get('tools', []))} Tools")
        self._send_response(response)
    
    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Behandelt initialize Request"""
        self.initialized = True
        
        # Initialisiere Tools mit Workspace Root
        # Referenziere os explizit am Anfang um UnboundLocalError zu vermeiden
        _ = os  # Stelle sicher, dass os als global behandelt wird
        workspace_root = params.get("workspaceRoot") or os.getcwd()
        try:
            # Initialisiere Tools mit Workspace Root (Manager können None sein)
            initialize_tools(None, None, None, workspace_root)
            logger.info(f"Tools initialisiert mit Workspace Root: {workspace_root}")
        except Exception as e:
            logger.warning(f"Tools konnten nicht initialisiert werden: {e}")
        
        # Bestimme Modell-Name dynamisch
        model_name = "Local AI"
        model_id = None
        if self.model_service.is_available():
            try:
                status = self.model_service.get_text_model_status()
                if status.get("loaded") and status.get("model_id"):
                    model_id = status.get("model_id")
                    model_name = f"Local AI ({model_id})"
            except Exception as e:
                logger.debug(f"Konnte Modell-Status nicht abrufen: {e}")
        
        # Lade MCP Settings um auto_model_silent_mode zu prüfen
        try:
            mcp_settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mcp_settings.json")
            auto_silent = True  # Default
            if os.path.exists(mcp_settings_file):
                with open(mcp_settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    auto_silent = settings.get("auto_model_silent_mode", True)
        except Exception as e:
            logger.debug(f"Konnte MCP Settings nicht laden: {e}")
            auto_silent = True
        
        # Erstelle Response mit serverInfo (WICHTIG für Cursor)
        # Verwende protocolVersion aus Request oder Standard
        requested_version = params.get("protocolVersion", "2024-11-05")
        # Unterstütze sowohl 2024-11-05 als auch 2025-11-25
        if requested_version.startswith("2025"):
            protocol_version = "2025-11-25"
        else:
            protocol_version = "2024-11-05"
        
        # Erstelle Response mit serverInfo (WICHTIG für Cursor)
        # Minimale Struktur - nur das Nötigste für Cursor
        result = {
            "protocolVersion": protocol_version,
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
        
        # Füge completionProvider nur hinzu, wenn Modell geladen ist
        if model_id:
            result["capabilities"]["completion"] = {
                "completionProvider": {
                    "model": "local-ai",
                    "name": model_name
                }
            }
        
        # Logge serverInfo für Debugging
        logger.info(f"[MCP] Initialize Response: protocolVersion={protocol_version}, serverInfo={result['serverInfo']}")
        
        return result
    
    def handle_tools_list(self) -> Dict[str, Any]:
        """Gibt Liste aller verfügbaren Tools zurück"""
        tools_list = list(self.tools.values())
        logger.info(f"[MCP] Tools-Liste: {len(tools_list)} Tools gefunden")
        
        # Validiere Tool-Struktur
        for tool in tools_list:
            if not isinstance(tool, dict):
                logger.error(f"[MCP] Tool ist kein Dict: {type(tool)}")
                continue
            if "name" not in tool:
                logger.error(f"[MCP] Tool hat kein 'name' Feld: {tool}")
            if "inputSchema" not in tool:
                logger.error(f"[MCP] Tool '{tool.get('name')}' hat kein 'inputSchema' Feld")
        
        result = {"tools": tools_list}
        # Logge vollständige Response für Debugging
        try:
            response_json = json.dumps(result, ensure_ascii=False, indent=2)
            logger.info(f"[MCP] Tools-Liste Response (erste 1000 Zeichen):\n{response_json[:1000]}...")
        except Exception as e:
            logger.error(f"[MCP] Fehler beim Serialisieren der Tools-Liste: {e}")
        
        return result
    
    def _check_model_service(self) -> None:
        """Prüft ob Model Service verfügbar ist, wirft RuntimeError wenn nicht"""
        if not self.model_service.is_available():
            raise RuntimeError("Model Service nicht verfügbar")
    
    def _format_text_response(self, content: Any) -> Dict[str, Any]:
        """Formatiert Text-Inhalt als MCP Response"""
        if isinstance(content, (dict, list)):
            text = json.dumps(content, ensure_ascii=False, indent=2)
        else:
            text = str(content)
        return {"content": [{"type": "text", "text": text}]}
    
    def _strip_local_prefix(self, message: str) -> tuple[str, bool]:
        """
        Entfernt "local:" oder "chat:" Prefix falls vorhanden
        
        Returns:
            (cleaned_message, was_local_prefix)
        """
        if not message:
            return message, False
        
        message_stripped = message.strip()
        if message_stripped.lower().startswith("local:"):
            cleaned = message_stripped[6:].strip()  # Entferne "local:" (6 Zeichen)
            logger.info(f"[MCP] 'local:' Prefix erkannt - verwende lokales Modell für: {cleaned[:50]}...")
            return cleaned, True
        elif message_stripped.lower().startswith("chat:"):
            cleaned = message_stripped[5:].strip()  # Entferne "chat:" (5 Zeichen)
            logger.info(f"[MCP] 'chat:' Prefix erkannt - verwende lokales Modell für: {cleaned[:50]}...")
            return cleaned, True
        return message, False
    
    def handle_tools_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Führt ein Tool aus"""
        logger.info(f"[MCP] Tool-Aufruf: {name} mit Arguments: {arguments}")
        if name not in self.tools:
            logger.error(f"[MCP] Tool '{name}' nicht gefunden")
            raise ValueError(f"Tool '{name}' nicht gefunden")
        
        try:
            # Rufe entsprechendes Tool auf
            if name == "web_search":
                query = arguments.get("query")
                if not query:
                    raise ValueError("query ist erforderlich")
                max_results = arguments.get("max_results", 5)
                result = web_search(query, max_results)
                return self._format_text_response(result)
            
            elif name == "read_file":
                file_path = arguments.get("file_path")
                if not file_path:
                    raise ValueError("file_path ist erforderlich")
                content = read_file(file_path)
                return self._format_text_response(content)
            
            elif name == "write_file":
                file_path = arguments.get("file_path")
                content = arguments.get("content")
                if not file_path:
                    raise ValueError("file_path ist erforderlich")
                if content is None:
                    raise ValueError("content ist erforderlich")
                success = write_file(file_path, content)
                message = f"Datei erfolgreich geschrieben: {file_path}" if success else "Fehler beim Schreiben"
                return self._format_text_response(message)
            
            elif name == "list_directory":
                directory_path = arguments.get("directory_path", ".")
                items = list_directory(directory_path)
                return self._format_text_response(items)
            
            elif name == "delete_file":
                file_path = arguments.get("file_path")
                if not file_path:
                    raise ValueError("file_path ist erforderlich")
                success = delete_file(file_path)
                message = f"Datei erfolgreich gelöscht: {file_path}" if success else "Fehler beim Löschen"
                return self._format_text_response(message)
            
            elif name == "file_exists":
                file_path = arguments.get("file_path")
                if not file_path:
                    raise ValueError("file_path ist erforderlich")
                exists = file_exists(file_path)
                return self._format_text_response({"exists": exists})
            
            elif name == "list_models":
                self._check_model_service()
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
                        "text": self.model_service.list_text_models(),
                        "image": self.model_service.list_image_models(),
                        "audio": self.model_service.list_audio_models()
                    }
                    return self._format_text_response(all_models)
                return self._format_text_response(models)
            
            elif name == "load_model":
                self._check_model_service()
                model_id = arguments.get("model_id")
                model_type = arguments.get("model_type")
                if not model_id or not model_type:
                    raise ValueError("model_id und model_type sind erforderlich")
                
                if model_type == "text":
                    success = self.model_service.load_text_model(model_id)
                elif model_type == "image":
                    success = self.model_service.load_image_model(model_id)
                elif model_type == "audio":
                    success = self.model_service.load_audio_model(model_id)
                else:
                    raise ValueError(f"Unbekannter Modell-Typ: {model_type}")
                return self._format_text_response({"success": success, "model_id": model_id, "model_type": model_type})
            
            elif name == "unload_model":
                self._check_model_service()
                model_type = arguments.get("model_type")
                if not model_type:
                    raise ValueError("model_type ist erforderlich")
                
                if model_type == "text":
                    success = self.model_service.unload_text_model()
                elif model_type == "image":
                    success = self.model_service.unload_image_model()
                elif model_type == "audio":
                    success = self.model_service.unload_audio_model()
                else:
                    raise ValueError(f"Unbekannter Modell-Typ: {model_type}")
                return self._format_text_response({"success": success, "model_type": model_type})
            
            elif name == "model_status":
                self._check_model_service()
                model_type = arguments.get("model_type")
                if not model_type:
                    raise ValueError("model_type ist erforderlich")
                
                if model_type == "text":
                    status = self.model_service.get_text_model_status()
                elif model_type == "image":
                    status = self.model_service.get_image_model_status()
                elif model_type == "audio":
                    status = self.model_service.get_audio_model_status()
                else:
                    raise ValueError(f"Unbekannter Modell-Typ: {model_type}")
                return self._format_text_response(status)
            
            elif name == "chat":
                logger.info(f"[MCP] Chat-Request erhalten: {arguments}")
                self._check_model_service()
                
                message = arguments.get("message")
                if not message:
                    raise ValueError("Message-Parameter fehlt")
                
                logger.info(f"[MCP] Original message: {message}")
                
                # Entferne "local:" oder "chat:" Prefix falls vorhanden
                message, was_local = self._strip_local_prefix(message)
                logger.info(f"[MCP] Nach Prefix-Entfernung: {message}, was_local={was_local}")
                
                # Extrahiere Parameter
                # WICHTIG: max_length=None, damit Model Service die persistent Settings verwenden kann
                max_length = arguments.get("max_length")  # None wenn nicht gesetzt -> Model Service verwendet Settings
                temperature = arguments.get("temperature")  # None wenn nicht gesetzt -> Model Service verwendet Profil
                profile = arguments.get("profile")
                logger.info(f"[MCP] Parameter: max_length={max_length} (None = Settings verwenden), temperature={temperature}, profile={profile}")
                
                # Rufe Model Service auf
                logger.info(f"[MCP] Sende Request an Model Service: message='{message[:50]}...'")
                result = self.model_service.chat(
                    message=message,
                    messages=[{"role": "user", "content": message}],
                    max_length=max_length,
                    temperature=temperature,
                    profile=profile
                )
                
                if not result:
                    logger.error("[MCP] Model Service hat keine Antwort zurückgegeben")
                    raise RuntimeError("Model Service hat keine Antwort zurückgegeben")
                
                response_text = result.get("response", "")
                logger.info(f"[MCP] Response-Text Länge: {len(response_text)} Zeichen")
                return self._format_text_response(response_text)
            
            else:
                raise ValueError(f"Tool '{name}' nicht implementiert")
                
        except Exception as e:
            logger.error(f"Fehler bei Tool-Ausführung {name}: {e}")
            raise
    
    def handle_chat(self, messages: List[Dict[str, str]], max_length: int = 2048, temperature: float = 0.3, profile: Optional[str] = None) -> Dict[str, Any]:
        """
        Behandelt Chat-Request (delegiert an Model Service)
        
        Args:
            messages: Liste von Nachrichten mit 'role' und 'content'
            max_length: Maximale Antwort-Länge (default: 2048)
            temperature: Temperature für Generierung (default: 0.3)
            profile: Verhaltensprofil (optional)
        """
        self._check_model_service()
        
        # Extrahiere letzte User-Nachricht
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content")
                break
        
        if not user_message:
            raise ValueError("Keine User-Nachricht gefunden")
        
        # Entferne "local:" oder "chat:" Prefix falls vorhanden
        user_message, was_local = self._strip_local_prefix(user_message)
        
        # Aktualisiere Messages-Liste mit bereinigter Nachricht
        cleaned_messages = messages.copy()
        for msg in reversed(cleaned_messages):
            if msg.get("role") == "user":
                msg["content"] = user_message
                break
        
        # Rufe Model Service auf
        logger.info(f"[MCP] Chat via handle_chat: max_length={max_length}, temperature={temperature}, profile={profile}")
        result = self.model_service.chat(
            message=user_message,
            messages=cleaned_messages,
            max_length=max_length,
            temperature=temperature,
            profile=profile
        )
        
        if not result:
            raise RuntimeError("Model Service hat keine Antwort zurückgegeben")
        
        response_text = result.get("response", "")
        return self._format_text_response(response_text)
    
    def process_request(self, request: Dict[str, Any]):
        """Verarbeitet eine JSON-RPC Request"""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        
        # Prüfe ob Request gültig ist
        if method is None:
            logger.warning("[MCP] Request ohne 'method' Feld erhalten")
            if request_id is not None:
                self._send_error(request_id, -32600, "Invalid Request: 'method' fehlt")
            return
        
        logger.info(f"[MCP] Processing request: {method} (id: {request_id})")
        
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
                max_length = params.get("max_length", 2048)
                temperature = params.get("temperature", 0.3)
                profile = params.get("profile")
                result = self.handle_chat(messages, max_length=max_length, temperature=temperature, profile=profile)
                self._send_result(request_id, result)
            
            elif method == "completion/complete":
                # Cursor Completion Request - delegiert an handle_chat
                if not self.initialized:
                    self._send_error(request_id, -32002, "Server nicht initialisiert")
                    return
                # Cursor sendet completion requests mit prompt oder messages
                prompt = params.get("prompt")
                messages = params.get("messages", [])
                
                # Konvertiere prompt zu messages falls nötig
                if prompt and not messages:
                    messages = [{"role": "user", "content": prompt}]
                
                if messages:
                    result = self.handle_chat(messages)
                    self._send_result(request_id, result)
                else:
                    self._send_error(request_id, -32602, "Keine Nachricht oder Prompt gefunden")
            
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
        logger.info("[MCP] MCP Server gestartet (stdio mode)")
        try:
            # Prüfe Model Service Verfügbarkeit (ohne auf Attribute zuzugreifen die nicht existieren)
            is_available = self.model_service.is_available()
            logger.info(f"[MCP] Model Service verfügbar: {is_available}")
        except Exception as e:
            logger.warning(f"[MCP] Konnte Model Service Verfügbarkeit nicht prüfen: {e}")
        
        # Wrappe stdin für besseres Handling
        stdin = TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
        
        try:
            for line in stdin:
                line = line.strip()
                if not line:
                    continue
                
                logger.info(f"[MCP] Request erhalten: {line[:200]}...")
                try:
                    request = json.loads(line)
                    logger.info(f"[MCP] Request geparst: method={request.get('method')}, id={request.get('id')}")
                    self.process_request(request)
                except json.JSONDecodeError as e:
                    logger.error(f"[MCP] Invalid JSON: {line[:100]}...")
                    self._send_error(None, -32700, f"Parse error: {str(e)}")
                except Exception as e:
                    logger.error(f"[MCP] Error processing request: {e}", exc_info=True)
                    self._send_error(None, -32603, f"Internal error: {str(e)}")
        
        except KeyboardInterrupt:
            logger.info("[MCP] MCP Server beendet")
        except Exception as e:
            logger.error(f"[MCP] Fatal error: {e}", exc_info=True)
        finally:
            stdin.close()


if __name__ == "__main__":
    server = MCPServer()
    server.run()

