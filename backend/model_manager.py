"""
Model Manager - Verwaltet das Laden und Wechseln von AI-Modellen
"""
import json
import os
import time
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Verwaltet AI-Modelle - lädt sie bei Bedarf und hält sie im Speicher"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.current_model_id: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Verwende Device: {self.device}")
        
        # Lade Performance-Einstellungen
        self._apply_performance_settings()
        
        # Wende GPU-Optimierungen an
        self._apply_gpu_optimizations()
    
    def _load_config(self) -> Dict[str, Any]:
        """Lädt die Konfiguration aus config.json"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config-Datei nicht gefunden: {self.config_path}")
            return {}
    
    def _apply_performance_settings(self):
        """Wendet Performance-Einstellungen an"""
        try:
            perf_settings_path = os.path.join(os.path.dirname(os.path.dirname(self.config_path)), "data", "performance_settings.json")
            if os.path.exists(perf_settings_path):
                with open(perf_settings_path, 'r', encoding='utf-8') as f:
                    perf_settings = json.load(f)
                    cpu_threads = perf_settings.get("cpu_threads")
                    if cpu_threads and cpu_threads > 0:
                        torch.set_num_threads(cpu_threads)
                        torch.set_num_interop_threads(cpu_threads)
                        logger.info(f"CPU-Threads auf {cpu_threads} gesetzt")
                    else:
                        # Auto
                        import os as os_module
                        num_threads = os_module.cpu_count() or 4
                        torch.set_num_threads(num_threads)
                        torch.set_num_interop_threads(num_threads)
        except Exception as e:
            logger.warning(f"Fehler beim Anwenden der Performance-Einstellungen: {e}")
    
    def _apply_gpu_optimizations(self):
        """Wendet GPU-Optimierungen an (cudnn.benchmark, tf32)"""
        if self.device == "cuda":
            # CUDNN Benchmark für konsistente Input-Größen
            torch.backends.cudnn.benchmark = True
            logger.info("CUDNN Benchmark aktiviert")
            
            # TF32 für Ampere+ GPUs (RTX 30xx, A100, etc.)
            if torch.cuda.is_available():
                try:
                    props = torch.cuda.get_device_properties(0)
                    compute_cap = props.major * 10 + props.minor
                    if compute_cap >= 80:  # Ampere (8.0) oder höher
                        torch.backends.cuda.matmul.allow_tf32 = True
                        logger.info(f"TF32 aktiviert für Ampere+ GPU (Compute Capability: {compute_cap})")
                    else:
                        logger.info(f"TF32 nicht verfügbar für GPU (Compute Capability: {compute_cap}, benötigt >= 8.0)")
                except Exception as e:
                    logger.warning(f"Fehler beim Prüfen der GPU-Generation für TF32: {e}")
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Gibt alle verfügbaren Modelle zurück"""
        return self.config.get("models", {})
    
    def get_current_model(self) -> Optional[str]:
        """Gibt die ID des aktuell geladenen Modells zurück"""
        return self.current_model_id
    
    def is_model_loaded(self) -> bool:
        """Prüft ob ein Modell geladen ist"""
        return self.model is not None and self.tokenizer is not None
    
    def health_check(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Prüft ob Modell wirklich funktioniert (echte Funktionsprüfung)
        
        Args:
            timeout: Timeout in Sekunden für Health-Check
            
        Returns:
            {
                "healthy": bool,
                "response_time_ms": float,
                "error": Optional[str],
                "last_check": float
            }
        """
        if not self.is_model_loaded():
            return {
                "healthy": False,
                "response_time_ms": 0,
                "error": "Modell nicht geladen",
                "last_check": time.time()
            }
        
        try:
            import time
            start_time = time.time()
            
            # Test mit minimalem Prompt
            test_messages = [{"role": "user", "content": "Test"}]
            response = self.generate(test_messages, max_length=10, temperature=0.0)
            
            response_time = (time.time() - start_time) * 1000
            
            # Prüfe ob Antwort valide ist
            if response and len(response) > 0 and response_time < timeout * 1000:
                return {
                    "healthy": True,
                    "response_time_ms": response_time,
                    "error": None,
                    "last_check": time.time()
                }
            else:
                return {
                    "healthy": False,
                    "response_time_ms": response_time,
                    "error": f"Antwort ungültig oder zu langsam ({response_time:.0f}ms)",
                    "last_check": time.time()
                }
        except Exception as e:
            return {
                "healthy": False,
                "response_time_ms": 0,
                "error": str(e),
                "last_check": time.time()
            }
    
    def load_model(self, model_id: str) -> bool:
        """
        Lädt ein Modell. Wenn bereits ein Modell geladen ist, wird es entladen.
        
        Args:
            model_id: Die ID des Modells aus der Config
            
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if model_id not in self.config.get("models", {}):
            logger.error(f"Modell nicht gefunden: {model_id}")
            return False
        
        model_info = self.config["models"][model_id]
        model_path = model_info["path"]
        
        if not os.path.exists(model_path):
            logger.error(f"Modell-Pfad existiert nicht: {model_path}")
            return False
        
        try:
            # Altes Modell entladen (Speicher freigeben)
            if self.model is not None:
                logger.info("Entlade aktuelles Modell...")
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.model = None
                self.tokenizer = None
            
            logger.info(f"Lade Modell: {model_id} von {model_path}")
            
            # Prüfe Performance-Settings für Flash Attention 2
            perf_settings_path = os.path.join(os.path.dirname(os.path.dirname(self.config_path)), "data", "performance_settings.json")
            use_flash_attention = True  # Default
            if os.path.exists(perf_settings_path):
                try:
                    with open(perf_settings_path, 'r', encoding='utf-8') as f:
                        perf_settings = json.load(f)
                        use_flash_attention = perf_settings.get("use_flash_attention", True)
                except:
                    pass
            
            # Prüfe ob Flash Attention 2 verfügbar ist
            flash_attention_available = False
            if use_flash_attention:
                try:
                    from flash_attn import flash_attn_func
                    flash_attention_available = True
                    logger.info("Flash Attention 2 ist verfügbar")
                except ImportError:
                    logger.info("Flash Attention 2 nicht verfügbar, verwende Standard-Attention")
            
            # Tokenizer laden
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Prüfe Performance-Settings für Quantisierung
            use_quantization = False
            quantization_bits = 8
            try:
                perf_settings_path = os.path.join(os.path.dirname(os.path.dirname(self.config_path)), "data", "performance_settings.json")
                if os.path.exists(perf_settings_path):
                    with open(perf_settings_path, 'r', encoding='utf-8') as f:
                        perf_settings = json.load(f)
                        use_quantization = perf_settings.get("use_quantization", False)
                        quantization_bits = perf_settings.get("quantization_bits", 8)
            except:
                pass
            
            # Modell laden
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True
            }
            
            # Quantisierung (8-bit/4-bit) mit bitsandbytes
            if use_quantization and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    
                    if quantization_bits == 8:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.bfloat16
                        )
                        logger.info("8-bit Quantisierung wird verwendet")
                    elif quantization_bits == 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        logger.info("4-bit Quantisierung wird verwendet")
                    else:
                        logger.warning(f"Unbekannte Quantisierungs-Bits: {quantization_bits}, verwende 8-bit")
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.bfloat16
                        )
                    
                    model_kwargs["quantization_config"] = quantization_config
                except ImportError:
                    logger.warning("bitsandbytes nicht verfügbar, Quantisierung wird übersprungen")
                except Exception as e:
                    logger.warning(f"Fehler bei Quantisierung: {e}, verwende unquantisiertes Modell")
            
            # Flash Attention 2 wird automatisch von Transformers verwendet wenn verfügbar
            # Wir müssen nur sicherstellen, dass es aktiviert ist
            if flash_attention_available:
                # Transformers aktiviert Flash Attention automatisch wenn verfügbar
                # Für explizite Aktivierung können wir attn_implementation setzen (Transformers 4.36+)
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Flash Attention 2 wird für Modell-Laden aktiviert")
                except:
                    # Fallback wenn attn_implementation nicht unterstützt wird
                    logger.debug("attn_implementation nicht unterstützt, Flash Attention wird automatisch verwendet wenn verfügbar")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Prüfe Performance-Settings für torch.compile()
            use_torch_compile = False
            try:
                perf_settings_path = os.path.join(os.path.dirname(os.path.dirname(self.config_path)), "data", "performance_settings.json")
                if os.path.exists(perf_settings_path):
                    with open(perf_settings_path, 'r', encoding='utf-8') as f:
                        perf_settings = json.load(f)
                        use_torch_compile = perf_settings.get("use_torch_compile", False)
            except:
                pass
            
            # torch.compile() Support (PyTorch 2.0+)
            if use_torch_compile:
                try:
                    # Prüfe PyTorch Version
                    torch_version = torch.__version__.split('.')
                    major_version = int(torch_version[0])
                    minor_version = int(torch_version[1]) if len(torch_version) > 1 else 0
                    
                    if major_version >= 2:
                        # Kompiliere Modell für bessere Performance
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                        logger.info("Modell mit torch.compile() optimiert")
                    else:
                        logger.warning(f"torch.compile() erfordert PyTorch 2.0+, aktuelle Version: {torch.__version__}")
                except Exception as e:
                    logger.warning(f"Fehler bei torch.compile(): {e}, verwende unkompiliertes Modell")
            
            self.current_model_id = model_id
            logger.info(f"Modell erfolgreich geladen: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            self.model = None
            self.tokenizer = None
            return False
    
    def generate(self, messages: List[Dict[str, str]], max_length: int = 2048, temperature: float = 0.3, max_retries: int = 2) -> str:
        """
        Generiert eine Antwort basierend auf Messages (Chat-Format) mit Validierung und Retry-Mechanismus
        
        Args:
            messages: Liste von Messages im Format [{"role": "user", "content": "..."}, ...]
            max_length: Maximale Länge der Antwort
            temperature: Kreativität (0.0 = deterministisch, 1.0 = kreativ) - niedriger = konsistenter
            max_retries: Maximale Anzahl von Retries bei ungültigen Antworten
            
        Returns:
            Die generierte Antwort (garantiert nicht leer)
        """
        if not self.is_model_loaded():
            raise RuntimeError("Kein Modell geladen!")
        
        current_max_length = max_length
        for attempt in range(max_retries + 1):
            try:
                # Normale Generierung
                response = self._generate_internal(messages, current_max_length, temperature)
                
                # Validierung
                validation_result = self._validate_response(response, messages)
                
                if validation_result:
                    # Response ist gültig, prüfe Vollständigkeit
                    completeness = self._check_completeness(response, messages)
                    
                    if completeness["complete"]:
                        return response
                    else:
                        # Response ist unvollständig
                        if attempt < max_retries and completeness.get("suggested_max_length"):
                            logger.warning(f"Response unvollständig: {completeness['reason']}, retry mit max_length={completeness['suggested_max_length']}")
                            current_max_length = completeness["suggested_max_length"]
                            continue
                        else:
                            # Response ist unvollständig, aber keine Retries mehr
                            logger.warning(f"Response unvollständig, aber keine Retries mehr: {completeness['reason']}")
                            return response  # Gebe trotzdem zurück, da teilweise gültig
                else:
                    logger.warning(f"Ungültige Response bei Versuch {attempt + 1}/{max_retries + 1}, retry...")
                    if attempt < max_retries:
                        # Erhöhe max_length für Retry
                        current_max_length = int(current_max_length * 1.5)
                        continue
                    else:
                        # Letzter Versuch fehlgeschlagen - Exception werfen statt Fallback
                        raise RuntimeError("Konnte nach mehreren Versuchen keine gültige Antwort generieren")
            except Exception as e:
                #logger.error(f"Fehler bei Generierung (Versuch {attempt + 1}): {e}")
                if attempt < max_retries:
                    current_max_length = int(current_max_length * 1.5)
                    continue
                else:
                    raise
        
        # Sollte nie erreicht werden
        raise RuntimeError("Konnte nach mehreren Versuchen keine gültige Antwort generieren")
    
    def _validate_response(self, response: str, messages: List[Dict[str, str]]) -> bool:
        """
        Validiert ob Response gültig ist
        
        Returns:
            True wenn Response gültig, False sonst
        """
        # Prüfe ob Response leer ist
        if not response or len(response.strip()) == 0:
            logger.debug("[Validation] Response ist leer")
            return False
        
        # Prüfe ob Response nur Whitespace ist
        if response.strip() == "":
            logger.debug("[Validation] Response ist nur Whitespace")
            return False
        
        # Prüfe ob Response zu kurz ist (wahrscheinlich abgeschnitten)
        # Reduziert von 10 auf 5 Zeichen - Mistral kann sehr kurze, gültige Antworten geben
        response_stripped = response.strip()
        if len(response_stripped) < 5:
            logger.debug(f"[Validation] Response zu kurz: {len(response_stripped)} Zeichen")
            return False
        
        # Prüfe ob Response vollständig ist (endet mit Satzzeichen oder ist vollständiger Satz)
        # Entspannt: Akzeptiere auch Antworten ohne Satzzeichen, wenn das letzte Wort vollständig ist
        if not response_stripped[-1] in ['.', '!', '?', ':', ';']:
            # Prüfe ob letztes Wort vollständig ist (kein abgeschnittenes Wort)
            words = response_stripped.split()
            if words:
                last_word = words[-1]
                # Reduziert von 3 auf 2 Zeichen - auch sehr kurze Wörter können gültig sein
                if len(last_word) < 2:  # Sehr kurzes letztes Wort = wahrscheinlich abgeschnitten
                    logger.debug(f"[Validation] Letztes Wort zu kurz: '{last_word}'")
                    return False
            # Wenn keine Wörter vorhanden, ist es ungültig
            elif len(response_stripped) < 3:
                logger.debug("[Validation] Response hat keine vollständigen Wörter")
                return False
        
        # Prüfe ob Response nicht nur System-Prompt-Phrasen enthält
        system_phrases = ["du bist ein hilfreicher", "ai-assistent", "antworte klar"]
        if all(phrase in response.lower() for phrase in system_phrases) and len(response.strip()) < 50:
            logger.debug("[Validation] Response enthält nur System-Prompt-Phrasen")
            return False
        
        # Prüfe ob Response nicht nur die User-Nachricht wiederholt
        if messages:
            last_user = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
            if last_user and last_user["content"].strip().lower() == response.strip().lower():
                logger.debug("[Validation] Response wiederholt nur User-Nachricht")
                return False
        
        logger.debug(f"[Validation] Response ist gültig: {len(response_stripped)} Zeichen")
        return True
    
    def _check_completeness(self, response: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Prüft ob Response vollständig ist
        
        Returns:
            {
                "complete": bool,
                "reason": str,
                "suggested_max_length": Optional[int]
            }
        """
        response_stripped = response.strip()
        
        # Prüfe ob Response mit unvollständigem Satz endet
        incomplete_indicators = [
            response_stripped.endswith(','),
            response_stripped.endswith('und'),
            response_stripped.endswith('oder'),
            response_stripped.endswith('aber'),
        ]
        
        # Prüfe auf unvollständige Wörter (abgeschnitten)
        words = response_stripped.split()
        if words:
            last_word = words[-1]
            if len(last_word) < 3:  # Sehr kurzes letztes Wort = wahrscheinlich abgeschnitten
                incomplete_indicators.append(True)
        
        if any(incomplete_indicators):
            # 🔧 FIX: Intelligente suggested_max_length basierend auf aktueller Response-Länge
            # Verwende Token-Count (grobe Schätzung: Wörter * 1.3) statt Wort-Count
            # Minimum 512 Tokens um sinnvolle Antworten zu ermöglichen
            word_count = len(response_stripped.split())
            estimated_tokens = int(word_count * 1.3)  # Grobe Token-Schätzung
            suggested = max(512, estimated_tokens * 2)  # Mindestens 512, sonst doppelt so viele Tokens
            
            return {
                "complete": False,
                "reason": "Response endet mit unvollständigem Satz/Wort",
                "suggested_max_length": suggested
            }
        
        # Prüfe ob Response zu kurz ist für die Frage
        # (Wenn Frage lang ist, sollte Antwort auch eine gewisse Länge haben)
        if messages:
            last_user = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
            if last_user:
                question_length = len(last_user["content"])
                if len(response_stripped) < 50 and len(response_stripped.split()) < 10 and question_length > 50:
                    return {
                        "complete": False,
                        "reason": "Response zu kurz für vollständige Antwort",
                        "suggested_max_length": 1024
                    }
        
        return {
            "complete": True,
            "reason": "Response erscheint vollständig",
            "suggested_max_length": None
        }
    
    def _clean_response_minimal(self, response: str, messages: List[Dict[str, str]], original_prompt: str = "") -> str:
        """
        🔧 NEUE MINIMALISTISCHE RESPONSE-BEREINIGUNG
        
        Macht NUR das Absolute Minimum an Bereinigung.
        Optionale Cleanings sind kommentiert und können bei Bedarf aktiviert werden.
        
        Args:
            response: Die rohe Response vom Modell
            messages: Die Message-History
            original_prompt: Der originale Prompt (optional)
            
        Returns:
            Bereinigte Response
        """
        # ============================================================================
        # MINIMALISTISCHE BASIS-BEREINIGUNG (IMMER AKTIV)
        # ============================================================================
        
        # 1. Entferne Chat-Template-Markierungen (KRITISCH für Qwen/LLaMA)
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "")
        
        # 2. Entferne führende/trailing Whitespace
        response = response.strip()
        
        # 3. Entferne "assistant:" Präfix falls vorhanden
        if response.lower().startswith('assistant:'):
            response = response[10:].strip()
        elif response.lower().startswith('assistant '):
            response = response[9:].strip()
        
        logger.info(f"[Clean] Minimal cleaning done: {len(response)} chars")
        
        # ============================================================================
        # OPTIONALE CLEANINGS (KOMMENTIERT - BEI BEDARF AKTIVIEREN)
        # ============================================================================
        
        # OPTION A: Entferne "assistant" Marker in der Mitte der Response
        # Nutzen: Wenn Modell vollständigen Chat generiert mit Markern
        # Risiko: Niedrig - nur Marker werden entfernt
        # AKTIVIEREN: Entferne die # vor den nächsten Zeilen
        # response_lower = response.lower()
        # for marker in ["assistant ", "assistant:", "assistant\n"]:
        #     pos = response_lower.find(marker)
        #     if pos > 0:  # Nicht am Anfang
        #         response = response[pos + len(marker):].strip()
        #         logger.debug(f"[Clean] 'assistant' marker removed from middle")
        #         break
        
        # OPTION B: Entferne System-Prompt-Phrasen
        # Nutzen: Wenn Modell System-Prompt in Response wiederholt
        # Risiko: MITTEL - Könnte legitime Antworten über "AI-Assistent" beschädigen
        # AKTIVIEREN: Entferne die # vor den nächsten Zeilen
        # system_phrases = ["Du bist ein hilfreicher", "AI-Assistent"]
        # for phrase in system_phrases:
        #     if phrase in response:
        #         lines = response.split('\n')
        #         response = '\n'.join([l for l in lines if phrase not in l]).strip()
        #         logger.debug(f"[Clean] System phrase '{phrase}' removed")
        
        # OPTION C: Stoppe bei mehrfachen Nachrichten (User:/Assistant:/System:)
        # Nutzen: Wenn Modell mehrere Chat-Turns generiert
        # Risiko: Niedrig - stoppt nur bei klaren Markern
        # AKTIVIEREN: Entferne die # vor den nächsten Zeilen
        # lines = response.split('\n')
        # cleaned_lines = []
        # for line in lines:
        #     if line.strip().startswith(('User:', 'Assistant:', 'System:')):
        #         break
        #     cleaned_lines.append(line)
        # response = '\n'.join(cleaned_lines).strip()
        # logger.debug(f"[Clean] Multiple messages stopped")
        
        # OPTION D: Entferne Prompt-Reste (falls Prompt in Response enthalten)
        # Nutzen: Wenn Tokenizer Prompt nicht korrekt entfernt
        # Risiko: HOCH - Könnte legitime Response-Teile entfernen die Prompt ähneln
        # AKTIVIEREN: Entferne die # vor den nächsten Zeilen
        # if original_prompt:
        #     prompt_start = original_prompt[:50]
        #     if prompt_start in response:
        #         response = response.replace(prompt_start, "").strip()
        #         logger.debug(f"[Clean] Prompt removed")
        
        # OPTION E: Schneide bei doppeltem Zeilenumbruch + Fragewort ab
        # Nutzen: Wenn Modell nach Antwort neue Fragen generiert
        # Risiko: MITTEL - Könnte mehrteilige Antworten abschneiden
        # AKTIVIEREN: Entferne die # vor den nächsten Zeilen
        # if '\n\n' in response:
        #     sections = response.split('\n\n')
        #     if len(sections) > 1:
        #         second = sections[1].strip().lower()
        #         if second.startswith(('#', 'wie', 'was', 'user', 'system')):
        #             response = sections[0].strip()
        #             logger.debug(f"[Clean] Multiple sections cut at question")
        
        return response
    
    def _generate_internal(self, messages: List[Dict[str, str]], max_length: int = 2048, temperature: float = 0.3) -> str:
        """
        Interne Generierungsmethode (ohne Validierung/Retry)
        
        Args:
            messages: Liste von Messages im Format [{"role": "user", "content": "..."}, ...]
            max_length: Maximale Länge der Antwort
            temperature: Kreativität (0.0 = deterministisch, 1.0 = kreativ) - niedriger = konsistenter
            
        Returns:
            Die generierte Antwort
        """
        try:
            ## Verwende Chat-Template wenn verfügbar (für Qwen, Phi-3, etc.)
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                # Moderne Chat-Modelle mit Chat-Template
                # WICHTIG: Für Qwen müssen wir sicherstellen, dass nur die Assistant-Antwort generiert wird
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                # Speichere Prompt-Länge für späteren Vergleich
                original_prompt = prompt
                
                #else:
                # Fallback für ältere Modelle
                prompt_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                prompt = "\n".join(prompt_parts) + "\nAssistant:"
                original_prompt = prompt
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # Stelle sicher, dass alle Inputs auf dem richtigen Device sind
            # WICHTIG: Wenn device_map="auto" verwendet wird, müssen Inputs auf dem ersten Device sein
            if hasattr(self.model, 'device'):
                # Modell hat ein device-Attribut (wenn device_map nicht verwendet wird)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            elif hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                # Modell verwendet device_map="auto" - Inputs auf erstes Device
                first_device = list(self.model.hf_device_map.values())[0]
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
            else:
                # Fallback: Verwende self.device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]
            
            # Modell-spezifische Limits
            model_limits = {
                "mistral": 4096,  # Mistral hat typischerweise 4096 Token Kontext
                "phi-3": 8192,    # Phi-3 hat 8192 Token Kontext
                "default": 2048   # Standard-Limit
            }
            
            # Bestimme Modell-Limit
            model_name = self.current_model_id.lower().split("-")[0] if self.current_model_id else "default"
            model_max_context = model_limits.get(model_name, model_limits["default"])
            
            # Berechne max_new_tokens korrekt (verfügbarer Platz für neue Tokens)
            max_new_tokens = min(
                max_length - input_length,  # Verfügbarer Platz basierend auf max_length
                model_max_context - input_length,  # Verfügbarer Platz basierend auf Modell-Limit
                2048  # Sicherheits-Limit (verhindert zu große Generierungen)
            )
            
            # Validierung: Prüfe ob max_new_tokens zu klein ist (BEVOR wir es auf 1 setzen)
            if max_new_tokens <= 0:
                raise ValueError(f"Input ist zu lang ({input_length} Tokens). Maximal erlaubt: {max_length}, Modell-Limit: {model_max_context}")
            
            # Stelle sicher, dass max_new_tokens mindestens 1 ist (nach Validierung)
            max_new_tokens = max(1, max_new_tokens)
            
            # Logging für Debugging
            logger.info(f"Input-Länge: {input_length}, max_length: {max_length}, max_new_tokens: {max_new_tokens}, Modell: {self.current_model_id}")
            
            # Generate mit besseren Parametern
            with torch.inference_mode():
                # Bestimme Modell-Typ für spezifische Behandlung
                is_mistral = "mistral" in self.current_model_id.lower() if self.current_model_id else False
                
                # Für Qwen: Verwende spezielle EOS-Tokens
                # Für Phi-3: Verwende auch spezielle EOS-Tokens
                eos_token_id = self.tokenizer.eos_token_id
                if hasattr(self.tokenizer, 'im_end_id'):  # Qwen-spezifisch
                    eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.im_end_id]
                elif self.current_model_id and "phi-3" in self.current_model_id.lower():
                    # Phi-3 verwendet <|endoftext|> als EOS
                    eos_token_id = self.tokenizer.eos_token_id
                elif is_mistral:
                    # Mistral: Füge zusätzliche Stop-Tokens hinzu
                    if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                        try:
                            stop_tokens = ["</s>", "<|end|>", "<|endoftext|>"]
                            stop_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in stop_tokens if self.tokenizer.convert_tokens_to_ids(t) is not None]
                            if stop_ids:
                                eos_token_id = [eos_token_id] + stop_ids
                        except:
                            pass
                
                repetition_penalty = 1.3 if is_mistral else 1.2
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else (0.7 if is_mistral else None),
                    do_sample=temperature > 0,
                    top_p=0.9 if temperature > 0 else None,
                    top_k=50 if temperature > 0 and is_mistral else None,  # Top-k für Mistral
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                    no_repeat_ngram_size=3,
                    early_stopping=True  # Stoppe früher wenn möglich
                )
            
            # Decode - nur die neuen Tokens (ohne Input)
            input_length = inputs['input_ids'].shape[1]
            output_length = outputs[0].shape[0]
            
            # Debug: Prüfe ob Output länger als Input ist
            if output_length <= input_length:
                logger.warning(f"Output ist nicht länger als Input! Input: {input_length}, Output: {output_length}")
                # Fallback: Dekodiere alles und entferne Prompt manuell
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if prompt in full_response:
                    response = full_response.replace(prompt, "").strip()
                else:
                    response = full_response
            else:
                new_tokens = outputs[0][input_length:]
                
                # Finde das erste EOS-Token und schneide dort ab
                eos_positions = []
                if isinstance(eos_token_id, list):
                    for eos_id in eos_token_id:
                        eos_pos = (new_tokens == eos_id).nonzero(as_tuple=True)[0]
                        if len(eos_pos) > 0:
                            eos_positions.append(eos_pos[0].item())
                else:
                    eos_pos = (new_tokens == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        eos_positions.append(eos_pos[0].item())
                
                # Schneide beim ersten EOS-Token ab
                if eos_positions:
                    new_tokens = new_tokens[:min(eos_positions)]
                
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                logger.debug(f"[Generate] Raw decoded response length: {len(response)} chars, first 100 chars: {response[:100]}")
                
                # 🔧 NEUE MINIMALISTISCHE BEREINIGUNG
                response = self._clean_response_minimal(response, messages, original_prompt)
                
                #logger.info(f"[Generate] Finale Response-Länge: {len(response)} chars, Modell: {self.current_model_id}")
            
            #return response
        
        except Exception as e:
            logger.error(f"Fehler bei der Generierung: {e}")
            raise
    
    def generate_stream(self, messages: List[Dict[str, str]], max_length: int = 2048, temperature: float = 0.3):
        """
        Generiert eine Antwort im Streaming-Modus (Generator)
        Verwendet TextIteratorStreamer für echtes Streaming
        
        Args:
            messages: Liste von Messages im Format [{"role": "user", "content": "..."}, ...]
            max_length: Maximale Länge der Antwort
            temperature: Kreativität (0.0 = deterministisch, 1.0 = kreativ)
            
        Yields:
            Token-Chunks als Strings
        """
        if not self.is_model_loaded():
            raise RuntimeError("Kein Modell geladen!")
        
        try:
            from transformers import TextIteratorStreamer
            import threading
            import queue
            
            # Verwende Chat-Template wenn verfügbar
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback für ältere Modelle
                prompt_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                prompt = "\n".join(prompt_parts) + "\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = inputs['input_ids'].shape[1]
            
            # Bestimme Modell-Typ
            is_mistral = "mistral" in self.current_model_id.lower() if self.current_model_id else False
            
            # EOS-Token-IDs
            eos_token_id = self.tokenizer.eos_token_id
            if hasattr(self.tokenizer, 'im_end_id'):
                eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.im_end_id]
            elif self.current_model_id and "phi-3" in self.current_model_id.lower():
                eos_token_id = self.tokenizer.eos_token_id
            elif is_mistral:
                if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                    try:
                        stop_tokens = ["</s>", "<|end|>", "<|endoftext|>"]
                        stop_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in stop_tokens if self.tokenizer.convert_tokens_to_ids(t) is not None]
                        if stop_ids:
                            eos_token_id = [eos_token_id] + stop_ids
                    except:
                        pass
            
            # Modell-spezifische Limits
            model_limits = {
                "mistral": 4096,  # Mistral hat typischerweise 4096 Token Kontext
                "phi-3": 8192,    # Phi-3 hat 8192 Token Kontext
                "default": 2048   # Standard-Limit
            }
            
            # Bestimme Modell-Limit
            model_name = self.current_model_id.lower().split("-")[0] if self.current_model_id else "default"
            model_max_context = model_limits.get(model_name, model_limits["default"])
            
            # Berechne max_new_tokens korrekt (verfügbarer Platz für neue Tokens)
            max_new_tokens = min(
                max_length - input_length,  # Verfügbarer Platz basierend auf max_length
                model_max_context - input_length,  # Verfügbarer Platz basierend auf Modell-Limit
                2048  # Sicherheits-Limit (verhindert zu große Generierungen)
            )
            
            # Validierung: Prüfe ob max_new_tokens zu klein ist (BEVOR wir es auf 1 setzen)
            if max_new_tokens <= 0:
                raise ValueError(f"Input ist zu lang ({input_length} Tokens). Maximal erlaubt: {max_length}, Modell-Limit: {model_max_context}")
            
            # Stelle sicher, dass max_new_tokens mindestens 1 ist (nach Validierung)
            max_new_tokens = max(1, max_new_tokens)
            
            # Logging für Debugging
            logger.info(f"Stream - Input-Länge: {input_length}, max_length: {max_length}, max_new_tokens: {max_new_tokens}, Modell: {self.current_model_id}")
            
            repetition_penalty = 1.3 if is_mistral else 1.2
            
            # Erstelle Streamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Generiere in separatem Thread
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature if temperature > 0 else (0.7 if is_mistral else None),
                "do_sample": temperature > 0,
                "top_p": 0.9 if temperature > 0 else None,
                "top_k": 50 if temperature > 0 and is_mistral else None,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
                "eos_token_id": eos_token_id,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "streamer": streamer
            }
            
            generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            generation_thread.start()
            
            # Yield Chunks vom Streamer
            for text in streamer:
                if text:
                    # Bereinige Chunk
                    cleaned = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
                    if cleaned:
                        yield cleaned
            
            generation_thread.join()
            
        except Exception as e:
            logger.error(f"Fehler bei Streaming-Generierung: {e}")
            raise

