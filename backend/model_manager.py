"""
Model Manager - Verwaltet das Laden und Wechseln von AI-Modellen
"""
import json
import os
import time
from typing import Optional, Dict, Any, List

# Setze PyTorch CUDA Allocator Config für besseres Memory Management (MUSS vor torch import sein)
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import os
from logging_utils import get_logger

# Strukturierter Logger
logger = get_logger(__name__, log_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "model_manager.log"))


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
            
            # Performance-Settings nur einmal laden
            perf_settings_path = os.path.join(os.path.dirname(os.path.dirname(self.config_path)), "data", "performance_settings.json")
            perf_settings: Dict[str, Any] = {}
            if os.path.exists(perf_settings_path):
                try:
                    with open(perf_settings_path, 'r', encoding='utf-8') as f:
                        perf_settings = json.load(f)
                except Exception as e:
                    logger.warning(f"Fehler beim Laden der Performance-Settings: {e}")
            
            # Modell-spezifische Lade-Overrides (optional)
            model_loading_cfg = model_info.get("loading", {}) if isinstance(model_info, dict) else {}
            has_model_specific = bool(model_loading_cfg)
            
            # Flash Attention Einstellung (global → modell-spezifisch)
            use_flash_attention = model_loading_cfg.get(
                "use_flash_attention",
                perf_settings.get("use_flash_attention", True)
            )
            
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
            
            # Performance-Settings für Quantisierung (global → modell-spezifisch)
            use_quantization = model_loading_cfg.get(
                "use_quantization",
                perf_settings.get("use_quantization", False)
            )
            quantization_bits = model_loading_cfg.get(
                "quantization_bits",
                perf_settings.get("quantization_bits", 8)
            )
            
            # CPU-Offloading Einstellung (global → modell-spezifisch)
            disable_cpu_offload = model_loading_cfg.get(
                "disable_cpu_offload",
                perf_settings.get("disable_cpu_offload", False)
            )
            
            # Modell laden
            # FIX: Verwende float16 statt bfloat16 - bfloat16 mit device_map="auto" führt zu "meta" device state
            # FIX: Wenn disable_cpu_offload aktiv, verwende device_map="cuda" statt "auto" um CPU-Offloading zu verhindern
            max_memory = None
            
            # Erlaubt modell-spezifische Vorgaben
            custom_device_map = model_loading_cfg.get("device_map")
            
            # Torch dtype (modell-spezifisch → fallback)
            torch_dtype_str = model_loading_cfg.get("torch_dtype")
            if torch_dtype_str == "float16":
                torch_dtype = torch.float16
            elif torch_dtype_str == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Prüfe verfügbaren GPU-Speicher (wenn CUDA verfügbar)
            available_gpu_memory_gb = None
            total_memory = None
            if self.device == "cuda":
                import torch.cuda as cuda
                try:
                    # Gesamter GPU-Speicher
                    total_memory = cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    # Bereits belegter Speicher
                    allocated_memory = cuda.memory_allocated(0) / (1024**3)  # GB
                    # Reservierter Speicher (PyTorch reserviert mehr als tatsächlich genutzt)
                    reserved_memory = cuda.memory_reserved(0) / (1024**3)  # GB
                    # Verfügbarer Speicher (konservativ: nutze reserviert als Basis)
                    available_gpu_memory_gb = total_memory - reserved_memory
                    logger.info(f"GPU-Speicher: {total_memory:.1f}GB total, {reserved_memory:.1f}GB reserviert, {available_gpu_memory_gb:.1f}GB verfügbar")
                except Exception as e:
                    logger.warning(f"Fehler beim Prüfen des GPU-Speichers: {e}")
                    # Fallback: Verwende total_memory wenn verfügbar
                    if total_memory is None:
                        try:
                            total_memory = cuda.get_device_properties(0).total_memory / (1024**3)
                        except:
                            pass
            
            if custom_device_map is not None:
                device_map = custom_device_map
                # FIX: Verwende device_map="cuda" wenn genug GPU-Speicher verfügbar ist
                # max_memory wird jetzt über UI/API konfigurierbar sein
                # Wenn device_map="cuda" und max_memory gesetzt: Verwende max_memory als Limit
                # Wenn device_map="cuda" und kein max_memory: Nutze ganze GPU
                if device_map == "cuda":
                    # Prüfe ob max_memory in Performance-Settings gesetzt ist
                    # Lade Performance-Settings (wird bereits oben geladen, aber hier nochmal für max_memory)
                    perf_settings_path = os.path.join(os.path.dirname(os.path.dirname(self.config_path)), "data", "performance_settings.json")
                    performance_settings = {}
                    if os.path.exists(perf_settings_path):
                        try:
                            with open(perf_settings_path, 'r', encoding='utf-8') as f:
                                performance_settings = json.load(f)
                        except Exception as e:
                            logger.warning(f"Fehler beim Laden der Performance-Settings: {e}")
                    ui_max_memory_gb = performance_settings.get("max_memory_gb")
                    
                    if ui_max_memory_gb is not None and ui_max_memory_gb > 0:
                        # max_memory wurde im UI gesetzt - verwende es
                        max_memory = {0: f"{ui_max_memory_gb}GB"}
                        logger.model_load(f"max_memory aus UI/Performance-Settings: {ui_max_memory_gb}GB")
                    else:
                        # Kein max_memory gesetzt - nutze ganze GPU (kein max_memory Parameter)
                        max_memory = None
                        logger.model_load("device_map='cuda' - nutze ganze GPU (kein max_memory Limit)")
                    
                    # device_map="cuda" bedeutet: Modell komplett auf GPU, kein CPU-Offloading
                    disable_cpu_offload = True
            else:
                if self.device == "cuda":
                    if use_quantization:
                        # Bei Quantisierung muss device_map="auto" sein, aber wir können max_memory setzen
                        device_map = "auto"
                        if disable_cpu_offload:
                            max_memory_gb = int(available_gpu_memory_gb * 0.9) if available_gpu_memory_gb else int(total_memory * 0.9)
                            max_memory = {0: f"{max_memory_gb}GB"}
                            logger.info(f"CPU-Offloading deaktiviert - max_memory auf GPU: {max_memory_gb}GB")
                        else:
                            logger.info("Quantisierung aktiviert - device_map='auto' mit CPU-Offloading erlaubt")
                    elif disable_cpu_offload:
                        # Verhindere CPU/Disk-Offloading - prüfe ob genug Speicher für device_map="cuda"
                        if available_gpu_memory_gb is not None and available_gpu_memory_gb >= 8.0:
                            # Genug Speicher: verwende device_map="cuda"
                            device_map = "cuda"
                            logger.info(f"CPU-Offloading deaktiviert - lade Modell vollständig auf GPU (device_map='cuda', {available_gpu_memory_gb:.1f}GB verfügbar)")
                        else:
                            # Nicht genug Speicher: verwende device_map="auto" mit max_memory
                            device_map = "auto"
                            max_memory_gb = int(available_gpu_memory_gb * 0.9) if available_gpu_memory_gb else int(total_memory * 0.9)
                            max_memory = {0: f"{max_memory_gb}GB"}
                            logger.info(
                                f"CPU-Offloading deaktiviert, aber nur {available_gpu_memory_gb:.1f}GB verfügbar. "
                                f"Verwende device_map='auto' mit max_memory={max_memory_gb}GB um OOM zu vermeiden."
                            )
                    else:
                        # Erlaube CPU-Offloading wenn nötig
                        device_map = "auto"
                        logger.info("CPU-Offloading erlaubt - device_map='auto'")
                else:
                    device_map = None
            
            # Zusammenfassung der Lade-Parameter loggen
            logger.info(
                f"Model-Loading Config -> model_id={model_id}, device_map={device_map}, "
                f"disable_cpu_offload={disable_cpu_offload}, use_quantization={use_quantization}, "
                f"quantization_bits={quantization_bits}, model_specific={has_model_specific}"
            )
            
            # OPTIMIERUNG: Bei device_map="cuda" sollte max_memory NICHT gesetzt werden (nutzt ganze GPU)
            # max_memory wird nur bei device_map="auto" benötigt
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": device_map,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,  # Reduziert CPU-Speicher während des Ladens
            }
            
            # Füge max_memory nur hinzu wenn gesetzt UND device_map="auto" (nicht bei "cuda")
            if max_memory is not None and device_map == "auto":
                model_kwargs["max_memory"] = max_memory
                # Setze Umgebungsvariable um caching_allocator_warmup zu deaktivieren
                os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            
            # Quantisierung (8-bit/4-bit) mit bitsandbytes
            if use_quantization and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    
                    if quantization_bits == 8:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16  # FIX: float16 statt bfloat16
                        )
                        logger.info("8-bit Quantisierung wird verwendet")
                    elif quantization_bits == 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,  # FIX: float16 statt bfloat16
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        logger.info("4-bit Quantisierung wird verwendet")
                    else:
                        logger.warning(f"Unbekannte Quantisierungs-Bits: {quantization_bits}, verwende 8-bit")
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16  # FIX: float16 statt bfloat16
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
            
            logger.model_load(f"Lade Modell mit device_map={device_map}, torch_dtype={torch_dtype}, max_memory={max_memory}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            logger.model_load("Modell-Objekt erstellt, prüfe Device-Platzierung...")
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            elif self.device == "cuda" and hasattr(self.model, 'hf_device_map'):
                # OPTIMIERUNG: Nur bei device_map="auto" validieren (bei "cuda" ist Validierung nicht nötig)
                # Bei device_map="cuda" wird das Modell direkt auf GPU geladen, daher keine "meta" device Probleme
                if device_map == "auto":
                    # OPTIMIERUNG: Stichproben-Validierung statt alle Module prüfen (viel schneller)
                    logger.info("Validiere dass Modell auf echten GPU-Devices geladen wurde (Stichprobe)...")
                    meta_modules = []
                    # Prüfe nur eine Stichprobe von Modulen (erste 10 + letzte 10) statt alle
                    all_modules = list(self.model.named_modules())
                    sample_modules = all_modules[:10] + all_modules[-10:] if len(all_modules) > 20 else all_modules
                    
                    for name, module in sample_modules:
                        try:
                            first_param = next(module.parameters(), None)
                            if first_param is not None and str(first_param.device) == "meta":
                                meta_modules.append(name)
                        except StopIteration:
                            continue
                    
                    if meta_modules:
                        error_msg = f"Modell wurde nicht korrekt geladen - folgende Module sind auf 'meta' device: {meta_modules[:5]}"
                        logger.error_log(f"{error_msg} (Gesamt {len(meta_modules)} Module auf 'meta' device in Stichprobe)")
                        raise RuntimeError(error_msg)
                    
                    logger.model_load("✓ Stichproben-Validierung erfolgreich (kein 'meta' device gefunden)")
                else:
                    # Bei device_map="cuda" ist Validierung nicht nötig - Modell ist direkt auf GPU
                    logger.model_load("✓ Modell direkt auf GPU geladen (device_map='cuda', keine Validierung nötig)")
            
            # Prüfe Performance-Settings für torch.compile() (wiederverwende bereits geladene perf_settings)
            use_torch_compile = perf_settings.get("use_torch_compile", False)
            
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
            logger.model_load(f"✓ Modell erfolgreich geladen: {model_id}")
            logger.model_load(f"Modell-Info: device={self.device}, dtype={type(self.model.dtype) if hasattr(self.model, 'dtype') else 'N/A'}")
            return True
            
        except Exception as e:
            logger.exception(f"Fehler beim Laden des Modells {model_id}: {str(e)}", tag="MODEL_LOAD")
            self.model = None
            self.tokenizer = None
            self.current_model_id = None
            return False
    
    def generate(self, messages: List[Dict[str, str]], max_length: int = 2048, temperature: float = 0.3, max_retries: int = 2, is_coding: bool = False) -> str:
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
                response = self._generate_internal(messages, current_max_length, temperature, is_coding=is_coding)
                
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
            # FIX: Akzeptiere kurze Wörter wenn sie mit Punktation enden (z.B. "4.")
            has_punctuation = last_word.endswith(('.', '!', '?', ':', ';'))
            if len(last_word) < 3 and not has_punctuation:  # Sehr kurzes letztes Wort OHNE Punktation = wahrscheinlich abgeschnitten
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
        # AKTIVIERT: Trimmt Response bei "Human:", "Assistant:", "System:" Markern
        import re
        # Trimme bei Markern auch mid-line
        markers = ['Human:', 'Assistant:', 'System:', 'User:']
        for marker in markers:
            if marker in response:
                # Finde Position des Markers
                pos = response.find(marker)
                # Wenn der Marker nicht am Anfang ist (dann ist es kein Fehler, sondern Content)
                if pos > 10:  # Mindestens 10 Zeichen Content vorher
                    response = response[:pos].strip()
                    logger.debug(f"[Clean] Trimmed at marker '{marker}'")
                    break
        
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
    
    def _generate_internal(self, messages: List[Dict[str, str]], max_length: int = 2048, temperature: float = 0.3, is_coding: bool = False) -> str:
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
                target_device = self.model.device
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            elif hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                # Modell verwendet device_map="auto" - Inputs auf erstes Device
                first_device = list(self.model.hf_device_map.values())[0]
                target_device = first_device
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            else:
                # Fallback: Verwende self.device
                target_device = self.device
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            
            input_length = inputs['input_ids'].shape[1]
            
            # Modell-spezifische Limits
            model_limits = {
                "mistral": 4096,      # Mistral hat typischerweise 4096 Token Kontext
                "phi-3": 8192,        # Phi-3 hat 8192 Token Kontext
                "qwen": 32768,        # Qwen-2.5-7B hat 32k Token Kontext
                "qwen2": 32768,       # Qwen-2.x hat auch 32k
                "default": 2048       # Standard-Limit
            }
            
            # Bestimme Modell-Limit
            # Prüfe zuerst auf "qwen" im gesamten Modell-Namen (nicht nur Prefix)
            if self.current_model_id and "qwen" in self.current_model_id.lower():
                model_max_context = model_limits.get("qwen", model_limits["default"])
            else:
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
            # Leere GPU Cache vor Generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.inference_mode():
                # Bestimme Modell-Typ für spezifische Behandlung
                is_mistral = "mistral" in self.current_model_id.lower() if self.current_model_id else False
                
                # Bestimme ob Qwen-Modell (durch Modell-Namen, nicht Tokenizer-Attribut)
                is_qwen = self.current_model_id and "qwen" in self.current_model_id.lower()
                
                # Für Qwen: Verwende spezielle EOS-Tokens
                # Für Phi-3: Verwende auch spezielle EOS-Tokens
                eos_token_id = self.tokenizer.eos_token_id
                if is_qwen:
                    # Qwen: Versuche im_end_id zu finden, sonst verwende nur eos_token_id
                    # Qwen-2.5 verwendet typischerweise beide Tokens: eos_token_id und im_end_id
                    try:
                        # Versuche im_end_id über Tokenizer zu finden
                        if hasattr(self.tokenizer, 'im_end_id'):
                            im_end_id = self.tokenizer.im_end_id
                            # WICHTIG: Prüfe ob im_end_id != eos_token_id (verhindert Duplikate)
                            if im_end_id != self.tokenizer.eos_token_id:
                                eos_token_id = [self.tokenizer.eos_token_id, im_end_id]
                                logger.debug(f"[Qwen] Verwende EOS-Token-Liste: {eos_token_id}")
                            else:
                                # im_end_id ist identisch mit eos_token_id, verwende nur eos_token_id
                                eos_token_id = [self.tokenizer.eos_token_id]
                                logger.debug(f"[Qwen] im_end_id ist identisch mit eos_token_id ({im_end_id}), verwende nur eos_token_id: {eos_token_id}")
                        else:
                            # Fallback: Versuche über convert_tokens_to_ids
                            im_end_token = "<|im_end|>"
                            if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                                im_end_id = self.tokenizer.convert_tokens_to_ids(im_end_token)
                                if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
                                    # WICHTIG: Prüfe ob im_end_id != eos_token_id (verhindert Duplikate)
                                    if im_end_id != self.tokenizer.eos_token_id:
                                        eos_token_id = [self.tokenizer.eos_token_id, im_end_id]
                                        logger.debug(f"[Qwen] Verwende EOS-Token-Liste (via convert_tokens_to_ids): {eos_token_id}")
                                    else:
                                        # im_end_id ist identisch mit eos_token_id, verwende nur eos_token_id
                                        eos_token_id = [self.tokenizer.eos_token_id]
                                        logger.debug(f"[Qwen] im_end_id ist identisch mit eos_token_id ({im_end_id}), verwende nur eos_token_id: {eos_token_id}")
                                else:
                                    # Nur eos_token_id verwenden
                                    eos_token_id = [self.tokenizer.eos_token_id]
                                    logger.model_gen(f"[Qwen] Verwende nur eos_token_id (im_end_id nicht gefunden): {eos_token_id}", level="debug")
                            else:
                                eos_token_id = [self.tokenizer.eos_token_id]
                                logger.model_gen(f"[Qwen] Verwende nur eos_token_id: {eos_token_id}", level="debug")
                    except Exception as e:
                        logger.warning(f"[Qwen] Fehler beim Bestimmen von EOS-Tokens: {e}, verwende nur eos_token_id", tag="MODEL_GEN")
                        eos_token_id = [self.tokenizer.eos_token_id]
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
                
                repetition_penalty = 1.1 if is_coding else (1.3 if is_mistral else 1.2)
                
                # Für Qwen: Liste beibehalten (model.generate() unterstützt Listen)
                # Für andere Modelle: Single Integer verwenden
                if is_qwen:
                    # Qwen: Verwende Liste mit beiden EOS-Tokens
                    # SICHERHEIT: Prüfe ob Liste nicht leer ist
                    if isinstance(eos_token_id, list) and len(eos_token_id) > 0:
                        eos_token_id_for_generate = eos_token_id  # Bleibt Liste
                    else:
                        # Fallback: Verwende nur eos_token_id als Liste
                        eos_token_id_for_generate = [self.tokenizer.eos_token_id]
                        logger.warning(f"[Qwen] EOS-Token-Liste war leer oder ungültig, verwende Fallback: {eos_token_id_for_generate}")
                    logger.debug(f"[Qwen] Verwende EOS-Token-Liste für generate(): {eos_token_id_for_generate}")
                else:
                    # Andere Modelle: Single Integer
                    eos_token_id_for_generate = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
                
                logger.warning(f"[DEBUG] BEFORE model.generate() - eos_token_id={eos_token_id_for_generate}, max_new_tokens={max_new_tokens}, temperature={temperature}")
                
                # DEBUG: GPU-Speicher-Status vor Generierung
                if torch.cuda.is_available():
                    gpu_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_memory_reserved_before = torch.cuda.memory_reserved() / 1024**3  # GB
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    gpu_memory_usage_percent = (gpu_memory_reserved_before / gpu_memory_total) * 100
                    
                    logger.debug(f"[DEBUG] GPU-Speicher VOR generate(): {gpu_memory_before:.2f}GB allocated, {gpu_memory_reserved_before:.2f}GB reserved ({gpu_memory_usage_percent:.1f}% von {gpu_memory_total:.2f}GB)")
                    
                    # Warnung wenn GPU-Speicher sehr hoch ist (möglicherweise durch anderes Programm blockiert)
                    if gpu_memory_usage_percent > 90:
                        logger.warning(f"[WARNUNG] GPU-Speicher sehr hoch ({gpu_memory_usage_percent:.1f}%)! Möglicherweise blockiert ein anderes Programm (z.B. ein Spiel) die GPU. Generierung könnte langsam sein oder hängen bleiben.")
                
                # DEBUG: Timing für Generierung
                import time
                import threading
                generate_start_time = time.time()
                logger.debug(f"[DEBUG] Starte model.generate() um {time.strftime('%H:%M:%S')}")
                
                # Heartbeat-Mechanismus: Logge alle 10 Sekunden, dass Generierung noch läuft
                heartbeat_stop = threading.Event()
                heartbeat_thread = None
                
                def heartbeat_logger():
                    """Loggt alle 10 Sekunden, dass Generierung noch läuft"""
                    elapsed = 0
                    while not heartbeat_stop.is_set():
                        time.sleep(10)  # Alle 10 Sekunden
                        if not heartbeat_stop.is_set():
                            elapsed = time.time() - generate_start_time
                            if torch.cuda.is_available():
                                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                                logger.warning(f"[HEARTBEAT] Generierung läuft noch... ({elapsed:.1f}s, GPU: {gpu_mem:.2f}GB)")
                            else:
                                logger.warning(f"[HEARTBEAT] Generierung läuft noch... ({elapsed:.1f}s)")
                
                heartbeat_thread = threading.Thread(target=heartbeat_logger, daemon=True)
                heartbeat_thread.start()
                
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else (0.7 if is_mistral else None),
                        do_sample=temperature > 0,
                        top_p=0.9 if temperature > 0 else None,
                        top_k=50 if temperature > 0 and is_mistral else None,  # Top-k für Mistral
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
                        eos_token_id=eos_token_id_for_generate,  # Kann jetzt Liste oder Integer sein
                        no_repeat_ngram_size=3
                        # early_stopping entfernt - invalid für sampling mode
                    )
                    
                    # Stoppe Heartbeat
                    heartbeat_stop.set()
                    if heartbeat_thread:
                        heartbeat_thread.join(timeout=1)
                    
                    generate_duration = time.time() - generate_start_time
                    logger.debug(f"[DEBUG] model.generate() abgeschlossen nach {generate_duration:.2f} Sekunden")
                    
                except Exception as e:
                    # Stoppe Heartbeat
                    heartbeat_stop.set()
                    if heartbeat_thread:
                        heartbeat_thread.join(timeout=1)
                    
                    generate_duration = time.time() - generate_start_time
                    logger.error(f"[DEBUG] model.generate() FEHLER nach {generate_duration:.2f} Sekunden: {e}")
                    raise
                
                # DEBUG: GPU-Speicher-Status nach Generierung
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_memory_reserved_after = torch.cuda.memory_reserved() / 1024**3  # GB
                    logger.debug(f"[DEBUG] GPU-Speicher NACH generate(): {gpu_memory_after:.2f}GB allocated, {gpu_memory_reserved_after:.2f}GB reserved")
                    logger.debug(f"[DEBUG] GPU-Speicher-Änderung: {gpu_memory_after - gpu_memory_before:.2f}GB allocated, {gpu_memory_reserved_after - gpu_memory_reserved_before:.2f}GB reserved")
                
                logger.model_gen(f"AFTER generate(): outputs.shape={outputs.shape if hasattr(outputs, 'shape') else 'unknown'}", level="debug")
                logger.debug(f"[DEBUG] Generierung abgeschlossen, starte Decoding...")
            
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
                
                decode_start_time = time.time()
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                decode_duration = time.time() - decode_start_time
                logger.debug(f"[Generate] Raw decoded response length: {len(response)} chars, first 100 chars: {response[:100]}")
                logger.debug(f"[DEBUG] Decoding dauerte {decode_duration:.2f} Sekunden")
                logger.model_gen(f"Decoding abgeschlossen, Länge: {len(response)} Zeichen", level="debug")
                
                # 🔧 NEUE MINIMALISTISCHE BEREINIGUNG
                clean_start_time = time.time()
                logger.model_gen(f"Vor Cleaning: {len(response)} Zeichen", level="debug")
                response = self._clean_response_minimal(response, messages, original_prompt)
                clean_duration = time.time() - clean_start_time
                logger.debug(f"[DEBUG] Cleaning dauerte {clean_duration:.2f} Sekunden")
                logger.model_gen(f"Nach Cleaning: {len(response)} Zeichen, Response: {response[:100]}...", level="debug")
            
                total_duration = time.time() - generate_start_time
                logger.model_gen(f"Response fertig, finale Länge: {len(response)} Zeichen, Gesamt-Dauer: {total_duration:.2f}s (generate: {generate_duration:.2f}s, decode: {decode_duration:.2f}s, clean: {clean_duration:.2f}s)", level="debug")
            return response
        
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
            
            # Für Qwen: Liste beibehalten (model.generate() unterstützt Listen)
            # Für andere Modelle: Single Integer verwenden
            if is_qwen:
                # Qwen: Verwende Liste mit beiden EOS-Tokens
                eos_token_id_for_stream = eos_token_id  # Bleibt Liste
                logger.debug(f"[Qwen] Streaming: Verwende EOS-Token-Liste: {eos_token_id_for_stream}")
            else:
                # Andere Modelle: Single Integer
                eos_token_id_for_stream = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
            
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
                "eos_token_id": eos_token_id_for_stream,
                "no_repeat_ngram_size": 3,
                # early_stopping entfernt - invalid für sampling mode
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

