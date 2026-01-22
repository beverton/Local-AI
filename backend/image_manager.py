"""
Image Manager - Verwaltet das Laden und Generieren von Bildern mit Diffusionsmodellen
"""
import json
import os
from typing import Optional, Dict, Any
import logging
from PIL import Image
import io
import base64
import torch
import threading
import queue
import time
import subprocess
import sys
import pickle
try:
    import psutil
except ImportError:
    psutil = None

# WICHTIG: Setze Hugging Face Cache-Verzeichnis BEVOR diffusers importiert wird
# Lade Config für Temp-Verzeichnis
_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
_temp_dir = "G:\\KI Modelle\\KI-Temp"  # Default
try:
    if os.path.exists(_config_path):
        with open(_config_path, 'r', encoding='utf-8') as f:
            _config = json.load(f)
            _temp_dir = _config.get("temp_directory", _temp_dir)
except Exception as e:
    pass  # Verwende Default

# Stelle sicher, dass Temp-Verzeichnis existiert
if _temp_dir and not os.path.exists(_temp_dir):
    try:
        os.makedirs(_temp_dir, exist_ok=True)
    except:
        pass  # Fallback auf System-Temp

# Setze Umgebungsvariablen für Hugging Face Cache
# Diese müssen gesetzt werden, BEVOR diffusers/huggingface_hub importiert wird
_hf_cache_dir = os.path.join(_temp_dir, "huggingface_cache")
os.environ['HF_HOME'] = _hf_cache_dir
os.environ['TRANSFORMERS_CACHE'] = os.path.join(_hf_cache_dir, "transformers")
os.environ['HF_DATASETS_CACHE'] = os.path.join(_hf_cache_dir, "datasets")
os.environ['DIFFUSERS_CACHE'] = os.path.join(_hf_cache_dir, "diffusers")
# Windows-spezifisch: TMP und TMPDIR
os.environ['TMP'] = _temp_dir
os.environ['TMPDIR'] = _temp_dir
# Python tempfile verwendet TMPDIR auf Windows
os.environ['TEMP'] = _temp_dir

# Prüfe ob CUDA verfügbar ist - nur dann xformers verwenden
USE_XFORMERS = torch.cuda.is_available()

# WICHTIG: Deaktiviere xformers für Flux, da es zu Crashes führen kann
# Flux funktioniert auch ohne xformers, nur langsamer
# xformers kann bei bestimmten CUDA-Versionen oder Modell-Kombinationen abstürzen
USE_XFORMERS = False  # Deaktiviert für Flux-Kompatibilität
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['DISABLE_XFORMERS'] = '1'
os.environ['XFORMERS_MORE_DETAILS'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import von diffusers - wird erst beim ersten Aufruf geladen
DIFFUSERS_AVAILABLE = None
DiffusionPipeline = None

def _load_diffusers():
    """Lädt diffusers lazy (erst bei Bedarf)"""
    global DIFFUSERS_AVAILABLE, DiffusionPipeline
    
    if DIFFUSERS_AVAILABLE is not None:
        return DIFFUSERS_AVAILABLE
    
    try:
        # Disable Triton (not available on Windows) - muss VOR Import gesetzt werden
        os.environ['DISABLE_TRITON'] = '1'
        os.environ['TRITON_DISABLE'] = '1'
        os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
        
        # Versuche diffusers zu importieren
        import warnings
        import sys
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            from diffusers import DiffusionPipeline as DP
            DiffusionPipeline = DP
            DIFFUSERS_AVAILABLE = True
            logger.info("Diffusers erfolgreich geladen")
            return True
    except Exception as e:
        import traceback
        error_str = str(e).lower()
        
        if 'xformers' in error_str or 'dll load failed' in error_str:
            logger.error(f"Diffusers konnte nicht geladen werden wegen xformers: {e}")
            logger.warning("Bildgenerierung wird nicht verfügbar sein")
            logger.info("LÖSUNG: Deinstallieren Sie xformers: pip uninstall xformers")
            logger.info("Flux funktioniert auch ohne xformers, nur langsamer")
        elif 'triton' in error_str:
            logger.error(f"Diffusers konnte nicht geladen werden wegen triton: {e}")
            logger.warning("Triton-Fehler wird ignoriert - diffusers sollte trotzdem funktionieren")
        else:
            logger.error(f"Diffusers konnte nicht geladen werden: {e}")
        DIFFUSERS_AVAILABLE = False
        DiffusionPipeline = None
        return False


class ImageManager:
    """Verwaltet Bildgenerierungsmodelle - lädt sie bei Bedarf und hält sie im Speicher"""
    
    # Klassenweites Lock für Modell-Laden (verhindert gleichzeitiges Laden über alle Instanzen)
    _load_lock = threading.Lock()
    _loading_instances = {}  # {thread_id: instance_id} - trackt welche Instanz gerade lädt
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.current_model_id: Optional[str] = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.performance_settings = self._load_performance_settings()
        self.gpu_allocation_budget_gb: Optional[float] = None  # GPU budget in GB for this model
        self.instance_id = id(self)  # Eindeutige ID für diese Instanz
        logger.info(f"Image Manager - Verwende Device: {self.device} (Instanz-ID: {self.instance_id})")
    
    def set_gpu_allocation_budget(self, budget_gb: Optional[float]):
        """Setzt das GPU-Allokations-Budget für dieses Modell (in GB)"""
        self.gpu_allocation_budget_gb = budget_gb
        if budget_gb is not None:
            logger.info(f"GPU-Allokations-Budget für Image-Modell gesetzt: {budget_gb:.2f}GB")
    
    def _load_performance_settings(self) -> Dict[str, Any]:
        """Lädt Performance-Einstellungen (delegiert an zentrales Modul)"""
        try:
            from settings_loader import load_performance_settings
            return load_performance_settings()
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Performance-Einstellungen: {e}")
            return {
                "gpu_optimization": "balanced",
                "disable_cpu_offload": False
            }
    
    def _load_config(self) -> Dict[str, Any]:
        """Lädt die Konfiguration aus config.json"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config-Datei nicht gefunden: {self.config_path}")
            return {}
    
    def _validate_pipeline(self) -> Dict[str, Any]:
        """
        Validiert, ob die Pipeline korrekt geladen wurde.
        
        Returns:
            Dict mit 'valid' (bool) und optional 'error' (str)
        """
        try:
            if self.pipeline is None:
                return {"valid": False, "error": "Pipeline ist None"}
            
            # Prüfe ob Pipeline die notwendigen Komponenten hat
            if not hasattr(self.pipeline, 'transformer') and not hasattr(self.pipeline, 'unet'):
                return {"valid": False, "error": "Pipeline hat weder transformer noch unet"}
            
            # Prüfe ob Pipeline auf dem richtigen Device ist (wenn CUDA verfügbar)
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    # Versuche ein Parameter zu finden
                    if hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, 'parameters'):
                        first_param = next(self.pipeline.transformer.parameters(), None)
                        if first_param is not None and first_param.device.type != 'cuda':
                            return {"valid": False, "error": f"Pipeline ist nicht auf CUDA, sondern auf {first_param.device}"}
                    elif hasattr(self.pipeline, 'unet') and hasattr(self.pipeline.unet, 'parameters'):
                        first_param = next(self.pipeline.unet.parameters(), None)
                        if first_param is not None and first_param.device.type != 'cuda':
                            return {"valid": False, "error": f"Pipeline ist nicht auf CUDA, sondern auf {first_param.device}"}
                except Exception as e:
                    logger.warning(f"Konnte Device-Validierung nicht durchführen: {e}")
            
            return {"valid": True}
        except Exception as e:
            return {"valid": False, "error": f"Validierungsfehler: {str(e)}"}
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Gibt alle verfügbaren Bildgenerierungsmodelle zurück"""
        models = {}
        for model_id, model_info in self.config.get("models", {}).items():
            if model_info.get("type") == "image":
                models[model_id] = model_info
        return models
    
    def get_current_model(self) -> Optional[str]:
        """Gibt die ID des aktuell geladenen Modells zurück"""
        return self.current_model_id
    
    def is_model_loaded(self) -> bool:
        """Prüft ob ein Modell geladen ist"""
        if not DIFFUSERS_AVAILABLE:
            return False
        if self.pipeline is None:
            return False
        
        # Prüfe ob Pipeline wirklich funktionsfähig ist
        # Prüfe ob wichtige Komponenten existieren (Flux verwendet transformer, andere verwenden unet)
        if not hasattr(self.pipeline, 'unet') and not hasattr(self.pipeline, 'transformer'):
            return False
        
        # Prüfe ob Pipeline auf erwartetem Device ist (optional, aber hilfreich)
        try:
            # Flux verwendet transformer, andere Modelle verwenden unet
            component = None
            if hasattr(self.pipeline, 'transformer') and self.pipeline.transformer is not None:
                component = self.pipeline.transformer
            elif hasattr(self.pipeline, 'unet') and self.pipeline.unet is not None:
                component = self.pipeline.unet
            
            if component is not None:
                device_check = next(component.parameters()).device
                # Wenn CUDA erwartet wird, prüfe ob Pipeline auf CUDA ist
                if self.device == "cuda" and device_check.type != "cuda":
                    logger.warning(f"Pipeline sollte auf CUDA sein, ist aber auf {device_check}")
                    # Zähle es trotzdem als geladen, da es auf CPU funktionieren kann
        except Exception as e:
            logger.debug(f"Konnte Device nicht prüfen: {e}")
            # Ignoriere Fehler bei Device-Prüfung
        
        return True
    
    def load_model(self, model_id: str) -> bool:
        """
        Lädt ein Bildgenerierungsmodell. Wenn bereits ein Modell geladen ist, wird es entladen.
        Verwendet ein Lock, um gleichzeitiges Laden über mehrere Instanzen zu verhindern.
        
        Args:
            model_id: Die ID des Modells aus der Config
            
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        # Prüfe ob bereits eine andere Instanz ein Modell lädt
        with ImageManager._load_lock:
            current_thread_id = threading.current_thread().ident
            if ImageManager._loading_instances:
                loading_instance_id = list(ImageManager._loading_instances.values())[0]
                if loading_instance_id != self.instance_id:
                    logger.warning(
                        f"Eine andere ImageManager-Instanz ({loading_instance_id}) lädt bereits ein Modell. "
                        f"Warte auf Abschluss... (Diese Instanz: {self.instance_id})"
                    )
                    # Warte nicht - gib False zurück, damit der Aufrufer entscheiden kann
                    return False
            
            # Markiere diese Instanz als lade
            ImageManager._loading_instances[current_thread_id] = self.instance_id
        
        try:
            if not _load_diffusers():
                logger.error("Diffusers ist nicht verfügbar. Bitte installieren Sie diffusers und xformers korrekt.")
                return False
            
            if model_id not in self.config.get("models", {}):
                logger.error(f"Modell nicht gefunden: {model_id}")
                return False
            
            model_info = self.config["models"][model_id]
            if model_info.get("type") != "image":
                logger.error(f"Modell {model_id} ist kein Bildgenerierungsmodell")
                return False
            
            model_path = model_info["path"]
            
            if not os.path.exists(model_path):
                logger.error(f"Modell-Pfad existiert nicht: {model_path}")
                return False
            
            # Altes Modell entladen (Speicher freigeben)
            if self.pipeline is not None:
                logger.info("Entlade aktuelles Modell...")
                del self.pipeline
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.pipeline = None
            
            logger.info(f"Lade Bildgenerierungsmodell: {model_id} von {model_path}")
            
            # CUDA-Version prüfen für Flux-Modelle
            cuda_version_info = {}
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    cuda_version = torch.version.cuda
                    cudnn_version = torch.backends.cudnn.version()
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_props = torch.cuda.get_device_properties(0)
                    compute_cap = f"{gpu_props.major}.{gpu_props.minor}"
                    
                    cuda_version_info = {
                        "cuda_version": cuda_version,
                        "cudnn_version": cudnn_version,
                        "gpu_name": gpu_name,
                        "compute_capability": compute_cap,
                        "pytorch_version": torch.__version__
                    }
                    
                    # Warnung bei möglicherweise inkompatibler CUDA-Version
                    cuda_version_parts = cuda_version.split('.')
                    if len(cuda_version_parts) >= 2:
                        cuda_major = int(cuda_version_parts[0])
                        cuda_minor = int(cuda_version_parts[1])
                        
                        # Flux benötigt normalerweise CUDA 11.8+ oder 12.4+
                        if cuda_major < 11 or (cuda_major == 11 and cuda_minor < 8):
                            logger.warning(f"WARNUNG: CUDA-Version {cuda_version} könnte zu alt für Flux sein!")
                            logger.warning("Flux empfiehlt CUDA 11.8+ oder 12.4+. Möglicherweise benötigen Sie eine separate Python-Umgebung mit neuerer CUDA-Version.")
                        elif cuda_major == 12 and cuda_minor < 4:
                            logger.warning(f"WARNUNG: CUDA-Version {cuda_version} könnte Probleme mit Flux verursachen!")
                            logger.warning("Flux funktioniert am besten mit CUDA 12.4+ oder CUDA 11.8+.")
                    
                    logger.info(f"CUDA-Info: Version {cuda_version}, GPU: {gpu_name}, Compute: {compute_cap}")
                except Exception as e:
                    logger.warning(f"Konnte CUDA-Version nicht prüfen: {e}")
            
            ## Pipeline laden
            # Flux-Modelle verwenden DiffusionPipeline
            try:
                # FluxPipeline unterstützt dtype-Parameter nicht direkt
                # Das Modell wird mit Standard-dtype geladen und dann auf GPU verschoben
                
                #### Unterdrücke triton-Warnungen während des Ladens
                import warnings
                # os ist bereits oben importiert, kein lokaler Import nötig
                # Setze Umgebungsvariable um triton-Fehler zu vermeiden
                os.environ.setdefault('TRITON_DISABLE_LINE_INFO', '1')
                
                # Optimierungen für schnelleres Laden
                # Setze CUDA-Device vor dem Laden (hilft bei GPU-Initialisierung)
                if self.device == "cuda":
                    torch.cuda.set_device(0)
                    # Leere GPU-Cache für mehr verfügbaren Speicher
                    torch.cuda.empty_cache()
                    logger.info("GPU-Cache geleert, bereit für Modell-Laden")
                    
                    ## Versuche zuerst mit Standard-Parameters
                # Flux unterstützt kein trust_remote_code und kein dtype-Parameter, also weglassen
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*triton.*")
                    warnings.filterwarnings("ignore", message=".*Keyword arguments.*dtype.*are not expected.*")
                    warnings.filterwarnings("ignore", category=UserWarning)
                    
                    # Lade-Parameter für optimale Performance
                    # FluxPipeline unterstützt dtype nicht, daher weglassen
                    # WICHTIG: Verwende local_files_only=True um Netzwerk-Downloads zu vermeiden
                    # und potenzielle Abstürze in Download-Code zu umgehen
                    load_kwargs = {
                        "local_files_only": True  # Nur lokale Dateien verwenden - verhindert Download-Abstürze
                    }
                    
                    ## Lade Pipeline ohne dtype-Parameter (FluxPipeline unterstützt ihn nicht)
                    # WICHTIG: from_pretrained kann in nativen Code abstürzen (SIGSEGV) oder hängen
                    # Wir verwenden einen Timeout-Mechanismus mit Threading, um Hänger zu erkennen
                    from_pretrained_start_time = time.time()
                    
                    ## Wrapper-Funktion für from_pretrained mit Timeout
                    def _load_pipeline_with_timeout(model_path, load_kwargs, timeout_seconds=300):
                        """Lädt Pipeline in separatem Thread mit Timeout"""
                        result_queue = queue.Queue()
                        error_queue = queue.Queue()
                        
                        def _load_worker():
                            try:
                                ## Setze Umgebungsvariablen um xformers/Triton zu deaktivieren
                                original_xformers = os.environ.get('XFORMERS_DISABLED', None)
                                original_triton = os.environ.get('TRITON_DISABLE_LINE_INFO', None)
                                os.environ['XFORMERS_DISABLED'] = '1'
                                os.environ['DISABLE_XFORMERS'] = '1'
                                os.environ['TRITON_DISABLE_LINE_INFO'] = '1'
                                
                                try:
                                    ## Versuche verschiedene Lade-Strategien
                                    # WICHTIG: device_map="cuda" funktioniert für Flux (war die Lösung)
                                    pipeline = None
                                    
                                    # Strategie 1: Mit device_map="cuda" für direktes GPU-Laden (bewährte Methode)
                                    try:
                                        logger.info("Versuche Modell direkt auf GPU zu laden (device_map='cuda')...")
                                        
                                        # Verwende SDPA (Scaled Dot Product Attention) statt Triton
                                        load_kwargs_with_attn = {
                                            **load_kwargs,
                                            'use_safetensors': True,
                                            'variant': None  # Keine spezielle Variante
                                        }
                                        
                                        pipeline = DiffusionPipeline.from_pretrained(
                                            model_path,
                                            device_map="cuda",
                                            dtype=torch.float16,  # Geändert zu float16 für bessere Kompatibilität
                                            **load_kwargs_with_attn
                                        )
                                        
                                        logger.info("Modell erfolgreich direkt auf GPU geladen (device_map='cuda')")
                                    except (TypeError, ValueError, RuntimeError) as e:
                                        ## Strategie 2: Mit low_cpu_mem_usage (lädt in CPU, dann auf GPU)
                                        logger.warning(f"device_map='cuda' nicht unterstützt ({e}), versuche mit low_cpu_mem_usage...")
                                        try:
                                            pipeline = DiffusionPipeline.from_pretrained(
                                                model_path,
                                                low_cpu_mem_usage=True,
                                                dtype=torch.float16,
                                                **load_kwargs
                                            )
                                            
                                            logger.info("Modell erfolgreich mit low_cpu_mem_usage geladen")
                                        except (TypeError, ValueError, RuntimeError) as e2:
                                            # Strategie 3: Ohne low_cpu_mem_usage
                                            logger.warning(f"low_cpu_mem_usage nicht unterstützt ({e2}), versuche ohne diese Option")
                                            
                                            pipeline = DiffusionPipeline.from_pretrained(
                                                model_path,
                                                dtype=torch.float16,
                                                **load_kwargs
                                            )
                                            logger.info("Modell erfolgreich ohne low_cpu_mem_usage geladen")
                                    
                                    result_queue.put(pipeline)
                                    
                                except Exception as e:
                                    error_queue.put(e)
                                finally:
                                    # Stelle ursprüngliche Umgebungsvariablen wieder her
                                    if original_xformers is not None:
                                        os.environ['XFORMERS_DISABLED'] = original_xformers
                                    else:
                                        os.environ.pop('XFORMERS_DISABLED', None)
                                    if original_triton is not None:
                                        os.environ['TRITON_DISABLE_LINE_INFO'] = original_triton
                                    else:
                                        os.environ.pop('TRITON_DISABLE_LINE_INFO', None)
                                        
                            except BaseException as e:
                                error_queue.put(e)
                        
                        # Starte Load-Thread
                        load_thread = threading.Thread(target=_load_worker, daemon=True, name="flux_loader")
                        load_thread.start()
                        
                        ## Warte auf Ergebnis mit Timeout und periodischen Checks
                        start_wait = time.time()
                        check_interval = 5.0  # Prüfe alle 5 Sekunden
                        last_check = start_wait
                        
                        while True:
                            elapsed = time.time() - start_wait
                            
                            # Prüfe ob Thread noch läuft
                            if not load_thread.is_alive():
                                break
                            
                            # Prüfe auf Timeout
                            if elapsed >= timeout_seconds:
                                ## Prüfe ob vielleicht doch ein Ergebnis da ist (Race Condition)
                                if not result_queue.empty():
                                    logger.info("Ergebnis nach Timeout gefunden - Thread war langsamer als erwartet")
                                    return result_queue.get()
                                if not error_queue.empty():
                                    error = error_queue.get()
                                    logger.error(f"Fehler nach Timeout gefunden: {error}")
                                    raise error
                                raise TimeoutError(f"Modell-Laden hängt nach {timeout_seconds} Sekunden. Möglicherweise ein Problem mit dem Modell oder CUDA.")
                            
                            # Periodische Heartbeat-Logs
                            if time.time() - last_check >= check_interval:
                                last_check = time.time()
                                ## Kurz warten bevor nächster Check
                            time.sleep(0.5)
                        
                        # Thread ist beendet, prüfe auf Ergebnis
                        load_thread.join()  # Warte auf vollständiges Beenden
                        
                        # Prüfe auf Fehler
                        if not error_queue.empty():
                            error = error_queue.get()
                            raise error
                        
                        # Prüfe auf Ergebnis
                        if not result_queue.empty():
                            return result_queue.get()
                        else:
                            raise RuntimeError("Load Thread beendet ohne Ergebnis oder Fehler")
                    
                    try:
                        # Versuche mit Memory-Optimierungen zu laden
                        # Flux unterstützt möglicherweise nicht alle Parameter, daher vorsichtig
                        # WICHTIG: Deaktiviere xformers komplett, da es zu Crashes führen kann
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            
                            # Verwende Timeout-Mechanismus (10 Minuten Timeout für große Modelle)
                            # Flux-Modelle können sehr groß sein und brauchen länger zum Laden
                            self.pipeline = _load_pipeline_with_timeout(model_path, load_kwargs, timeout_seconds=600)
                        
                    except TimeoutError as timeout_error:
                        logger.error(f"TIMEOUT: {timeout_error}")
                        raise
                    except Exception as from_pretrained_error:
                        raise
                    except BaseException as from_pretrained_crash:
                        # Fange auch SystemExit, KeyboardInterrupt etc.
                        raise
                
                ## Pipeline erfolgreich geladen - setze current_model_id jetzt (vor GPU-Transfer)
                # So können wir auch bei GPU-Fehlern erkennen, dass Pipeline geladen wurde
                self.current_model_id = model_id
                logger.info(f"Pipeline erfolgreich geladen, Modell-ID gesetzt: {model_id}")
                
                # Logge Pipeline-Status nach from_pretrained
                pipeline_info = {
                    "pipeline_type": type(self.pipeline).__name__ if self.pipeline else None,
                    "has_unet": hasattr(self.pipeline, 'unet') if self.pipeline else False,
                    "has_vae": hasattr(self.pipeline, 'vae') if self.pipeline else False,
                    "has_text_encoder": hasattr(self.pipeline, 'text_encoder') if self.pipeline else False,
                    "current_model_id": self.current_model_id
                }
                logger.info(f"Pipeline-Status nach from_pretrained: {pipeline_info}")
            except Exception as e1:
                #logger.warning(f"Erster Ladeversuch fehlgeschlagen: {e1}, versuche mit local_files_only=True...")
                try:
                    # Fallback: Nur lokale Dateien
                    self.pipeline = DiffusionPipeline.from_pretrained(
                        model_path,
                        dtype=torch.float32,  # Verwende float32 für CPU
                        local_files_only=True
                    )
                except Exception as e2:
                    #logger.error(f"Fehler beim Laden der Pipeline: {e2}")
                    raise
                
            # Auf GPU verschieben wenn verfügbar
            # WICHTIG: Wenn device_map verwendet wurde, ist das Modell bereits auf der GPU
            if self.device == "cuda":
                try:
                    # Prüfe ob Pipeline bereits ein device_map hat (dann ist es bereits auf GPU)
                    has_device_map = hasattr(self.pipeline, 'hf_device_map') and self.pipeline.hf_device_map is not None
                    
                    if has_device_map:
                        logger.info("Pipeline wurde bereits mit device_map auf GPU geladen, überspringe .to(device)")
                        # Prüfe ob Modell wirklich auf GPU ist
                        try:
                            # Versuche ein Parameter des Modells zu finden
                            if hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, 'parameters'):
                                first_param = next(self.pipeline.transformer.parameters(), None)
                                if first_param is not None:
                                    device_check = first_param.device
                                    logger.info(f"Pipeline ist bereits auf Device: {device_check}")
                                    if device_check.type == 'cuda':
                                        logger.info("Pipeline ist korrekt auf GPU geladen")
                                    else:
                                        logger.warning(f"Pipeline sollte auf CUDA sein, ist aber auf {device_check}")
                            elif hasattr(self.pipeline, 'unet') and hasattr(self.pipeline.unet, 'parameters'):
                                first_param = next(self.pipeline.unet.parameters(), None)
                                if first_param is not None:
                                    device_check = first_param.device
                                    logger.info(f"Pipeline ist bereits auf Device: {device_check}")
                        except Exception as e:
                            logger.debug(f"Konnte Device nicht prüfen: {e}")
                    else:
                        logger.info("Verschiebe Pipeline auf GPU...")
                        # Verwende non_blocking für asynchrones Transfer (schneller)
                        # Aber: DiffusionPipeline.to() unterstützt kein non_blocking direkt
                        # Verschiebe Komponenten einzeln für bessere Kontrolle
                        self.pipeline = self.pipeline.to(self.device)
                        
                        # Synchronisiere GPU-Operationen für korrekte Timing-Messung
                        torch.cuda.synchronize()
                        logger.info(f"Pipeline erfolgreich auf GPU verschoben: {torch.cuda.get_device_name(0)}")
                    
                    logger.info(f"GPU-Speicher verwendet: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                    logger.info(f"GPU-Speicher reserviert: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
                    logger.info(f"Pipeline-Status nach GPU-Transfer: current_model_id={self.current_model_id}")
                    
                    # Optimierungen für bessere Performance
                    # Versuche xformers zu aktivieren (nur wenn CUDA verfügbar)
                    if USE_XFORMERS:
                        try:
                            if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                                logger.info("Aktiviere xformers für bessere GPU-Performance...")
                                self.pipeline.enable_xformers_memory_efficient_attention()
                                logger.info("xformers erfolgreich aktiviert!")
                        except Exception as e:
                            logger.warning(f"xformers konnte nicht aktiviert werden: {e}")
                            logger.info("Flux funktioniert auch ohne xformers, nur langsamer")
                    else:
                        logger.info("xformers wird übersprungen (CUDA nicht verfügbar)")
                    
                except RuntimeError as gpu_error:
                    # Spezielle Behandlung für device_map-Fehler
                    if "device mapping strategy" in str(gpu_error) or "device placement" in str(gpu_error):
                        logger.info("Pipeline wurde bereits mit device_map auf GPU geladen, .to(device) nicht nötig")
                        # Modell ist bereits auf GPU, alles OK
                    else:
                        logger.warning(f"Fehler beim Verschieben auf GPU: {gpu_error}, verwende CPU")
                        self.device = "cpu"
                except Exception as gpu_error:
                    logger.warning(f"Fehler beim Verschieben auf GPU: {gpu_error}, verwende CPU")
                    self.device = "cpu"
                    
                    # GPU-Optimierungen basierend auf Performance-Einstellungen und GPU-Allokation
                    gpu_opt = self.performance_settings.get("gpu_optimization", "balanced")
                    disable_offload = self.performance_settings.get("disable_cpu_offload", False)
                    
                    # Prüfe GPU-Allokations-Budget
                    has_budget_limit = self.gpu_allocation_budget_gb is not None and self.gpu_allocation_budget_gb > 0
                    if has_budget_limit:
                        # GPU-Budget ist gesetzt - verwende speicher-effiziente Einstellungen
                        logger.info(f"GPU-Allokations-Budget aktiv ({self.gpu_allocation_budget_gb:.2f}GB) - aktiviere speicher-effiziente Optimierungen")
                        if hasattr(self.pipeline, 'enable_attention_slicing'):
                            self.pipeline.enable_attention_slicing()
                            logger.info("Attention-Slicing aktiviert (GPU-Budget)")
                        if hasattr(self.pipeline, 'enable_vae_slicing'):
                            self.pipeline.enable_vae_slicing()
                            logger.info("VAE-Slicing aktiviert (GPU-Budget)")
                        # CPU-Offloading nur wenn Budget sehr klein ist
                        if self.gpu_allocation_budget_gb < 8.0 and hasattr(self.pipeline, 'enable_model_cpu_offload') and not disable_offload:
                            self.pipeline.enable_model_cpu_offload()
                            logger.info("CPU-Offloading aktiviert (kleines GPU-Budget)")
                    else:
                        # Kein Budget-Limit - verwende normale Optimierungen
                        logger.info(f"Performance-Einstellungen: gpu_optimization={gpu_opt}, disable_cpu_offload={disable_offload}")
                        
                        if gpu_opt == "speed" and not disable_offload:
                            # Maximale Geschwindigkeit: Deaktiviere CPU-Offloading und Attention-Slicing
                            logger.info("GPU-Optimierung: Maximale Geschwindigkeit (kein CPU-Offloading, kein Attention-Slicing)")
                            # Nichts aktivieren = maximale GPU-Nutzung
                        elif gpu_opt == "memory":
                            # Speicher-effizient: Aktiviere CPU-Offloading
                            if hasattr(self.pipeline, 'enable_model_cpu_offload') and not disable_offload:
                                logger.info("Aktiviere CPU-Offloading für besseren Speicherverbrauch...")
                                self.pipeline.enable_model_cpu_offload()
                                logger.info("CPU-Offloading aktiviert")
                            elif hasattr(self.pipeline, 'enable_attention_slicing'):
                                logger.info("Aktiviere Attention-Slicing...")
                                self.pipeline.enable_attention_slicing()
                                logger.info("Attention-Slicing aktiviert")
                        else:
                            # Balanced: Standard-Optimierungen
                            if hasattr(self.pipeline, 'enable_attention_slicing') and not disable_offload:
                                logger.info("Aktiviere Attention-Slicing (balanced)...")
                                self.pipeline.enable_attention_slicing()
                                logger.info("Attention-Slicing aktiviert (balanced)")
                        
                        # Wenn CPU-Offloading deaktiviert werden soll
                        if disable_offload:
                            logger.info("CPU-Offloading ist deaktiviert - maximale GPU-Nutzung")
                    
                    # Prüfe ob Pipeline wirklich auf GPU ist
                    if hasattr(self.pipeline, 'unet'):
                        device_check = next(self.pipeline.unet.parameters()).device
                        logger.info(f"UNet ist auf Device: {device_check}")
                        logger.info(f"Pipeline-Status nach Optimierungen: current_model_id={self.current_model_id}")
                    
                except Exception as e:
                    logger.warning(f"Fehler beim Verschieben auf GPU: {e}, verwende CPU")
                    self.device = "cpu"
                    self.pipeline = self.pipeline.to(self.device)
                    # current_model_id bleibt gesetzt, da Pipeline geladen wurde
            else:
                logger.info("Verwende CPU für Bildgenerierung")
                self.pipeline = self.pipeline.to(self.device)
                # current_model_id wurde bereits nach Pipeline-Laden gesetzt
            
            # Validiere Pipeline nach vollständigem Laden
            logger.info(f"Validiere Pipeline nach vollständigem Laden: current_model_id={self.current_model_id}")
            validation_result = self._validate_pipeline()
            if not validation_result["valid"]:
                error_msg = f"Pipeline-Validierung fehlgeschlagen: {validation_result['error']}"
                logger.error(error_msg)
                logger.error(f"Pipeline-Status bei Validierungsfehler: current_model_id={self.current_model_id}, pipeline={self.pipeline is not None}")
                # Setze Status zurück bei Validierungsfehler
                self.current_model_id = None
                self.pipeline = None
                return False
            
            logger.info(f"Bildgenerierungsmodell erfolgreich geladen und validiert: {model_id}")
            logger.info(f"Finaler Pipeline-Status: current_model_id={self.current_model_id}, is_model_loaded()={self.is_model_loaded()}")
            return True
            
        except Exception as e:
            #logger.error(f"Fehler beim Laden des Bildgenerierungsmodells: {e}")
            self.pipeline = None
            return False
        finally:
            # Gib das Lock immer frei, auch bei Fehlern
            with ImageManager._load_lock:
                current_thread_id = threading.current_thread().ident
                if current_thread_id in ImageManager._loading_instances:
                    del ImageManager._loading_instances[current_thread_id]
                    logger.debug(f"Lock freigegeben für Thread {current_thread_id} (Instanz {self.instance_id})")
    
    def _calculate_dimensions_from_ratio(self, aspect_ratio: str, base_size: int = 1024) -> tuple[int, int]:
        """
        Berechnet width/height aus Aspect-Ratio
        
        Args:
            aspect_ratio: Ratio-String (z.B. "16:9", "custom:2.5:1")
            base_size: Basis-Größe für die größere Dimension
            
        Returns:
            (width, height) tuple
        """
        # Preset-Ratios
        ratios = {
            "1:1": (1, 1),
            "16:9": (16, 9),
            "9:16": (9, 16),
            "4:3": (4, 3),
            "3:4": (3, 4)
        }
        
        # Prüfe ob Preset
        if aspect_ratio in ratios:
            w_ratio, h_ratio = ratios[aspect_ratio]
        elif aspect_ratio.startswith("custom:"):
            # Custom-Ratio parsen: "custom:W:H"
            try:
                custom_part = aspect_ratio[7:]  # Entferne "custom:"
                parts = custom_part.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Ungültiges Custom-Ratio-Format: {aspect_ratio}")
                w_ratio = float(parts[0])
                h_ratio = float(parts[1])
            except (ValueError, IndexError) as e:
                logger.error(f"Fehler beim Parsen von Custom-Ratio {aspect_ratio}: {e}")
                raise ValueError(f"Ungültiges Custom-Ratio-Format: {aspect_ratio}")
        else:
            raise ValueError(f"Unbekanntes Ratio-Format: {aspect_ratio}")
        
        # Berechne Dimensionen: base_size ist die größere Dimension
        ratio_value = w_ratio / h_ratio
        if ratio_value >= 1.0:
            # Breiter als hoch
            width = base_size
            height = int(base_size / ratio_value)
        else:
            # Höher als breit
            height = base_size
            width = int(base_size * ratio_value)
        
        # Runde auf Vielfache von 8 (für Diffusion-Modelle empfohlen)
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        return (width, height)
    
    def _estimate_gpu_memory_needed(self, width: int, height: int, num_inference_steps: int) -> float:
        """
        Schätzt benötigten GPU-Speicher für Bildgenerierung
        
        Args:
            width: Bildbreite
            height: Bildhöhe
            num_inference_steps: Anzahl Inferenz-Schritte
            
        Returns:
            Geschätzter Speicherbedarf in GB
        """
        if not torch.cuda.is_available():
            return 0.0
        
        # Basis-Speicher für geladenes Modell (dynamisch je nach Modell)
        # SDXL: ~7GB, FLUX: ~30GB, SD1.5: ~4GB
        if self.current_model_id:
            model_id_lower = self.current_model_id.lower()
            if 'sdxl' in model_id_lower:
                base_model_memory = 7.0  # SDXL Base
            elif 'flux' in model_id_lower:
                base_model_memory = 30.0  # FLUX
            elif 'sd3' in model_id_lower:
                base_model_memory = 10.0  # SD3
            else:
                base_model_memory = 5.0  # SD1.5 und andere
        else:
            base_model_memory = 7.0  # Default für SDXL
        
        # Zusätzlicher Speicher für Bildgenerierung
        # Realistischere Formel: ~0.5-1GB für 1024x1024
        pixel_count = width * height
        megapixels = pixel_count / (1024 * 1024)
        # ~1GB pro Megapixel bei 20 Steps
        generation_memory = megapixels * 0.05 * num_inference_steps / 20.0
        
        # Sicherheitspuffer: 20%
        total_memory = (base_model_memory + generation_memory) * 1.2
        
        return total_memory
    
    def _check_gpu_memory_available(self) -> float:
        """
        Prüft verfügbaren GPU-Speicher
        
        Returns:
            Verfügbarer Speicher in GB
        """
        if not torch.cuda.is_available():
            return 0.0
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
        
        # Verfügbar = Total - Reserviert (reserved ist meist größer als allocated)
        available = total_memory - reserved_memory
        
        return max(0.0, available)
    
    def _auto_resize_dimensions(self, width: int, height: int, reduction_factor: float = 0.75) -> tuple[int, int]:
        """
        Reduziert Bilddimensionen um einen Faktor, behält Aspect-Ratio bei
        
        Args:
            width: Aktuelle Breite
            height: Aktuelle Höhe
            reduction_factor: Faktor für Reduktion (0.75 = 25% kleiner)
            
        Returns:
            (neue_width, neue_height)
        """
        new_width = int(width * reduction_factor)
        new_height = int(height * reduction_factor)
        
        # Runde auf Vielfache von 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Mindestgröße: 256px
        new_width = max(256, new_width)
        new_height = max(256, new_height)
        
        return (new_width, new_height)
    
    def generate_image(self, prompt: str, negative_prompt: str = "", num_inference_steps: int = 20, 
                      guidance_scale: float = 7.5, width: int = 1024, height: int = 1024, 
                      aspect_ratio: Optional[str] = None) -> Optional[Image.Image]:
        """
        Generiert ein Bild basierend auf einem Prompt
        
        Args:
            prompt: Der Text-Prompt für die Bildgenerierung
            negative_prompt: Was nicht im Bild sein soll
            num_inference_steps: Anzahl der Inferenz-Schritte (mehr = besser, aber langsamer)
            guidance_scale: Wie stark der Prompt befolgt wird
            width: Bildbreite
            height: Bildhöhe
            aspect_ratio: Optional Aspect-Ratio (überschreibt width/height wenn gesetzt)
            
        Returns:
            PIL Image oder None bei Fehler
        """
        if not _load_diffusers():
            logger.error("Diffusers ist nicht verfügbar")
            return None
        
        if not self.is_model_loaded():
            raise RuntimeError("Kein Bildgenerierungsmodell geladen!")
        
        # Berechne Dimensionen aus Ratio falls angegeben
        if aspect_ratio:
            try:
                # Lade Preset-Werte aus Config
                base_size = 1024  # Default
                try:
                    if os.path.exists(_config_path):
                        with open(_config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            image_gen_config = config.get("image_generation", {})
                            presets = image_gen_config.get("resolution_presets", {})
                            # Versuche Preset aus aspect_ratio zu extrahieren (falls vorhanden)
                            # Format könnte sein: "l:16:9" oder nur "16:9"
                            if ":" in aspect_ratio and not aspect_ratio.startswith("custom:"):
                                parts = aspect_ratio.split(":", 1)
                                if len(parts) == 2 and parts[0] in presets:
                                    preset_key = parts[0]
                                    base_size = presets[preset_key]
                                    aspect_ratio = parts[1]  # Verwende nur den Ratio-Teil
                                    logger.info(f"Preset {preset_key} erkannt, base_size: {base_size}")
                except Exception as e:
                    logger.warning(f"Konnte Preset-Werte nicht laden: {e}, verwende Default {base_size}")
                
                width, height = self._calculate_dimensions_from_ratio(aspect_ratio, base_size=base_size)
                logger.info(f"Dimensionen aus Ratio {aspect_ratio} berechnet: {width}x{height} (base_size: {base_size})")
            except ValueError as e:
                logger.error(f"Fehler bei Ratio-Berechnung: {e}")
                raise
        
        # GPU-Speicher-Management
        use_cpu_offload = False
        original_width, original_height = width, height
        
        if self.device == "cuda" and torch.cuda.is_available():
            available_memory = self._check_gpu_memory_available()
            estimated_memory = self._estimate_gpu_memory_needed(width, height, num_inference_steps)
            
            logger.info(f"GPU-Speicher: Verfügbar: {available_memory:.2f} GB, Benötigt: {estimated_memory:.2f} GB")
            
            # Auto-Resize wenn nicht genug Speicher
            max_retries = 3
            retry_count = 0
            
            while estimated_memory > available_memory and retry_count < max_retries:
                logger.warning(f"Nicht genug GPU-Speicher. Reduziere Bildgröße (Versuch {retry_count + 1}/{max_retries})")
                width, height = self._auto_resize_dimensions(width, height, reduction_factor=0.75)
                estimated_memory = self._estimate_gpu_memory_needed(width, height, num_inference_steps)
                retry_count += 1
                logger.info(f"Neue Dimensionen: {width}x{height}, Geschätzter Speicher: {estimated_memory:.2f} GB")
            
            # Wenn immer noch nicht genug Speicher, aktiviere CPU-Offload
            if estimated_memory > available_memory:
                logger.warning("GPU-Speicher reicht nicht aus. Aktiviere CPU-Offload als Fallback.")
                use_cpu_offload = True
                # Versuche CPU-Offload zu aktivieren
                try:
                    if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                        self.pipeline.enable_model_cpu_offload()
                        logger.info("CPU-Offload aktiviert")
                    else:
                        logger.warning("Pipeline unterstützt kein CPU-Offload")
                except Exception as e:
                    logger.error(f"Fehler beim Aktivieren von CPU-Offload: {e}")
        
        try:
            logger.info(f"Generiere Bild mit Prompt: {prompt[:50]}... (Größe: {width}x{height})")
            
            ## Generiere Bild mit Retry-Logik bei OOM
            max_oom_retries = 2
            oom_retry_count = 0
            
            while oom_retry_count <= max_oom_retries:
                try:
                    with torch.no_grad():
                        # Flux-spezifische Parameter
                        # Flux-Pipeline erwartet: prompt, num_inference_steps, guidance_scale, width, height
                        # und gibt ein Dictionary mit "images" zurück
                        pipeline_kwargs = {
                            "prompt": prompt,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "width": width,
                            "height": height,
                            "output_type": "pil"  # Stelle sicher, dass PIL Images zurückgegeben werden
                        }
                        
                        # Negative Prompt nur wenn angegeben
                        if negative_prompt:
                            pipeline_kwargs["negative_prompt"] = negative_prompt
                        
                        #result = self.pipeline(**pipeline_kwargs)
                    break  # Erfolgreich - verlasse Retry-Schleife
                        
                except torch.cuda.OutOfMemoryError as oom_error:
                    oom_retry_count += 1
                    if oom_retry_count > max_oom_retries:
                        logger.error(f"GPU Out of Memory nach {max_oom_retries} Versuchen. Dimensionen: {width}x{height}")
                        raise RuntimeError(f"GPU-Speicher reicht nicht aus für {width}x{height}. Bitte reduzieren Sie die Bildgröße.")
                    
                    logger.warning(f"GPU Out of Memory (Versuch {oom_retry_count}/{max_oom_retries}). Reduziere Größe und versuche erneut...")
                    torch.cuda.empty_cache()
                    
                    # Reduziere Größe weiter
                    width, height = self._auto_resize_dimensions(width, height, reduction_factor=0.75)
                    logger.info(f"Neue Dimensionen nach OOM: {width}x{height}")
                    
                    # Aktualisiere pipeline_kwargs
                    pipeline_kwargs["width"] = width
                    pipeline_kwargs["height"] = height
                    
                    # Versuche CPU-Offload wenn noch nicht aktiviert
                    if not use_cpu_offload:
                        try:
                            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                                self.pipeline.enable_model_cpu_offload()
                                use_cpu_offload = True
                                logger.info("CPU-Offload nach OOM aktiviert")
                        except Exception as e:
                            logger.warning(f"Konnte CPU-Offload nicht aktivieren: {e}")
                
                ## Extrahiere Bild (kann variieren je nach Pipeline)
            if isinstance(result, dict):
                # Flux gibt normalerweise {"images": [PIL.Image]} zurück
                images = result.get("images", [])
                if images:
                    image = images[0]
                else:
                    # Fallback: Prüfe andere mögliche Keys
                    image = result.get("image") or result.get("output")
            elif isinstance(result, (list, tuple)):
                image = result[0] if len(result) > 0 else None
            elif hasattr(result, 'images'):
                # Pipeline-Objekt mit images-Attribut
                images = result.images
                image = images[0] if isinstance(images, (list, tuple)) and len(images) > 0 else images
            else:
                # Direktes PIL Image
                image = result
            
            #if image is None:
                logger.error("Kein Bild von Pipeline erhalten")
                return None
            
            logger.info("Bild erfolgreich generiert")
            
            # Rückgabe mit zusätzlichen Informationen
            return {
                "image": image,
                "width": width,
                "height": height,
                "auto_resized": (width, height) != (original_width, original_height),
                "cpu_offload_used": use_cpu_offload
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Bildgenerierung: {e}")
            raise
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Konvertiert ein PIL Image zu Base64-String"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

