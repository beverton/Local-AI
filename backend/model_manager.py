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
        # Fail-fast: prüfe Laufzeit-Abhängigkeiten früh und zentral
        self._check_requirements()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_allocation_budget_gb: Optional[float] = None  # GPU budget in GB for this model
        logger.info(f"Verwende Device: {self.device}")
        
        # Lade Performance-Einstellungen
        self._apply_performance_settings()
        
        # Wende GPU-Optimierungen an
        self._apply_gpu_optimizations()

    @staticmethod
    def _version_geq(current: str, minimum: str) -> bool:
        """
        Robuster Versionsvergleich (ohne harte Abhängigkeit auf packaging).
        Unterstützt übliche Semver-Strings wie '4.40.0', '2.3.1+cu121'.
        """
        try:
            from packaging import version as _pkg_version  # type: ignore
            return _pkg_version.parse(current) >= _pkg_version.parse(minimum)
        except Exception:
            def _normalize(v: str) -> List[int]:
                # extrahiere führende Zahlenblöcke: "2.3.1+cu121" -> [2,3,1]
                parts: List[int] = []
                for token in str(v).replace("+", ".").replace("-", ".").split("."):
                    num = ""
                    for ch in token:
                        if ch.isdigit():
                            num += ch
                        else:
                            break
                    if num != "":
                        parts.append(int(num))
                return parts

            cur = _normalize(current)
            minv = _normalize(minimum)
            # pad to equal length
            n = max(len(cur), len(minv))
            cur += [0] * (n - len(cur))
            minv += [0] * (n - len(minv))
            return cur >= minv

    def _check_requirements(self) -> None:
        """Prüft Mindestversionen für Kern-Abhängigkeiten und gibt klare Fehler aus."""
        try:
            import transformers as _transformers  # type: ignore
        except Exception as e:
            raise RuntimeError(f"transformers konnte nicht importiert werden: {e}") from e

        # torch ist bereits importiert; Import hier nur zur Symmetrie/Fehlermeldung
        try:
            import torch as _torch  # type: ignore
        except Exception as e:
            raise RuntimeError(f"torch konnte nicht importiert werden: {e}") from e

        transformers_min = "4.37.0"
        torch_recommended = "2.3.0"

        tver = getattr(_transformers, "__version__", "0.0.0")
        pver = getattr(_torch, "__version__", "0.0.0")

        if not self._version_geq(str(tver), transformers_min):
            raise RuntimeError(
                f"transformers>={transformers_min} erforderlich (Qwen Chat-Template/Tools). "
                f"Gefunden: {tver}"
            )

        if not self._version_geq(str(pver), torch_recommended):
            logger.warning(f"torch>={torch_recommended} empfohlen, gefunden: {pver}")

    def _build_stopping_criteria(self):
        """
        StoppingCriteria um unnötiges Weitergenerieren zu verhindern.
        Hauptfall: Modell beginnt nach fertiger Antwort neue Rollen wie 'User:' / 'Human:' zu erzeugen.
        """
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except Exception:
            return None

        stop_strings = ["\nUser:", "\nHuman:", "\nAssistant:", "User:", "Human:"]
        stop_seqs: List[List[int]] = []
        for s in stop_strings:
            try:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_seqs.append(ids)
            except Exception:
                continue

        if not stop_seqs:
            return None

        class _StopOnTokenSequences(StoppingCriteria):
            def __init__(self, sequences: List[List[int]]):
                super().__init__()
                self.sequences = sequences

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                try:
                    seq = input_ids[0].tolist()
                    for pat in self.sequences:
                        if len(seq) >= len(pat) and seq[-len(pat):] == pat:
                            return True
                except Exception:
                    return False
                return False

        return StoppingCriteriaList([_StopOnTokenSequences(stop_seqs)])

    def _auto_cap_max_new_tokens(self, messages: List[Dict[str, str]], max_new_tokens: int, is_coding: bool = False) -> int:
        """
        Reduziert max_new_tokens bei klar kurzen Anfragen (schneller, weniger Runaway).
        Greift nur, wenn die User-Nachricht explizit kurz/limitiert ist.
        """
        try:
            import re

            if max_new_tokens <= 1 or is_coding:
                return max_new_tokens

            user_text = ""
            for m in reversed(messages or []):
                if m.get("role") == "user":
                    user_text = str(m.get("content") or "")
                    break
            if not user_text:
                return max_new_tokens

            t = user_text.lower()

            # Sehr kurze Formate
            if re.search(r"\bnur\s+(?:die\s+)?zahl(?:en)?\b", t) or re.search(r"\bantwort\s+nur\s+mit\s+(?:einer\s+)?zahl\b", t):
                capped = min(max_new_tokens, 32)
                if capped < max_new_tokens:
                    logger.info(f"[AutoCap] Kürze max_new_tokens {max_new_tokens} -> {capped} (nur Zahl)")
                return capped

            # "in X Sätzen"
            m = re.search(r"\bin\s+(\d{1,2})\s+(?:s[äa]tz|sae?tz)", t)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 8:
                    # grobe Heuristik: ~55 Tokens pro Satz (Deutsch, kurz & stabil)
                    capped = min(max_new_tokens, max(64, 55 * n))
                    if capped < max_new_tokens:
                        logger.info(f"[AutoCap] Kürze max_new_tokens {max_new_tokens} -> {capped} (in {n} Sätzen)")
                    return capped

            # Bulletpoints / Liste mit fixer Anzahl ("in 6 bulletpoints", "6 stichpunkte", "nenne 5 ...")
            m = re.search(r"\b(?:in\s+)?(\d{1,2})\s+(?:bulletpoints?|stichpunkte?|punkte)\b", t)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 20:
                    # ~28 Tokens pro Punkt (damit wir <30s bleiben auf langsamer HW)
                    capped = min(max_new_tokens, max(80, 28 * n))
                    if capped < max_new_tokens:
                        logger.info(f"[AutoCap] Kürze max_new_tokens {max_new_tokens} -> {capped} ({n} Punkte)")
                    return capped

            m = re.search(r"\bnenn(?:e|en)\s+(\d{1,2})\b", t)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 20:
                    capped = min(max_new_tokens, max(80, 28 * n))
                    if capped < max_new_tokens:
                        logger.info(f"[AutoCap] Kürze max_new_tokens {max_new_tokens} -> {capped} (nenne {n})")
                    return capped

            # Anleitung/Schritt-für-Schritt: bewusst kurz halten (verhindert 30s max_time runs)
            if "schritt" in t or "schritt-fuer-schritt" in t or "schritt für schritt" in t or "anleitung" in t:
                capped = min(max_new_tokens, 180)
                if capped < max_new_tokens:
                    logger.info(f"[AutoCap] Kürze max_new_tokens {max_new_tokens} -> {capped} (Anleitung)")
                return capped

            # "kurz" / "kurze Antwort"
            if "kurz" in t or "kurze antwort" in t or "kurz erklären" in t or "kurz erklaere" in t:
                capped = min(max_new_tokens, 192)
                if capped < max_new_tokens:
                    logger.info(f"[AutoCap] Kürze max_new_tokens {max_new_tokens} -> {capped} (kurz)")
                return capped

        except Exception:
            return max_new_tokens

        return max_new_tokens

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_length: int = 2048,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Generiert entweder eine normale Antwort oder Tool-Calls (Function Calling).

        Rückgabeformat:
        - {"content": "<assistant_text>", "tool_calls": [...], "raw": "<raw_decoded>"}

        Hinweis: Tool-Ausführung passiert außerhalb (z.B. im ChatAgent).
        """
        if not self.is_model_loaded():
            raise RuntimeError("Kein Modell geladen!")

        # 1) Prompt bauen (mit Tools, falls Template das unterstützt/auswertet)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                # Fallback: Template akzeptiert tools kwarg nicht (oder ignoriert es)
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            # Fallback für ältere Modelle
            prompt_parts: List[str] = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            prompt = "\n".join(prompt_parts) + "\nAssistant:"

        # 2) Tokenize + Device placement (device_map="auto" kompatibel, wie in _generate_internal)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            target_device = self.model.device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        elif hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            first_device = list(self.model.hf_device_map.values())[0]
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 3) EOS handling (konservativ, kompatibel mit Qwen/Mistral/Phi)
        eos_token_id_for_generate: Any = self.tokenizer.eos_token_id
        try:
            if hasattr(self.tokenizer, "im_end_id"):
                # Qwen ChatML
                eos_token_id_for_generate = [self.tokenizer.eos_token_id, self.tokenizer.im_end_id]
        except Exception:
            pass

        max_new_tokens = max(1, int(max_length))
        do_sample = bool(temperature and float(temperature) > 0.0)

        # 4) Generate
        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=1.15,
            eos_token_id=eos_token_id_for_generate,
        )
        if do_sample:
            gen_kwargs.update(
                temperature=float(temperature),
                top_p=0.9,
            )
        out = self.model.generate(**gen_kwargs)

        # 5) Decode nur neu generierte Tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = out[0][input_len:]

        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        tool_calls = self._parse_tool_calls_from_text(raw)
        if not tool_calls:
            raw_alt = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            tool_calls = self._parse_tool_calls_from_text(raw_alt)
            raw = raw_alt if tool_calls else raw

        content = raw.strip()
        return {"content": content, "tool_calls": tool_calls, "raw": raw}

    @staticmethod
    def _parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
        """
        Best-effort Parser für Tool-Calls aus Model-Text.
        Unterstützt u.a. Hermes-Style:
          <<tool_call>>{...}<< /tool_call >>
        sowie freie JSON-Objekte mit {name, arguments}.
        """
        import json as _json
        import re as _re

        if not text:
            return []

        def _extract_from_obj(obj: Any) -> List[Dict[str, Any]]:
            out_calls: List[Dict[str, Any]] = []
            if isinstance(obj, dict):
                if "name" in obj and ("arguments" in obj or "parameters" in obj):
                    args = obj.get("arguments", obj.get("parameters", {}))
                    if isinstance(args, str):
                        try:
                            args = _json.loads(args)
                        except Exception:
                            args = {"_raw": args}
                    out_calls.append({"name": obj["name"], "arguments": args})
                elif "function" in obj and isinstance(obj["function"], dict):
                    fn = obj["function"]
                    if "name" in fn and ("arguments" in fn or "parameters" in fn):
                        args = fn.get("arguments", fn.get("parameters", {}))
                        if isinstance(args, str):
                            try:
                                args = _json.loads(args)
                            except Exception:
                                args = {"_raw": args}
                        out_calls.append({"name": fn["name"], "arguments": args})
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "name" in item:
                        args = item.get("arguments", item.get("parameters", {}))
                        if isinstance(args, str):
                            try:
                                args = _json.loads(args)
                            except Exception:
                                args = {"_raw": args}
                        out_calls.append({"name": item["name"], "arguments": args})
            return out_calls

        stripped = text.strip()
        # Fast-path: kompletter Text ist JSON (häufig bei Tool-Calls)
        if stripped and stripped[0] in "{[" and stripped[-1] in "}]":
            try:
                obj0 = _json.loads(stripped)
                direct_calls = _extract_from_obj(obj0)
                if direct_calls:
                    return direct_calls
            except Exception:
                pass

        candidates: List[str] = []

        # Hermes-style tags (varianten)
        tag_patterns = [
            ("<<tool_call>>", "<</tool_call>>"),
            ("<tool_call>", "</tool_call>"),
            ("<<tool_call>>", "<<|im_end|>>"),
        ]
        for start_tag, end_tag in tag_patterns:
            try:
                pattern = _re.escape(start_tag) + r"\s*([\s\S]*?)\s*" + _re.escape(end_tag)
                for m in _re.finditer(pattern, text):
                    candidates.append(m.group(1).strip())
            except Exception:
                continue

        # Falls keine Tags: versuche JSON-Objekte direkt im Text zu finden
        if not candidates:
            # naive Extraktion von {...} Blöcken
            for m in _re.finditer(r"\{[\s\S]*?\}", text):
                blob = m.group(0).strip()
                # Quick filter: Tool-call muss mindestens name enthalten
                if '"name"' in blob or "'name'" in blob:
                    candidates.append(blob)

        tool_calls: List[Dict[str, Any]] = []
        for cand in candidates:
            try:
                obj = _json.loads(cand)
            except Exception:
                continue

            tool_calls.extend(_extract_from_obj(obj))

        return tool_calls
    
    def set_gpu_allocation_budget(self, budget_gb: Optional[float]):
        """Setzt das GPU-Allokations-Budget für dieses Modell (in GB)"""
        self.gpu_allocation_budget_gb = budget_gb
        if budget_gb is not None:
            logger.info(f"GPU-Allokations-Budget für Text-Modell gesetzt: {budget_gb:.2f}GB")
    
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

    def unload_model(self) -> bool:
        """Entlädt das aktuelle Modell und gibt GPU-Speicher frei"""
        try:
            if self.model is not None:
                logger.info("Entlade aktuelles Modell...")
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.model = None
                self.tokenizer = None
                self.current_model_id = None
            return True
        except Exception as e:
            logger.warning(f"Fehler beim Entladen des Modells: {e}")
            return False
    
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
            
            # Performance-Settings nur einmal laden (nutze zentrales Modul mit Caching)
            try:
                from settings_loader import load_performance_settings
                perf_settings = load_performance_settings()
                logger.debug(f"[SETTINGS] Performance-Settings geladen: use_quantization={perf_settings.get('use_quantization')}")
            except Exception as e:
                logger.warning(f"Fehler beim Laden der Performance-Settings: {e}, verwende leeres Dict")
                perf_settings = {}
            
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
            
            # Qwen: EOS-Token an <|im_end|> ausrichten (verhindert Chat-Marker-Leakage)
            if "qwen" in model_id.lower():
                try:
                    im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                    if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
                        self.tokenizer.eos_token = "<|im_end|>"
                except Exception:
                    pass
            
            # Performance-Settings für Quantisierung (global → modell-spezifisch)
            # WICHTIG: Wenn nicht in model_loading_cfg, verwende perf_settings
            use_quantization = model_loading_cfg.get("use_quantization") if "use_quantization" in model_loading_cfg else perf_settings.get("use_quantization", False)
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
                # WICHTIG: GPU-Allokations-Budget hat Priorität über custom_device_map
                # Prüfe GPU-Budget IMMER zuerst, unabhängig von custom_device_map
                if self.gpu_allocation_budget_gb is not None and self.gpu_allocation_budget_gb > 0:
                    # GPU-Allokations-Budget ist gesetzt - verwende es
                    max_memory = {0: f"{self.gpu_allocation_budget_gb:.2f}GB"}
                    logger.model_load(f"max_memory aus GPU-Allokation: {self.gpu_allocation_budget_gb:.2f}GB (überschreibt custom_device_map={custom_device_map})")
                    # Bei Budget-Limit sollte device_map="auto" verwendet werden (auch wenn Config "cuda" sagt)
                    device_map = "auto"
                elif device_map == "cuda":
                    # Kein Budget gesetzt, aber device_map="cuda" - prüfe ob max_memory in Performance-Settings gesetzt ist
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
                        device_map = "auto"  # Bei max_memory sollte device_map="auto" sein
                    else:
                        # Kein max_memory gesetzt - nutze ganze GPU (kein max_memory Parameter)
                        max_memory = None
                        logger.model_load("device_map='cuda' - nutze ganze GPU (kein max_memory Limit)")
                    
                    # device_map="cuda" bedeutet: Modell komplett auf GPU, kein CPU-Offloading
                    # Aber wenn max_memory gesetzt ist, verwenden wir "auto"
                    if max_memory is None:
                        disable_cpu_offload = True
            else:
                if self.device == "cuda":
                    # Prüfe GPU-Allokations-Budget zuerst (hat Priorität)
                    if self.gpu_allocation_budget_gb is not None and self.gpu_allocation_budget_gb > 0:
                        # GPU-Allokations-Budget ist gesetzt - verwende es
                        device_map = "auto"
                        max_memory = {0: f"{self.gpu_allocation_budget_gb:.2f}GB"}
                        logger.info(f"GPU-Allokations-Budget aktiv: max_memory={self.gpu_allocation_budget_gb:.2f}GB")
                        # WICHTIG: Quantisierung sollte trotzdem angewendet werden, wenn aktiviert
                        # (wird später in model_kwargs hinzugefügt)
                    elif use_quantization:
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
            # DEBUG: Logge auch perf_settings für Debugging
            logger.debug(f"DEBUG: perf_settings keys={list(perf_settings.keys())}, perf_settings['use_quantization']={perf_settings.get('use_quantization')}, model_loading_cfg.get('use_quantization')={model_loading_cfg.get('use_quantization')}, final use_quantization={use_quantization}")
            
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
                        logger.info(f"8-bit Quantisierung wird verwendet (GPU-Budget: {self.gpu_allocation_budget_gb}GB, max_memory: {max_memory})")
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
            
            # OPTIMIERUNG: Wenn Modell mit device_map="auto" geladen wurde, prüfe ob wir es vollständig auf GPU verschieben können
            # WICHTIG: Nur wenn KEINE Quantisierung aktiv ist (Quantisierung funktioniert nur mit device_map="auto")
            # Dies verbessert die Generierungsgeschwindigkeit erheblich
            if device_map == "auto" and self.device == "cuda" and torch.cuda.is_available() and not use_quantization:
                try:
                    # Prüfe ob Modell auf mehrere Devices verteilt ist
                    if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                        devices_used = set(self.model.hf_device_map.values())
                        if len(devices_used) > 1 or 'cpu' in str(devices_used).lower():
                            # Modell ist auf mehrere Devices verteilt - versuche es auf GPU zu konsolidieren
                            logger.info("Modell wurde mit device_map='auto' geladen und ist auf mehrere Devices verteilt")
                            
                            # Prüfe verfügbaren GPU-Speicher
                            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                            gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                            gpu_memory_available = gpu_memory_total - gpu_memory_reserved
                            
                            # Grobe Schätzung: 7B Modell braucht ~14GB (FP16), 3B Modell braucht ~6GB
                            # Bei Quantisierung: 7B braucht ~7.5GB (8-bit), 3B braucht ~3.5GB
                            model_size_estimate = 14.0 if "7b" in model_id.lower() else 6.0
                            
                            # Prüfe ob Budget es erlaubt (wenn Budget gesetzt)
                            budget_allows = True
                            if self.gpu_allocation_budget_gb is not None and self.gpu_allocation_budget_gb > 0:
                                # Budget ist gesetzt - prüfe ob genug Budget für vollständiges GPU-Laden
                                if gpu_memory_available < self.gpu_allocation_budget_gb * 0.9:
                                    budget_allows = False
                                    logger.info(f"GPU-Budget ({self.gpu_allocation_budget_gb}GB) erlaubt kein vollständiges GPU-Laden")
                            
                            # Wenn genug GPU-Speicher verfügbar ist UND Budget es erlaubt, verschiebe Modell auf GPU
                            if budget_allows and gpu_memory_available >= model_size_estimate * 0.8:  # 80% des geschätzten Modell-Speichers
                                logger.info(f"Genug GPU-Speicher verfügbar ({gpu_memory_available:.2f}GB) - verschiebe Modell vollständig auf GPU für bessere Performance")
                                try:
                                    # Verschiebe Modell auf GPU (device_map entfernen)
                                    self.model = self.model.to("cuda")
                                    logger.info("Modell erfolgreich auf GPU verschoben - Generierung wird schneller sein")
                                except Exception as move_error:
                                    logger.warning(f"Konnte Modell nicht auf GPU verschieben: {move_error} - verwende device_map='auto'")
                            else:
                                logger.info(f"Nicht genug GPU-Speicher verfügbar ({gpu_memory_available:.2f}GB) oder Budget-Limit - behalte device_map='auto'")
                except Exception as e:
                    logger.debug(f"Fehler beim Prüfen der Device-Verteilung: {e}")
            elif use_quantization:
                logger.debug("Quantisierung aktiv - behalte device_map='auto' (Quantisierung erfordert device_map='auto')")
            
            # Prüfe Performance-Settings für torch.compile() (wiederverwende bereits geladene perf_settings)
            use_torch_compile = perf_settings.get("use_torch_compile", False)
            
            # torch.compile() Support (PyTorch 2.0+)
            if use_torch_compile:
                try:
                    # Prüfe PyTorch Version (zentral, robust)
                    if self._version_geq(str(torch.__version__), "2.0.0"):
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
                    # #region agent log
                    try:
                        with open(r'g:\04-CODING\Local Ai\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A2","location":"model_manager.py:602","message":"Completeness check","data":{"is_complete":completeness["complete"],"response_length":len(response),"reason":completeness.get("reason")},"timestamp":int(time.time()*1000)})+"\n")
                    except: pass
                    # #endregion
                    
                    if completeness["complete"]:
                        return response
                    else:
                        # Response ist unvollständig - OPTIMIERT: Nur retry wenn wirklich kritisch
                        # Unvollständige Sätze sind OK wenn Response lang genug ist (>100 Zeichen)
                        is_critical_incomplete = len(response.strip()) < 100 or completeness.get("reason", "").startswith("Response zu kurz")
                        # #region agent log
                        try:
                            with open(r'g:\04-CODING\Local Ai\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A2","location":"model_manager.py:612","message":"Incomplete response decision","data":{"is_critical":is_critical_incomplete,"attempt":attempt,"max_retries":max_retries},"timestamp":int(time.time()*1000)})+"\n")
                        except: pass
                        # #endregion
                        
                        if is_critical_incomplete and attempt < max_retries and completeness.get("suggested_max_length"):
                            logger.warning(f"Response kritisch unvollständig: {completeness['reason']}, retry mit max_length={completeness['suggested_max_length']}")
                            current_max_length = completeness["suggested_max_length"]
                            continue
                        else:
                            # Response ist unvollständig, aber nicht kritisch - gebe zurück
                            if not is_critical_incomplete:
                                logger.debug(f"Response unvollständig aber akzeptabel ({len(response)} Zeichen): {completeness['reason']}")
                            else:
                                logger.warning(f"Response unvollständig, aber keine Retries mehr: {completeness['reason']}")
                            return response  # Gebe trotzdem zurück, da teilweise gültig
                else:
                    # Response ist ungültig - OPTIMIERT: Prüfe ob wirklich kritisch
                    # Wenn Response hauptsächlich Sonderzeichen ist, ist es kritisch
                    # Wenn Response nur zu kurz ist, ist es weniger kritisch
                    is_critical_invalid = len(response.strip()) < 20 or not any(c.isalnum() for c in response)
                    
                    if is_critical_invalid and attempt < max_retries:
                        logger.warning(f"Kritisch ungültige Response bei Versuch {attempt + 1}/{max_retries + 1}, retry...")
                        # Erhöhe max_length für Retry
                        current_max_length = int(current_max_length * 1.5)
                        continue
                    else:
                        if not is_critical_invalid:
                            logger.debug(f"Response ungültig aber akzeptabel ({len(response)} Zeichen), gebe zurück")
                            return response  # Gebe zurück auch wenn ungültig
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
            if self._is_short_response_allowed(messages, response_stripped):
                logger.debug("[Validation] Kurze Antwort ist erlaubt")
                return True
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
        
        # NEUE ROBUSTE PRÜFUNG: Verwende _validate_response_quality für Sonderzeichen-Erkennung
        if not self._validate_response_quality(response):
            logger.debug(f"[Validation] Response-Qualität ungültig (hauptsächlich Sonderzeichen)")
            return False
        
        import re
        words = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]+\b', response_stripped)
        logger.debug(f"[Validation] Response ist gültig: {len(response_stripped)} Zeichen, {len(words)} Wörter")
        return True

    def _is_short_response_allowed(self, messages: List[Dict[str, str]], response: str) -> bool:
        """
        Erlaubt sehr kurze Antworten, wenn die User-Nachricht explizit danach fragt.
        """
        if not response:
            return False
        response_lower = response.strip().lower()
        if not messages:
            return False
        last_user = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        if not last_user:
            return False
        prompt = last_user.get("content", "").lower()
        short_intents = [
            "nur mit dem wort", "nur das wort", "nur das", "antworte kurz",
            "kurz", "ja oder nein", "yes or no", "true or false", "ok"
        ]
        if any(k in prompt for k in short_intents):
            allowed = {"ok", "ja", "nein", "yes", "no", "true", "false", "1", "0"}
            return response_lower in allowed or len(response_lower) <= 4
        # Datei-Operationen: kurze Bestätigung akzeptieren
        if ("datei" in prompt and ("schreibe" in prompt or "speichere" in prompt)) or "write_file" in prompt:
            allowed = {"ok", "ja", "nein", "yes", "no", "true", "false", "1", "0"}
            return response_lower in allowed or len(response_lower) <= 4
        return False
    
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
                if self._is_short_response_allowed(messages, response_stripped):
                    return {
                        "complete": True,
                        "reason": "Kurzantwort explizit erlaubt",
                        "suggested_max_length": None
                    }
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
    
    def _validate_response_quality(self, response: str) -> bool:
        """
        Prüft ob Response gültige Text-Inhalte enthält (nicht nur Sonderzeichen)
        
        Args:
            response: Die zu prüfende Response
            
        Returns:
            True wenn Response gültig ist, False wenn hauptsächlich Sonderzeichen
        """
        import re
        
        if not response or len(response.strip()) < 5:
            return False
        
        # Entferne Whitespace für Berechnung
        text_without_whitespace = response.replace(' ', '').replace('\n', '').replace('\t', '')
        
        if len(text_without_whitespace) == 0:
            return False
        
        # Zähle alphanumerische Zeichen (Buchstaben und Ziffern)
        alphanumeric_chars = len(re.findall(r'[a-zA-ZäöüßÄÖÜ0-9]', response))
        total_chars = len(text_without_whitespace)
        
        # Berechne Anteil alphanumerischer Zeichen
        alphanumeric_ratio = alphanumeric_chars / total_chars if total_chars > 0 else 0
        
        # Wenn weniger als 30% alphanumerische Zeichen: wahrscheinlich nur Sonderzeichen
        if alphanumeric_ratio < 0.3:
            logger.warning(f"[VALIDATION] Response enthält hauptsächlich Sonderzeichen: {alphanumeric_ratio:.2%} alphanumerisch, Response: {repr(response[:100])}")
            return False
        
        # Prüfe ob mindestens ein Wort vorhanden ist (mindestens 2 Buchstaben)
        words = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]{2,}\b', response)
        if len(words) == 0:
            logger.warning(f"[VALIDATION] Response enthält keine Wörter (mind. 2 Buchstaben), Response: {repr(response[:100])}")
            return False
        
        return True
    
    def _clean_response_minimal(self, response: str, messages: List[Dict[str, str]], original_prompt: str = "") -> str:
        """
        🔧 ROBUSTE RESPONSE-BEREINIGUNG (4-Phasen-Ansatz)
        
        Implementiert einen robusten, validierten Cleaning-Prozess mit Fallback-Mechanismus.
        Garantiert dass Code-Blocks intakt bleiben und Responses gültig sind.
        
        Prinzipien:
        1. Fail-Safe: Immer Fallback auf Original
        2. Validierung nach jedem Schritt
        3. Modell-spezifisch
        4. Code-Block-Priorität
        5. Minimal Invasiv
        
        Args:
            response: Die rohe Response vom Modell
            messages: Die Message-History
            original_prompt: Der originale Prompt (optional)
            
        Returns:
            Bereinigte Response
        """
        import re
        
        # Backup für Fallback
        original_response = response
        
        # ============================================================
        # PHASE 1: Token-Level Cleaning (KRITISCH)
        # ============================================================
        response = self._clean_tokens(response)
        if not self._validate_basic(response):
            logger.warning("[Clean] Phase 1 (Token) fehlgeschlagen, verwende Fallback")
            return self._fallback_clean(original_response)
        
        # ============================================================
        # PHASE 2: Struktur-Level Cleaning (WICHTIG)
        # ============================================================
        response = self._clean_structure(response, messages)
        if not self._validate_basic(response):
            logger.warning("[Clean] Phase 2 (Struktur) fehlgeschlagen, verwende Fallback")
            return self._fallback_clean(original_response)
        
        # ============================================================
        # PHASE 3: Content-Level Cleaning (OPTIONAL)
        # ============================================================
        if self._needs_content_cleaning(response):
            response = self._clean_content(response)
            if not self._validate_basic(response):
                logger.warning("[Clean] Phase 3 (Content) fehlgeschlagen, verwende Fallback")
                return self._fallback_clean(original_response)
        
        # ============================================================
        # PHASE 4: Final Validation
        # ============================================================
        if not self._validate_final(response):
            logger.warning("[Clean] Finale Validierung fehlgeschlagen, verwende Fallback")
            return self._fallback_clean(original_response)
        
        logger.info(f"[Clean] Response erfolgreich bereinigt: {len(original_response)} -> {len(response)} Zeichen")
        return response

    @staticmethod
    def _normalize_mixed_script_homoglyphs(text: str) -> str:
        """
        Normalisiert häufige Cyrillic-Homoglyphen (z.B. "Schritт", "marmelадe")
        zu lateinischen Zeichen, wenn der Text überwiegend lateinisch ist.

        Ziel: sichtbare Tippfehler durch gemischte Alphabete beheben, ohne echte
        nicht-lateinische Antworten aggressiv zu verändern.
        """
        import re

        if not text:
            return text

        # Nur anwenden, wenn Cyrillic vorkommt UND der Text überwiegend lateinisch wirkt.
        cyr = re.findall(r"[\u0400-\u04FF]", text)
        if not cyr:
            return text
        latin = re.findall(r"[A-Za-zÄÖÜäöüß]", text)
        if len(latin) < max(12, len(cyr) * 3):
            return text

        # Minimaler, konservativer Satz an Mappings:
        # - primär visuell ähnliche Zeichen + häufige "Fehlgriffe" (д -> d, л -> l)
        mapping = {
            "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I",
            "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "с": "c", "т": "t", "х": "x", "і": "i",
            # häufige Zusatz-Fehlgriffe (nicht nur Homoglyphen)
            "д": "d", "Д": "D",
            "л": "l", "Л": "L",
            "у": "y", "У": "Y",
        }

        # Schneller Pfad: nur ersetzen, wenn tatsächlich ein Mapping vorkommt
        if not any(ch in mapping for ch in cyr):
            return text

        return "".join(mapping.get(ch, ch) for ch in text)
    
    def _clean_tokens(self, response: str) -> str:
        """
        Phase 1: Token-Level Cleaning
        Entfernt Modell-spezifische Tokens und Stop-Sequenzen
        """
        # 1. Chat-Template-Tokens (Qwen, LLaMA)
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "")
        
        # 2. Special Tokens
        special_tokens = ["</s>", "<|end|>", "<|endoftext|>"]
        for token in special_tokens:
            response = response.replace(token, "")
        
        # 3. Assistant Prefix entfernen
        response = response.strip()
        if response.lower().startswith('assistant:'):
            response = response[10:].strip()
        elif response.lower().startswith('assistant '):
            response = response[9:].strip()
        
        # 4. VERBESSERT: Unicode-Vollbreiten-Ziffern (全角数字) zu ASCII-Ziffern konvertieren
        # Qwen-Modelle können manchmal Unicode-Vollbreiten-Ziffern generieren
        fullwidth_to_ascii = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            '＋': '+', '－': '-', '×': '*', '÷': '/', '＝': '=',
            '。': '.', '，': ',', '：': ':', '；': ';', '？': '?', '！': '!',
            # Zusätzliche Varianten die auftreten können
            '１': '1', '２': '2', '３': '3', '５': '5', '７': '7', '８': '8'
        }
        for fullwidth, ascii_char in fullwidth_to_ascii.items():
            response = response.replace(fullwidth, ascii_char)
        
        # VERBESSERT: Konvertiere auch alle anderen Vollbreiten-Zeichen zu ASCII
        import unicodedata
        try:
            # Konvertiere Vollbreiten-Zeichen zu normalen Zeichen
            response = unicodedata.normalize('NFKC', response)
        except:
            pass  # Fallback wenn unicodedata nicht verfügbar

        # 5. Cyrillic-Homoglyphen in überwiegend lateinischem Text normalisieren
        response = self._normalize_mixed_script_homoglyphs(response)
        
        return response
    
    def _clean_structure(self, response: str, messages: List[Dict[str, str]]) -> str:
        """
        Phase 2: Struktur-Level Cleaning
        Entfernt Chat-Marker und schützt Code-Blocks
        """
        import re
        
        response_before = response
        
        # 1. Code-Blocks schützen (temporär ersetzen) - WICHTIG: Multiline Pattern
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, response)
        code_block_map = {}
        for i, code_block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            code_block_map[placeholder] = code_block
            response = response.replace(code_block, placeholder, 1)
        
        # 2. Prüfe ob Response hauptsächlich Code ist (mehr als 50% Code-Blocks)
        total_code_length = sum(len(cb) for cb in code_blocks)
        if total_code_length > len(response_before) * 0.5:
            # Response ist hauptsächlich Code - sehr vorsichtig mit Trimming
            logger.debug("[Clean] Response ist hauptsächlich Code, verwende konservative Trimming-Strategie")
            # Nur trimmen wenn Marker sehr weit hinten (>80% der Response)
            user_markers = ['Human:', 'User:']
            for marker in user_markers:
                positions = [m.start() for m in re.finditer(re.escape(marker), response, re.IGNORECASE)]
                for pos in positions:
                    if pos > len(response) * 0.8:  # Nur wenn sehr weit hinten
                        response = response[:pos].strip()
                        logger.debug(f"[Clean] Trimmed at user marker '{marker}' at position {pos} (Code-Response)")
                        break
        else:
            # Normale Response - normale Trimming-Strategie
            user_markers = ['Human:', 'User:']  # Immer trimmen
            assistant_markers = ['Assistant:', 'System:']  # Nur trimmen wenn wenig Content
            
            # Zuerst: User-Marker (intelligente Behandlung)
            for marker in user_markers:
                positions = [m.start() for m in re.finditer(re.escape(marker), response, re.IGNORECASE)]
                for pos in positions:
                    if pos > 50:  # Marker ist nicht am Anfang
                        # VERBESSERT: Wenn "User:" Marker gefunden wird, schneide IMMER VOR dem Marker ab
                        # Das Modell hat weiter generiert und User/Assistant-Marker erstellt - die eigentliche Antwort ist VOR dem Marker
                        before_marker = response[:pos].strip()
                        
                        # Prüfe ob vor Marker genug Content ist (mindestens 10 Zeichen)
                        if len(before_marker) >= 10:
                            response = before_marker
                            logger.debug(f"[Clean] User-Marker gefunden, schneide VOR Marker ab ({len(response)} Zeichen)")
                            # #region agent log
                            try:
                                import time
                                with open(r'g:\04-CODING\Local Ai\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A1","location":"model_manager.py:1008","message":"Cut before User marker","data":{"before_length":len(before_marker),"after_length":len(response[pos:])},"timestamp":int(time.time()*1000)})+"\n")
                            except: pass
                            # #endregion
                            break
                        else:
                            # Zu wenig Content vor Marker - möglicherweise falscher Marker, ignoriere
                            logger.debug(f"[Clean] User-Marker gefunden, aber zu wenig Content davor ({len(before_marker)} Zeichen), ignoriere")
                if len(response) < len(response_before) * 0.7:  # Nur wenn weniger als 70% übrig
                    break
            
            # Dann: Assistant-Marker (nur wenn wenig Content)
            for marker in assistant_markers:
                positions = [m.start() for m in re.finditer(re.escape(marker), response, re.IGNORECASE)]
                for pos in positions:
                    if pos > 100:  # Mindestens 100 Zeichen vom Anfang
                        after_marker = response[pos + len(marker):].strip()
                        # Wenn nach Marker wenig Content ODER keine Buchstaben
                        if len(after_marker) < 50 or not re.search(r'[a-zA-ZäöüßÄÖÜ]', after_marker[:100]):
                            response = response[:pos].strip()
                            logger.debug(f"[Clean] Trimmed at assistant marker '{marker}' at position {pos}")
                            break
                if len(response) < len(response_before) * 0.7:  # Nur wenn weniger als 70% übrig
                    break
        
        # 3. Code-Blocks wieder einfügen
        for placeholder, code_block in code_block_map.items():
            response = response.replace(placeholder, code_block)
        
        return response
    
    def _needs_content_cleaning(self, response: str) -> bool:
        """
        Prüft ob Content-Level Cleaning nötig ist
        """
        import re
        
        # Prüfe ob HTML-Tags vorhanden
        if re.search(r'<[^>]+>', response):
            return True

        # Häufige Markdown-Artefakte / Escapes
        if re.search(r'(?m)^\s*\d{1,2}\\\.', response):
            return True

        # CamelCase / fehlende Leerzeichen durch Innen-Großbuchstaben (z.B. "währendUDP", "DatagramProtocol")
        if re.search(r'[a-zäöüß][A-ZÄÖÜ]', response):
            return True

        # Offensichtlicher Sprach-Mix (kleines, gezieltes Pattern)
        if "Sufficient Sleep" in response:
            return True

        # Häufiger Sprach-Mix in DE-Antworten
        if re.search(r'\bStep\s+\d{1,2}\b', response):
            return True

        # Plural/Typo, der häufig vorkommt
        if "Backend-Servers" in response or "Backend-Server" in response:
            # auch wenn korrekt, Cleaning ist harmlos (nur Normalisierung)
            return True
        
        # Prüfe ob CJK-Zeichen vorhanden (nur wenn Response Buchstaben hat)
        if re.search(r'[a-zA-ZäöüßÄÖÜ]', response):
            cjk_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf]'
            if re.search(cjk_pattern, response):
                return True
        
        return False
    
    def _clean_content(self, response: str) -> str:
        """
        Phase 3: Content-Level Cleaning (OPTIONAL)
        Entfernt Artefakte ohne Content zu beschädigen
        WICHTIG: Code-Blocks werden geschützt
        """
        import re
        
        # 1. Code-Blocks schützen (temporär ersetzen)
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, response)
        code_block_map = {}
        for i, code_block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            code_block_map[placeholder] = code_block
            response = response.replace(code_block, placeholder, 1)
        
        # 2. HTML-Tags entfernen (nur außerhalb Code-Blocks)
        html_pattern = r'<[^>]+>'
        response = re.sub(html_pattern, '', response)
        
        # 3. CJK-Zeichen entfernen (nur wenn Buchstaben vorhanden UND nicht hauptsächlich Code)
        # Prüfe ob Response hauptsächlich Code ist
        total_code_length = sum(len(cb) for cb in code_blocks)
        is_mostly_code = total_code_length > len(response) * 0.5
        
        # VERBESSERT: Entferne CJK-Zeichen aggressiver, besonders wenn sie mitten in deutscher Antwort stehen
        if re.search(r'[a-zA-ZäöüßÄÖÜ]', response) and not is_mostly_code:
            # Entferne CJK-Zeichen (Chinesisch, Japanisch, Koreanisch)
            cjk_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+'
            response = re.sub(cjk_pattern, '', response)
            
            # Entferne auch chinesische Satzzeichen die übrig bleiben könnten
            chinese_punct = ['，', '。', '：', '；', '？', '！', '、', '…', '「', '」', '『', '』']
            for punct in chinese_punct:
                response = response.replace(punct, '')
        
        # 4. VERBESSERT: Behebe unvollständige Wörter (z.B. "zu-toolerngigen" → "zu toolerngigen")
        # Finde Wörter mit Bindestrich die unvollständig aussehen
        hyphen_word_pattern = r'\b\w+-\w+\b'
        def fix_hyphen_word(match):
            word = match.group(0)
            # Wenn das Wort nach dem Bindestrich sehr kurz ist (< 5 Zeichen), entferne Bindestrich
            parts = word.split('-')
            if len(parts) == 2 and len(parts[1]) < 5:
                # Prüfe ob es wie ein unvollständiges Wort aussieht
                if not re.match(r'^[a-z]+$', parts[1], re.IGNORECASE):
                    return parts[0] + ' ' + parts[1]  # Ersetze Bindestrich durch Leerzeichen
            return word
        
        response = re.sub(hyphen_word_pattern, fix_hyphen_word, response)

        # 4b. Fixe häufige Markdown-Escapes in nummerierten Listen: "1\\." -> "1."
        response = re.sub(r'(?m)^(\s*\d{1,2})\\\.', r'\1.', response)

        # 4c. CamelCase/Innen-Großbuchstaben in Wörtern auflösen: "DatagramProtocol" -> "Datagram Protocol"
        # (nur außerhalb Code-Blöcke; URLs werden hier ebenfalls nicht erwartet)
        response = re.sub(r'([a-zäöüß])([A-ZÄÖÜ])', r'\1 \2', response)

        # 4d. Mini-Language-Fix: häufige englische Phrase, die sonst als "Gemisch" wirkt
        response = response.replace("Sufficient Sleep", "Ausreichender Schlaf")
        response = re.sub(r"\bStep\s+(\d{1,2})\b", r"Schritt \1", response)
        response = re.sub(r"\bBackend-Servers?\b", "Backend-Server", response)
        
        # 5. Whitespace normalisieren (nur außerhalb Code-Blocks)
        # Vorsichtig: Nur mehrfache Leerzeichen in Text, nicht in Code
        response = re.sub(r' +', ' ', response)  # Mehrfache Leerzeichen
        response = re.sub(r'\n\s*\n+', '\n\n', response)  # Mehrfache Zeilenumbrüche
        
        # 6. Code-Blocks wieder einfügen
        for placeholder, code_block in code_block_map.items():
            response = response.replace(placeholder, code_block)
        
        # Entferne verwaiste Code-Fences wenn kein Code-Block erkannt wurde
        if "```" in response and not code_blocks:
            response = response.replace("```", "")
        
        # 6. Letzte Sicherheits-Trim bei Chat-Markern (weniger aggressiv)
        marker_matches = []
        for marker in ['User:', 'Assistant:']:
            positions = [m.start() for m in re.finditer(re.escape(marker), response, re.IGNORECASE)]
            marker_matches.extend(positions)
        if marker_matches:
            first_marker = min(marker_matches)
            # Nur abschneiden wenn Marker weit genug hinten ist (mindestens 20 Zeichen Response davor)
            if first_marker >= 20:
                response = response[:first_marker].strip()
            else:
                # Marker ist zu früh - entferne nur den Marker selbst, nicht den Text danach
                response = re.sub(r'\b(?:User|Assistant)\s*[:：]\s*', '', response, flags=re.IGNORECASE, count=1)
        
        return response.strip()
    
    def _validate_basic(self, response: str) -> bool:
        """
        Basis-Validierung: Prüft ob Response grundsätzlich gültig ist
        Code-Blocks werden als gültig akzeptiert, auch ohne Buchstaben
        """
        import re
        
        if not response or len(response.strip()) == 0:
            return False
        if len(response.strip()) < 5:
            short = response.strip().lower()
            allowed_short = {"ok", "ja", "nein", "yes", "no", "true", "false", "1", "0"}
            if short in allowed_short:
                return True
            # Erlaube sehr kurze alphanumerische Antworten (mind. 2 Zeichen)
            if len(short) >= 2 and any(c.isalnum() for c in short):
                return True
            return False
        
        # Prüfe ob Response Code-Blocks enthält
        code_block_pattern = r'```[\s\S]*?```'
        has_code_blocks = bool(re.search(code_block_pattern, response))
        
        # Wenn Code-Blocks vorhanden, ist Response gültig (auch ohne Buchstaben)
        if has_code_blocks:
            return True
        
        # Prüfe ob Response Buchstaben enthält
        if not re.search(r'[a-zA-ZäöüßÄÖÜ]', response):
            return False
        return True
    
    def _validate_final(self, response: str) -> bool:
        """
        Finale Validierung: Prüft ob Response vollständig und gültig ist
        Code-Blocks werden als gültig akzeptiert, auch ohne Wörter
        """
        import re
        
        if not self._validate_basic(response):
            return False
        
        # Prüfe ob Response Code-Blocks enthält
        code_block_pattern = r'```[\s\S]*?```'
        has_code_blocks = bool(re.search(code_block_pattern, response))
        
        # Wenn Code-Blocks vorhanden, ist Response gültig (auch ohne Wörter)
        if has_code_blocks:
            return True
        
        # Prüfe ob Response Wörter enthält (nicht nur Sonderzeichen)
        words = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]+\b', response)
        if len(words) == 0:
            return False
        
        return True
    
    def _fallback_clean(self, original_response: str) -> str:
        """
        Fallback: Minimal Cleaning auf Original
        WICHTIG: Code-Blocks werden geschützt, nur sicherste Schritte
        Weniger aggressiv als vorher - behält mehr Text
        """
        import re
        
        response = original_response
        
        # 1. Code-Blocks schützen (temporär ersetzen)
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, response)
        code_block_map = {}
        for i, code_block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            code_block_map[placeholder] = code_block
            response = response.replace(code_block, placeholder, 1)
        
        # 2. Prüfe ob Response hauptsächlich Code ist
        total_code_length = sum(len(cb) for cb in code_blocks)
        is_mostly_code = total_code_length > len(original_response) * 0.5
        
        # 3. HTML entfernen (nur außerhalb Code-Blocks) - weniger aggressiv
        # Entferne nur bekannte HTML-Tags, nicht alle <...>
        html_tags_to_remove = ['<br>', '<br/>', '<p>', '</p>', '<div>', '</div>', '<span>', '</span>']
        for tag in html_tags_to_remove:
            response = response.replace(tag, '')
        # Entferne nur unbekannte Tags wenn sie offensichtlich HTML sind (z.B. <tag>)
        html_pattern = r'<[a-zA-Z][^>]*>'
        response = re.sub(html_pattern, '', response)
        
        # 4. CJK entfernen NUR wenn Response hauptsächlich lateinische Zeichen hat
        # UND Response ist lang genug (>50 Zeichen) UND nicht hauptsächlich Code
        if len(response) > 50 and re.search(r'[a-zA-ZäöüßÄÖÜ]', response) and not is_mostly_code:
            # Prüfe ob Response hauptsächlich lateinische Zeichen hat
            latin_chars = len(re.findall(r'[a-zA-ZäöüßÄÖÜ]', response))
            cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', response))
            # Nur entfernen wenn CJK weniger als 20% der Zeichen sind
            if cjk_chars > 0 and latin_chars > cjk_chars * 4:
                cjk_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf]+'
                response = re.sub(cjk_pattern, '', response)
        
        # 5. Chat-Marker trimmen (nur wenn NICHT hauptsächlich Code ODER sehr weit hinten)
        if not is_mostly_code:
            markers = ['Human:', 'Assistant:', 'User:', 'System:']
            for marker in markers:
                positions = [m.start() for m in re.finditer(re.escape(marker), response, re.IGNORECASE)]
                for pos in positions:
                    if pos > 100:  # Mindestens 100 Zeichen vom Anfang
                        after_marker = response[pos + len(marker):].strip()
                        if len(after_marker) < 50:  # Wenn nach Marker wenig Content
                            response = response[:pos].strip()
                            logger.debug(f"[Clean] Fallback: Trimmed at marker '{marker}'")
                            break
        else:
            # Bei Code-Response: Nur trimmen wenn Marker sehr weit hinten (>90%)
            markers = ['Human:', 'User:']
            for marker in markers:
                positions = [m.start() for m in re.finditer(re.escape(marker), response, re.IGNORECASE)]
                for pos in positions:
                    if pos > len(response) * 0.9:  # Nur wenn sehr weit hinten
                        response = response[:pos].strip()
                        logger.debug(f"[Clean] Fallback: Trimmed at marker '{marker}' (Code-Response)")
                        break
        
        # 6. Code-Blocks wieder einfügen
        for placeholder, code_block in code_block_map.items():
            response = response.replace(placeholder, code_block)
        
        # 7. Whitespace normalisieren (nur außerhalb Code-Blocks)
        response = re.sub(r'\n\s*\n+', '\n\n', response)
        
        logger.warning(f"[Clean] Fallback verwendet: {len(original_response)} -> {len(response)} Zeichen")
        return response.strip()
    
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
                # DEBUG: Logge formatierten Prompt (nur für Debugging)
                logger.debug(f"[DEBUG] Formatierten Prompt (erste 500 Zeichen): {prompt[:500]}")
                logger.debug(f"[DEBUG] Formatierten Prompt (letzte 200 Zeichen): {prompt[-200:]}")
                
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
            
            # Verwende max_length als gewünschte Ausgabelänge (max_new_tokens)
            desired_new_tokens = max_length
            if desired_new_tokens >= model_max_context:
                desired_new_tokens = max(1, model_max_context - 1)
                logger.warning(f"max_length zu hoch für Modell-Kontext. Setze gewünschte Ausgabelänge auf {desired_new_tokens} Tokens (Modell-Limit: {model_max_context}).")
            
            # Maximale Eingabe-Länge basierend auf gewünschter Ausgabelänge
            max_input_tokens = model_max_context - desired_new_tokens
            if max_input_tokens < 1:
                raise ValueError(f"Modell-Kontext zu klein für gewünschte Ausgabelänge: model_max_context={model_max_context}, desired_new_tokens={desired_new_tokens}")
            
            # FIX: Wenn Input zu lang ist, kürze die Messages automatisch
            if input_length > max_input_tokens:
                logger.warning(f"Input ist zu lang ({input_length} Tokens). Maximal erlaubte Eingabe: {max_input_tokens}, Modell-Limit: {model_max_context}. Kürze Messages...")
                
                # Versuche, die letzten Messages zu kürzen (behält System-Prompt und erste User-Message)
                if len(messages) > 2:
                    # Behalte System-Prompt (wenn vorhanden) und erste User-Message
                    system_msg = messages[0] if messages[0].get("role") == "system" else None
                    first_user_msg = None
                    other_messages = []
                    
                    for msg in messages:
                        if msg.get("role") == "system":
                            continue  # System wird separat behandelt
                        elif msg.get("role") == "user" and first_user_msg is None:
                            first_user_msg = msg
                        else:
                            other_messages.append(msg)
                    
                    # Kürze andere Messages (entferne älteste zuerst)
                    while input_length > max_input_tokens and len(other_messages) > 0:
                        removed = other_messages.pop(0)  # Entferne älteste Message
                        logger.debug(f"Entferne Message aus History: {removed.get('role', 'unknown')}")
                        
                        # Rebuild prompt und prüfe Länge
                        new_messages = []
                        if system_msg:
                            new_messages.append(system_msg)
                        if first_user_msg:
                            new_messages.append(first_user_msg)
                        new_messages.extend(other_messages)
                        
                        new_prompt = self._format_messages(new_messages)
                        new_inputs = self.tokenizer(new_prompt, return_tensors="pt")
                        input_length = new_inputs['input_ids'].shape[1]
                        
                        # Update messages und inputs
                        messages = new_messages
                        prompt = new_prompt
                        inputs = new_inputs
                    
                    # Wenn immer noch zu lang, kürze die letzte User-Message
                    if input_length > max_input_tokens and first_user_msg:
                        # Kürze die erste User-Message auf max. 50% der erlaubten Eingabelänge
                        max_user_tokens = max_input_tokens // 2
                        user_content = first_user_msg.get("content", "")
                        user_inputs = self.tokenizer(user_content, return_tensors="pt")
                        user_length = user_inputs['input_ids'].shape[1]
                        
                        if user_length > max_user_tokens:
                            # Kürze Text (ungefähr)
                            ratio = max_user_tokens / user_length
                            new_user_content = user_content[:int(len(user_content) * ratio * 0.9)]  # 90% Sicherheitsmarge
                            first_user_msg["content"] = new_user_content
                            logger.warning(f"User-Message gekürzt von {user_length} auf ~{max_user_tokens} Tokens")
                            
                            # Rebuild prompt
                            new_messages = []
                            if system_msg:
                                new_messages.append(system_msg)
                            new_messages.append(first_user_msg)
                            new_messages.extend(other_messages)
                            
                            prompt = self._format_messages(new_messages)
                            inputs = self.tokenizer(prompt, return_tensors="pt")
                            input_length = inputs['input_ids'].shape[1]
                            messages = new_messages
                    
                    logger.info(f"Messages gekürzt. Neue Input-Länge: {input_length} Tokens")
                else:
                    # Nur 1-2 Messages - kann nicht gekürzt werden
                    raise ValueError(f"Input ist zu lang ({input_length} Tokens). Maximal erlaubte Eingabe: {max_input_tokens}, Modell-Limit: {model_max_context}. Bitte kürze deine Nachricht.")
            
            # Berechne max_new_tokens korrekt (max_length = gewünschte Ausgabelänge)
            max_new_tokens = min(
                desired_new_tokens,
                model_max_context - input_length
            )
            
            # Validierung: Prüfe ob max_new_tokens zu klein ist (BEVOR wir es auf 1 setzen)
            if max_new_tokens <= 0:
                raise ValueError(f"Input ist zu lang ({input_length} Tokens). Maximal erlaubte Eingabe: {max_input_tokens}, Modell-Limit: {model_max_context}")
            
            # Stelle sicher, dass max_new_tokens mindestens 1 ist (nach Validierung)
            max_new_tokens = max(1, max_new_tokens)

            # Auto-Cap für kurze Anfragen (spart Zeit & vermeidet Runaway)
            max_new_tokens = self._auto_cap_max_new_tokens(messages, max_new_tokens, is_coding=is_coding)
            
            # Logging für Debugging
            logger.info(f"Input-Länge: {input_length}, gewünschte Ausgabelänge: {desired_new_tokens}, max_new_tokens: {max_new_tokens}, Modell: {self.current_model_id}")
            
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
                
                # Optimierte Repetition Penalty für verschiedene Use Cases
                if is_coding:
                    repetition_penalty = 1.1  # Niedriger für Code (erlaubt Wiederholungen)
                elif is_mistral:
                    repetition_penalty = 1.3  # Höher für Mistral (verhindert Loops)
                elif is_qwen:
                    repetition_penalty = 1.15  # Mittel für Qwen (ausgewogen)
                else:
                    repetition_penalty = 1.2  # Standard
                
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
                
                # Dynamisches max_time: min(30s) war zu hoch und führte häufig zu 30s-Runs,
                # wenn Qwen kein EOS emittiert. Lieber max_new_tokens "gewinnen" lassen.
                max_time_seconds = max(12.0, min(180.0, max_new_tokens * 0.25))
                logger.warning(f"[DEBUG] BEFORE model.generate() - eos_token_id={eos_token_id_for_generate}, max_new_tokens={max_new_tokens}, temperature={temperature}, max_time={max_time_seconds}s")
                
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
                    # Optimierte Parameter für Qwen
                    if is_qwen:
                        # Qwen: Verwende moderate Temperature für bessere Qualität
                        effective_temperature = max(0.3, temperature) if temperature > 0 else 0.7
                        effective_top_p = 0.9
                        effective_top_k = None  # Qwen funktioniert besser ohne top_k
                    else:
                        effective_temperature = temperature if temperature > 0 else (0.7 if is_mistral else None)
                        effective_top_p = 0.9 if temperature > 0 else None
                        effective_top_k = 50 if temperature > 0 and is_mistral else None
                    
                    logger.debug(f"[DEBUG] Generate-Parameter: temp={effective_temperature}, top_p={effective_top_p}, top_k={effective_top_k}, rep_penalty={repetition_penalty}")
                    
                    # Für Qualität/Determinismus: erst ab höheren Temperaturen samplen.
                    do_sample = bool(effective_temperature is not None and float(effective_temperature) >= 0.5)
                    gen_kwargs: Dict[str, Any] = dict(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        max_time=max_time_seconds,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
                        eos_token_id=eos_token_id_for_generate,  # Kann jetzt Liste oder Integer sein
                        no_repeat_ngram_size=3,
                    )
                    if do_sample:
                        gen_kwargs.update(
                            temperature=effective_temperature,
                            top_p=effective_top_p,
                            top_k=effective_top_k,
                        )
                    else:
                        # Greedy decoding: keine Sampling-Parameter setzen
                        gen_kwargs.pop("temperature", None)
                        gen_kwargs.pop("top_p", None)
                        gen_kwargs.pop("top_k", None)

                    stopping_criteria = self._build_stopping_criteria()
                    if stopping_criteria is not None:
                        gen_kwargs["stopping_criteria"] = stopping_criteria

                    outputs = self.model.generate(**gen_kwargs)
                    
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
                
                # VERBESSERT: Prüfe auf <|endoftext|> Token nur wenn es nicht bereits in eos_token_id enthalten ist
                # UND nur wenn es wirklich am Ende steht (nicht mitten in der Antwort)
                try:
                    endoftext_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
                    if endoftext_token_id is not None and endoftext_token_id != self.tokenizer.unk_token_id:
                        # Prüfe ob endoftext_token_id bereits in eos_token_id enthalten ist
                        is_already_eos = False
                        if isinstance(eos_token_id, list):
                            is_already_eos = endoftext_token_id in eos_token_id
                        else:
                            is_already_eos = endoftext_token_id == eos_token_id
                        
                        if not is_already_eos:
                            endoftext_positions = (new_tokens == endoftext_token_id).nonzero(as_tuple=True)[0]
                            if len(endoftext_positions) > 0:
                                # Nur behandeln wenn es in den letzten 10% der Tokens steht (echtes Ende)
                                # ODER wenn es das einzige Token ist
                                for pos in endoftext_positions:
                                    pos_val = pos.item()
                                    if pos_val >= len(new_tokens) * 0.9 or len(new_tokens) < 10:
                                        eos_positions.append(pos_val)
                                        logger.debug(f"[FIX] <|endoftext|> Token gefunden an Position {pos_val} (am Ende)")
                                        # #region agent log
                                        try:
                                            with open(r'g:\04-CODING\Local Ai\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A2","location":"model_manager.py:1675","message":"endoftext token at end","data":{"position":pos_val,"total_tokens":len(new_tokens),"percent":pos_val/len(new_tokens)*100},"timestamp":int(time.time()*1000)})+"\n")
                                        except: pass
                                        # #endregion
                                    else:
                                        logger.debug(f"[FIX] <|endoftext|> Token ignoriert an Position {pos_val} (nicht am Ende, {pos_val/len(new_tokens)*100:.1f}%)")
                                        # #region agent log
                                        try:
                                            with open(r'g:\04-CODING\Local Ai\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A2","location":"model_manager.py:1685","message":"endoftext token ignored","data":{"position":pos_val,"total_tokens":len(new_tokens),"percent":pos_val/len(new_tokens)*100},"timestamp":int(time.time()*1000)})+"\n")
                                        except: pass
                                        # #endregion
                except Exception as e:
                    logger.debug(f"[FIX] Fehler bei <|endoftext|> Token-Prüfung: {e}")
                    pass
                
                # Schneide beim ersten EOS-Token ab
                if eos_positions:
                    new_tokens = new_tokens[:min(eos_positions)]
                else:
                    # WARNUNG: Kein EOS-Token gefunden - intelligente Abschneide-Logik
                    logger.warning(f"[DEBUG] Kein EOS-Token in Response gefunden! Verwende intelligente Abschneide-Logik.")
                    # #region agent log
                    try:
                        with open(r'g:\04-CODING\Local Ai\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A1","location":"model_manager.py:1672","message":"No EOS token found","data":{"new_tokens_count":len(new_tokens),"max_new_tokens":max_new_tokens,"eos_token_id":str(eos_token_id)},"timestamp":int(time.time()*1000)})+"\n")
                    except: pass
                    # #endregion
                    # Dekodiere temporär, um Satzenden zu finden
                    temp_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    import re
                    
                    # Finde alle vollständigen Sätze (endet mit Punkt, Ausrufezeichen oder Fragezeichen)
                    # Suche nach Satzenden gefolgt von Leerzeichen oder Zeilenumbruch
                    sentence_pattern = r'[.!?][\s\n]+'
                    sentence_matches = list(re.finditer(sentence_pattern, temp_response))
                    
                    # VERBESSERT: Prüfe auch auf "Human:" und "User:" Marker (zeigt dass Modell weiter generiert)
                    human_marker = re.search(r'Human:', temp_response, re.IGNORECASE)
                    user_marker = re.search(r'User:', temp_response, re.IGNORECASE)
                    
                    # Finde den ersten Marker (Human oder User)
                    first_marker = None
                    if human_marker and user_marker:
                        if human_marker.start() < user_marker.start():
                            first_marker = human_marker
                            marker_name = "Human:"
                        else:
                            first_marker = user_marker
                            marker_name = "User:"
                    elif human_marker:
                        first_marker = human_marker
                        marker_name = "Human:"
                    elif user_marker:
                        first_marker = user_marker
                        marker_name = "User:"
                    
                    if first_marker and first_marker.start() > 0:
                        # Schneide vor Marker ab
                        cut_pos = first_marker.start()
                        cut_text = temp_response[:cut_pos].strip()
                        # Re-encode um exakte Token-Position zu finden
                        cut_tokens = self.tokenizer.encode(cut_text, add_special_tokens=False)
                        if len(cut_tokens) < len(new_tokens):
                            new_tokens = new_tokens[:len(cut_tokens)]
                            logger.info(f"[FIX] Response vor '{marker_name}' Marker abgeschnitten: {len(new_tokens)} Tokens")
                            # #region agent log
                            try:
                                with open(r'g:\04-CODING\Local Ai\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A1","location":"model_manager.py:1755","message":"Cut before marker","data":{"marker":marker_name,"cut_tokens":len(cut_tokens),"original_tokens":len(new_tokens)+len(cut_tokens)},"timestamp":int(time.time()*1000)})+"\n")
                            except: pass
                            # #endregion
                    elif len(sentence_matches) > 0:
                        # VERBESSERT: Verwende mehrere Sätze statt nur den ersten
                        # Wenn max_new_tokens erreicht wurde, verwende die vollständige Antwort
                        # Ansonsten verwende bis zu 10 Sätze oder 80% der Tokens, je nachdem was kleiner ist
                        if len(new_tokens) >= max_new_tokens * 0.95:
                            # Antwort ist fast so lang wie max_new_tokens - verwende vollständige Antwort
                            logger.info(f"[FIX] max_new_tokens erreicht ({len(new_tokens)}/{max_new_tokens}), verwende vollständige Antwort")
                            # new_tokens bleibt unverändert
                        else:
                            # Verwende mehrere Sätze (bis zu 10 oder bis zu 80% der Tokens)
                            max_sentences = min(10, len(sentence_matches))
                            max_tokens_80_percent = int(len(new_tokens) * 0.8)
                            
                            # Finde die Position nach dem letzten gewünschten Satz
                            if max_sentences <= len(sentence_matches):
                                cut_pos = sentence_matches[max_sentences - 1].end()
                            else:
                                cut_pos = sentence_matches[-1].end()
                            
                            cut_text = temp_response[:cut_pos].strip()
                            cut_tokens = self.tokenizer.encode(cut_text, add_special_tokens=False)
                            
                            # Verwende das Minimum von: Satz-basierter Cut oder 80% der Tokens
                            if len(cut_tokens) < max_tokens_80_percent:
                                if len(cut_tokens) < len(new_tokens):
                                    new_tokens = new_tokens[:len(cut_tokens)]
                                    logger.info(f"[FIX] Response nach {max_sentences} Sätzen abgeschnitten: {len(new_tokens)} Tokens")
                            else:
                                # 80% ist kleiner - aber schneide nach vollständigem Wort ab
                                cut_text_80 = self.tokenizer.decode(new_tokens[:max_tokens_80_percent], skip_special_tokens=True)
                                # Finde letztes vollständiges Wort (suche rückwärts nach Leerzeichen oder Satzzeichen)
                                last_space = cut_text_80.rfind(' ')
                                last_punct = max(cut_text_80.rfind('.'), cut_text_80.rfind('!'), cut_text_80.rfind('?'))
                                last_word_end = max(last_space, last_punct)
                                
                                if last_word_end > len(cut_text_80) * 0.7:  # Mindestens 70% der 80%-Länge
                                    # Schneide nach vollständigem Wort
                                    cut_text_final = cut_text_80[:last_word_end + 1].strip()
                                    cut_tokens_final = self.tokenizer.encode(cut_text_final, add_special_tokens=False)
                                    if len(cut_tokens_final) < len(new_tokens):
                                        new_tokens = new_tokens[:len(cut_tokens_final)]
                                        logger.info(f"[FIX] Response nach 80% der Tokens abgeschnitten (nach vollständigem Wort): {len(new_tokens)} Tokens")
                                        # #region agent log
                                        try:
                                            with open(r'g:\04-CODING\Local Ai\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A1","location":"model_manager.py:1735","message":"Cut after complete word","data":{"original_tokens":len(new_tokens)+len(cut_tokens_final),"cut_tokens":len(cut_tokens_final),"cut_text_end":cut_text_final[-30:]},"timestamp":int(time.time()*1000)})+"\n")
                                        except: pass
                                        # #endregion
                                    else:
                                        # Fallback: verwende 80% direkt
                                        new_tokens = new_tokens[:max_tokens_80_percent]
                                        logger.info(f"[FIX] Response nach 80% der Tokens abgeschnitten: {len(new_tokens)} Tokens")
                                else:
                                    # Kein gutes Wort-Ende gefunden - verwende 80% direkt
                                    new_tokens = new_tokens[:max_tokens_80_percent]
                                    logger.info(f"[FIX] Response nach 80% der Tokens abgeschnitten: {len(new_tokens)} Tokens")
                    else:
                        # Keine Satzenden gefunden - verwende 80% der Tokens, aber nach vollständigem Wort
                        min_tokens = min(100, len(new_tokens))
                        max_tokens_80_percent = int(len(new_tokens) * 0.8)
                        cut_tokens = max(min_tokens, max_tokens_80_percent)
                        
                        if cut_tokens < len(new_tokens):
                            # Dekodiere bis zum Cut-Punkt und finde letztes vollständiges Wort
                            cut_text = self.tokenizer.decode(new_tokens[:cut_tokens], skip_special_tokens=True)
                            # Finde letztes vollständiges Wort (suche rückwärts nach Leerzeichen)
                            last_space = cut_text.rfind(' ')
                            last_punct = max(cut_text.rfind('.'), cut_text.rfind('!'), cut_text.rfind('?'))
                            last_word_end = max(last_space, last_punct)
                            
                            if last_word_end > len(cut_text) * 0.7:  # Mindestens 70% der Cut-Länge
                                # Schneide nach vollständigem Wort
                                cut_text_final = cut_text[:last_word_end + 1].strip()
                                cut_tokens_final = self.tokenizer.encode(cut_text_final, add_special_tokens=False)
                                if len(cut_tokens_final) < len(new_tokens):
                                    new_tokens = new_tokens[:len(cut_tokens_final)]
                                    logger.info(f"[FIX] Keine Satzenden gefunden, schneide nach vollständigem Wort ab: {len(new_tokens)} Tokens (ursprünglich {cut_tokens})")
                                else:
                                    # Fallback: verwende ursprünglichen Cut
                                    new_tokens = new_tokens[:cut_tokens]
                                    logger.warning(f"[FIX] Keine Satzenden gefunden, schneide nach {cut_tokens} Tokens ab (80% von {len(new_tokens)})")
                            else:
                                # Kein gutes Wort-Ende gefunden - verwende ursprünglichen Cut
                                new_tokens = new_tokens[:cut_tokens]
                                logger.warning(f"[FIX] Keine Satzenden gefunden, schneide nach {cut_tokens} Tokens ab (80% von {len(new_tokens)})")
                
                decode_start_time = time.time()
                # WICHTIG: Verwende skip_special_tokens=True standardmäßig, um korrekte Dekodierung zu gewährleisten
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                decode_duration = time.time() - decode_start_time
                logger.debug(f"[Generate] Raw decoded response length: {len(response)} chars, first 100 chars: {response[:100]}")
                logger.warning(f"[DEBUG] RAW RESPONSE (vollständig): {repr(response)}")  # WICHTIG: Vollständige Raw-Response für Debugging
                
                # Prüfe ob Response nur Sonderzeichen enthält (wird später in _validate_response_quality geprüft)
                # Versuche alternative Decodierung als Fallback
                import re
                has_letters = bool(re.search(r'[a-zA-ZäöüßÄÖÜ]', response))
                if not has_letters or len(response.strip()) < 5:
                    logger.warning(f"[WARNING] Response enthält keine Buchstaben oder ist zu kurz! Versuche alternative Decodierung...")
                    # Versuche mit skip_special_tokens=False als Fallback
                    response_alt = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                    logger.warning(f"[DEBUG] Alternative Decodierung (skip_special_tokens=False): {repr(response_alt[:100])}")
                    if bool(re.search(r'[a-zA-ZäöüßÄÖÜ]', response_alt)) and len(response_alt.strip()) >= 5:
                        response = response_alt
                        logger.info(f"[FIX] Alternative Decodierung hat bessere Response gefunden, verwende diese")
                    else:
                        logger.warning(f"[WARNING] Alternative Decodierung hat auch keine gültige Response gefunden. Validierung wird später prüfen und ggf. Retry auslösen.")
                logger.debug(f"[DEBUG] Decoding dauerte {decode_duration:.2f} Sekunden")
                logger.model_gen(f"Decoding abgeschlossen, Länge: {len(response)} Zeichen", level="debug")
                
                # 🔧 NEUE MINIMALISTISCHE BEREINIGUNG
                clean_start_time = time.time()
                logger.model_gen(f"Vor Cleaning: {len(response)} Zeichen", level="debug")
                logger.warning(f"[DEBUG] VOR CLEANING (vollständig): {repr(response)}")  # WICHTIG: Response vor Cleaning
                response = self._clean_response_minimal(response, messages, original_prompt)
                clean_duration = time.time() - clean_start_time
                logger.debug(f"[DEBUG] Cleaning dauerte {clean_duration:.2f} Sekunden")
                logger.warning(f"[DEBUG] NACH CLEANING (vollständig): {repr(response)}")  # WICHTIG: Response nach Cleaning
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
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # Device placement kompatibel mit device_map="auto"
            if hasattr(self.model, "device"):
                target_device = self.model.device
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            elif hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
                first_device = list(self.model.hf_device_map.values())[0]
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]
            
            # Bestimme Modell-Typ
            is_mistral = "mistral" in self.current_model_id.lower() if self.current_model_id else False
            is_qwen = self.current_model_id and "qwen" in self.current_model_id.lower()
            
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
                "qwen": 32768,    # Qwen-2.x hat 32k Token Kontext
                "qwen2": 32768,
                "default": 2048   # Standard-Limit
            }
            
            # Bestimme Modell-Limit
            if self.current_model_id and "qwen" in self.current_model_id.lower():
                model_max_context = model_limits.get("qwen", model_limits["default"])
            else:
                model_name = self.current_model_id.lower().split("-")[0] if self.current_model_id else "default"
                model_max_context = model_limits.get(model_name, model_limits["default"])
            
            # Verwende max_length als gewünschte Ausgabelänge (max_new_tokens)
            desired_new_tokens = max_length
            if desired_new_tokens >= model_max_context:
                desired_new_tokens = max(1, model_max_context - 1)
                logger.warning(f"[Stream] max_length zu hoch für Modell-Kontext. Setze gewünschte Ausgabelänge auf {desired_new_tokens} Tokens (Modell-Limit: {model_max_context}).")
            
            max_input_tokens = model_max_context - desired_new_tokens
            
            # Berechne max_new_tokens korrekt (max_length = gewünschte Ausgabelänge)
            max_new_tokens = min(
                desired_new_tokens,
                model_max_context - input_length
            )
            
            # Validierung: Prüfe ob max_new_tokens zu klein ist (BEVOR wir es auf 1 setzen)
            if max_new_tokens <= 0:
                raise ValueError(f"Input ist zu lang ({input_length} Tokens). Maximal erlaubte Eingabe: {max_input_tokens}, Modell-Limit: {model_max_context}")
            
            # Stelle sicher, dass max_new_tokens mindestens 1 ist (nach Validierung)
            max_new_tokens = max(1, max_new_tokens)

            # Auto-Cap für kurze Anfragen (Stream) – leichte Coding-Heuristik
            try:
                import re
                last_user = ""
                for m in reversed(messages or []):
                    if m.get("role") == "user":
                        last_user = str(m.get("content") or "")
                        break
                is_coding_hint = bool(re.search(r"\b(code|python|javascript|typescript|java|c\\+\\+|rust|debug|funktion|klasse|implementier)\b", last_user.lower())) or "```" in last_user
            except Exception:
                is_coding_hint = False
            max_new_tokens = self._auto_cap_max_new_tokens(messages, max_new_tokens, is_coding=is_coding_hint)
            
            # Logging für Debugging
            logger.info(f"Stream - Input-Länge: {input_length}, gewünschte Ausgabelänge: {desired_new_tokens}, max_new_tokens: {max_new_tokens}, Modell: {self.current_model_id}")
            
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
            # Dynamisches max_time: so wählen, dass max_new_tokens normalerweise VOR max_time erreicht wird.
            # Hintergrund: Qwen emittiert auf manchen Setups selten EOS -> min(30s) führte zu häufigen 30s Runs.
            max_time_seconds = max(12.0, min(180.0, max_new_tokens * 0.25))
            # Sampling ist einer der Hauptgründe für "kein EOS" / runaway bei kleinen Temperaturen.
            # Für typische QA/Chat (temp <= 0.3) verwenden wir greedy decoding.
            do_sample = bool(temperature and float(temperature) >= 0.5)
            generation_kwargs: Dict[str, Any] = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                max_time=max_time_seconds,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id_for_stream,
                no_repeat_ngram_size=3,
                streamer=streamer,
            )
            if do_sample:
                generation_kwargs.update(
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50 if is_mistral else None,
                )
            stopping_criteria = self._build_stopping_criteria()
            if stopping_criteria is not None:
                generation_kwargs["stopping_criteria"] = stopping_criteria
            
            generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            generation_thread.start()
            
            # Yield Chunks vom Streamer
            stream_chunk_count = 0
            stream_start_time = time.time()
            for text in streamer:
                if text:
                    stream_chunk_count += 1
                    # Bereinige Chunk
                    # WICHTIG: Kein strip() — sonst verschwinden Leerzeichen/Zeilenumbrüche zwischen Chunks.
                    cleaned = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
                    cleaned = self._normalize_mixed_script_homoglyphs(cleaned)
                    if cleaned != "":
                        # #region agent log
                        try:
                            with open(r'g:\04-CODING\Local Ai\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C3","location":"model_manager.py:1963","message":"Streamer chunk yielded","data":{"chunk_num":stream_chunk_count,"chunk_length":len(cleaned),"time_since_start":time.time()-stream_start_time},"timestamp":int(time.time()*1000)})+"\n")
                        except: pass
                        # #endregion
                        yield cleaned
            
            generation_thread.join()
            
        except Exception as e:
            logger.error(f"Fehler bei Streaming-Generierung: {e}")
            raise

