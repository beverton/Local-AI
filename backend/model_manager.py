"""
Model Manager - Verwaltet das Laden und Wechseln von AI-Modellen
"""
import json
import os
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
    
    def _load_config(self) -> Dict[str, Any]:
        """Lädt die Konfiguration aus config.json"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config-Datei nicht gefunden: {self.config_path}")
            return {}
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Gibt alle verfügbaren Modelle zurück"""
        return self.config.get("models", {})
    
    def get_current_model(self) -> Optional[str]:
        """Gibt die ID des aktuell geladenen Modells zurück"""
        return self.current_model_id
    
    def is_model_loaded(self) -> bool:
        """Prüft ob ein Modell geladen ist"""
        return self.model is not None and self.tokenizer is not None
    
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
            
            # Tokenizer laden
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Modell laden
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.current_model_id = model_id
            logger.info(f"Modell erfolgreich geladen: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            self.model = None
            self.tokenizer = None
            return False
    
    def generate(self, messages: List[Dict[str, str]], max_length: int = 512, temperature: float = 0.3) -> str:
        """
        Generiert eine Antwort basierend auf Messages (Chat-Format)
        
        Args:
            messages: Liste von Messages im Format [{"role": "user", "content": "..."}, ...]
            max_length: Maximale Länge der Antwort
            temperature: Kreativität (0.0 = deterministisch, 1.0 = kreativ) - niedriger = konsistenter
            
        Returns:
            Die generierte Antwort
        """
        if not self.is_model_loaded():
            raise RuntimeError("Kein Modell geladen!")
        
        try:
            # Verwende Chat-Template wenn verfügbar (für Qwen, Phi-3, etc.)
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
                original_prompt = prompt
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate mit besseren Parametern
            with torch.no_grad():
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
                
                # Für Mistral: Kürzere Antworten, stärkere Kontrolle
                max_tokens = min(max_length, 200 if is_mistral else 256)
                repetition_penalty = 1.3 if is_mistral else 1.2
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
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
                
                # Sicherheitscheck: Falls Response noch den Prompt enthält, entferne ihn
                # Prüfe verschiedene Teile des Prompts
                prompt_start = original_prompt[:50] if len(original_prompt) > 50 else original_prompt
                if prompt_start in response:
                    logger.warning("Response enthält noch Prompt-Teil, entferne ihn...")
                    response = response.replace(prompt_start, "").strip()
                
                # Entferne auch spezifische System-Prompt-Phrasen
                system_phrases = [
                    "Du bist ein hilfreicher",
                    "AI-Assistent",
                    "Antworte klar und direkt auf Deutsch"
                ]
                for phrase in system_phrases:
                    if phrase in response:
                        # Finde die Zeile und entferne sie
                        lines = response.split('\n')
                        response = '\n'.join([l for l in lines if phrase not in l]).strip()
            
            # Entferne mögliche Chat-Template-Markierungen
            response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
            
            # ENTFERNE KOMPLETTEN PROMPT - EINFACHE METHODE: Finde "assistant" und nimm nur den Inhalt danach
            # Das ist die zuverlässigste Methode, da Qwen manchmal den kompletten Prompt generiert
            
            response_lower = response.lower()
            
            # Suche nach "assistant" in verschiedenen Formaten
            assistant_markers = ["assistant ", "assistant:", "assistant\n"]
            assistant_pos = -1
            
            for marker in assistant_markers:
                pos = response_lower.find(marker)
                if pos != -1:
                    assistant_pos = pos + len(marker)
                    break
            
            # Wenn "assistant" gefunden wurde, nimm nur den Teil danach
            if assistant_pos > 0:
                response = response[assistant_pos:].strip()
                logger.info("Response nach 'assistant' Marker extrahiert")
            else:
                # Fallback: Entferne System/User-Markierungen manuell
                lines = response.split('\n')
                cleaned = []
                found_assistant = False
                for line in lines:
                    line_lower = line.strip().lower()
                    if line_lower.startswith('assistant'):
                        found_assistant = True
                        # Nimm nur den Teil nach "assistant"
                        content = line.split(':', 1)[-1].strip() if ':' in line else line[9:].strip()
                        if content:
                            cleaned.append(content)
                        continue
                    if found_assistant:
                        # Nach "assistant" alle Zeilen hinzufügen, außer weitere Markierungen
                        if not (line_lower.startswith('system ') or line_lower.startswith('user ')):
                            cleaned.append(line)
                    elif not found_assistant:
                        # Vor "assistant": Überspringe System/User-Zeilen
                        if not (line_lower.startswith('system ') or 
                                line_lower.startswith('user ') or
                                "du bist ein hilfreicher" in line_lower):
                            cleaned.append(line)
                response = '\n'.join(cleaned).strip()
            
            # Finale Bereinigung: Entferne System-Prompt-Phrasen falls noch vorhanden
            system_keywords = ["du bist ein hilfreicher", "ai-assistent", "antworte klar und direkt"]
            for keyword in system_keywords:
                if keyword in response.lower():
                    lines = response.split('\n')
                    response = '\n'.join([l for l in lines if keyword.lower() not in l.lower()]).strip()
            
            # Entferne User-Nachricht falls noch vorhanden
            if messages:
                last_user = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
                if last_user and last_user["content"] in response:
                    response = response.replace(last_user["content"], "").strip()
            
            # WICHTIG: Entferne mehrfache Nachrichten (falls Modell mehrere generiert)
            # Suche nach "User:" oder "Assistant:" in der Antwort und schneide dort ab
            lines = response.split('\n')
            cleaned_lines = []
            found_stop = False
            for line in lines:
                line_stripped = line.strip()
                
                # Stoppe bei weiteren User/Assistant/System-Markierungen
                if line_stripped.startswith('User:') or line_stripped.startswith('Assistant:') or line_stripped.startswith('System:'):
                    found_stop = True
                    break
                
                # Stoppe bei Markdown-Formatierungen die wie neue Fragen aussehen
                if line_stripped.startswith('# ') and len(line_stripped) > 10:
                    if '?' in line_stripped or 'Wie' in line_stripped or 'Was' in line_stripped:
                        found_stop = True
                        break
                
                # Stoppe wenn wir "system" oder "user" in der Zeile sehen (kleingeschrieben)
                if 'system' in line_stripped.lower() and ('Du bist' in line_stripped or 'AI-Assistent' in line_stripped):
                    found_stop = True
                    break
                
                cleaned_lines.append(line)
            
            response = '\n'.join(cleaned_lines).strip()
            
            # Entferne führende "Assistant:" oder "assistant" falls vorhanden
            if response.lower().startswith('assistant:'):
                response = response[10:].strip()
            elif response.lower().startswith('assistant '):
                response = response[9:].strip()
            
            # Entferne leere Zeilen am Anfang/Ende
            response = response.strip()
            
            # Wenn die Antwort sehr lang ist und mehrere Abschnitte hat, nimm nur den ersten
            # (verhindert Halluzinationen von mehreren Antworten)
            if len(response) > 500 and '\n\n' in response:
                sections = response.split('\n\n')
                if len(sections) > 1:
                    first_section = sections[0]
                    second_section = sections[1][:100] if len(sections[1]) > 100 else sections[1]
                    # Wenn der zweite Abschnitt mit "#", "Wie", "System", "User" beginnt, ist es eine neue Nachricht
                    second_lower = second_section.strip().lower()
                    if (second_lower.startswith('#') or 
                        second_lower.startswith('wie') or 
                        second_lower.startswith('system') or 
                        second_lower.startswith('user')):
                        response = first_section.strip()
            
            # Finale Bereinigung: Entferne alle Zeilen die "system", "user" oder "assistant" als erstes Wort haben
            final_lines = []
            skip_mode = False
            for line in response.split('\n'):
                line_stripped = line.strip()
                line_lower = line_stripped.lower()
                
                # Stoppe bei System/User/Assistant-Markierungen
                if (line_lower.startswith('system ') or 
                    line_lower.startswith('user ') or 
                    (line_lower.startswith('assistant ') and len(final_lines) > 0)):
                    break
                
                # Prüfe ob Zeile System-Prompt-Text enthält
                if ("Du bist ein" in line_stripped and "AI-Assistent" in line_stripped) or "Antworte klar und direkt" in line_stripped:
                    skip_mode = True
                    continue
                
                # Wenn wir im Skip-Modus sind, überspringe bis zur nächsten normalen Zeile
                if skip_mode:
                    if line_lower.startswith('user ') or line_lower.startswith('assistant '):
                        skip_mode = False
                        # Wenn es "assistant" ist, nimm diese Zeile
                        if line_lower.startswith('assistant '):
                            final_lines.append(line_stripped.replace('assistant ', '', 1).replace('Assistant ', '', 1).strip())
                    continue
                
                final_lines.append(line)
            
            response = '\n'.join(final_lines).strip()
            
            # Mistral-spezifische Bereinigung: Entferne technische Phrasen und lange Listen
            is_mistral = "mistral" in self.current_model_id.lower() if self.current_model_id else False
            if is_mistral:
                # Entferne häufige Mistral-Phrasen die nicht zur Antwort gehören
                mistral_phrases = [
                    "Um Ihr System angemessen zu beschreiben",
                    "benötige ich weitere Informationen",
                    "Hier sind einige wichtige Faktoren",
                    "Mit dieser Information kannst du",
                    "Alternativ kannst Du auch Tools wie",
                    "TensorFlow Profiler",
                    "PyTorch Profiler",
                    "TensorBoard",
                    "PytorchProfiler",
                    "maximALE Länge",
                    "maximALLE generation length",
                    "Berührungsgang der GPU",
                    "CPU-Leiste",
                    "Clock Speed",
                    "Memory Size",
                    "Bandwidth",
                    "Trainingdatengrößen",
                    "================================",
                ]
                
                for phrase in mistral_phrases:
                    if phrase in response:
                        # Finde die Zeile und entferne sie und alles danach
                        lines = response.split('\n')
                        cleaned = []
                        found_phrase = False
                        for line in lines:
                            if phrase in line:
                                found_phrase = True
                                break
                            if not found_phrase:
                                cleaned.append(line)
                        response = '\n'.join(cleaned).strip()
                        break
                
                # Entferne Zeilen mit vielen Sonderzeichen (Formatierungsfehler)
                lines = response.split('\n')
                cleaned = []
                for line in lines:
                    # Überspringe Zeilen mit vielen Sonderzeichen oder Formatierungsfehlern
                    special_chars = sum(1 for c in line if c in '<>[]{}|*+=_-')
                    if special_chars > len(line) * 0.3:  # Mehr als 30% Sonderzeichen
                        continue
                    # Überspringe sehr lange Zeilen ohne Punkt (wahrscheinlich Formatierungsfehler)
                    if len(line) > 200 and '.' not in line and '?' not in line:
                        continue
                    cleaned.append(line)
                response = '\n'.join(cleaned).strip()
                
                # Begrenze Antwort auf maximal 500 Zeichen für Mistral (falls zu lang)
                if len(response) > 500:
                    # Versuche bei einem Satzende abzuschneiden
                    sentences = response.split('. ')
                    if len(sentences) > 1:
                        # Nimm die ersten Sätze bis 500 Zeichen
                        truncated = []
                        total_len = 0
                        for sentence in sentences:
                            if total_len + len(sentence) + 2 > 500:
                                break
                            truncated.append(sentence)
                            total_len += len(sentence) + 2
                        response = '. '.join(truncated) + '.'
                    else:
                        # Fallback: Einfach abschneiden
                        response = response[:500].rsplit(' ', 1)[0] + '...'
            
            # Entferne führende "assistant" falls noch vorhanden
            if response.lower().startswith('assistant'):
                response = response.split(':', 1)[-1].strip() if ':' in response else response[9:].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Fehler bei der Generierung: {e}")
            raise

