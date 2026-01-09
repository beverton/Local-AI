"""
Test-Script: Beste Lade-Methode für Qwen finden
Testet verschiedene Konfigurationen und dokumentiert die Ergebnisse
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import time
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setze PyTorch CUDA Allocator Config
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Qwen Modell-Pfad
QWEN_MODEL_PATH = r"G:\KI Modelle\coding\qwen-2.5-7b-instruct"

# Test-Prompt
TEST_PROMPT = "Was ist 2+3? Antworte kurz."


class LoadingTestResult:
    """Speichert Testergebnisse"""
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.load_time = 0.0
        self.load_success = False
        self.error_message = None
        self.gpu_memory_used_gb = 0.0
        self.gpu_memory_total_gb = 0.0
        self.inference_time = 0.0
        self.inference_success = False
        self.response = None
        self.config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "load_time_seconds": round(self.load_time, 2),
            "load_success": self.load_success,
            "error": self.error_message,
            "gpu_memory_used_gb": round(self.gpu_memory_used_gb, 2),
            "gpu_memory_total_gb": round(self.gpu_memory_total_gb, 2),
            "inference_time_seconds": round(self.inference_time, 2),
            "inference_success": self.inference_success,
            "response_length": len(self.response) if self.response else 0,
            "config": self.config
        }


def get_gpu_memory_info() -> tuple:
    """Gibt GPU-Speicher-Info zurück"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        return allocated, reserved, total
    return 0.0, 0.0, 0.0


def clear_gpu_memory():
    """Räumt GPU-Speicher auf"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_loading_method(method_name: str, config: Dict[str, Any]) -> LoadingTestResult:
    """Testet eine spezifische Lade-Methode"""
    result = LoadingTestResult(method_name)
    result.config = config.copy()
    
    print(f"\n{'='*70}")
    print(f"TEST: {method_name}")
    print(f"{'='*70}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    model = None
    tokenizer = None
    
    try:
        # Speicher vor Laden
        clear_gpu_memory()
        time.sleep(1)  # Warte kurz auf Cache-Clear
        mem_before = get_gpu_memory_info()[1]  # Reserved memory
        
        # Tokenizer laden
        print("Lade Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_PATH,
            trust_remote_code=True
        )
        
        # Modell laden
        print("Lade Modell...")
        start_time = time.time()
        
        # Baue model_kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        
        # torch_dtype
        if config.get("torch_dtype") == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif config.get("torch_dtype") == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif config.get("torch_dtype") == "float32":
            model_kwargs["torch_dtype"] = torch.float32
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # device_map
        if config.get("device_map"):
            model_kwargs["device_map"] = config["device_map"]
            if config["device_map"] == "auto" and config.get("max_memory"):
                model_kwargs["max_memory"] = config["max_memory"]
        
        # Quantisierung
        if config.get("use_quantization"):
            try:
                from transformers import BitsAndBytesConfig
                bits = config.get("quantization_bits", 8)
                if bits == 8:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16
                    )
                elif bits == 4:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
            except ImportError:
                print("WARNUNG: bitsandbytes nicht verfügbar, überspringe Quantisierung")
        
        # Flash Attention
        if config.get("use_flash_attention"):
            try:
                from flash_attn import flash_attn_func
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Flash Attention 2 aktiviert")
            except ImportError:
                print("Flash Attention 2 nicht verfügbar")
        
        # Lade Modell
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_PATH,
            **model_kwargs
        )
        
        load_time = time.time() - start_time
        result.load_time = load_time
        result.load_success = True
        
        # Speicher nach Laden
        mem_after = get_gpu_memory_info()
        result.gpu_memory_used_gb = mem_after[1] - mem_before  # Reserved memory difference
        result.gpu_memory_total_gb = mem_after[2]
        
        print(f"✓ Modell geladen in {load_time:.2f}s")
        print(f"✓ GPU-Speicher: {result.gpu_memory_used_gb:.2f}GB verwendet (von {result.gpu_memory_total_gb:.2f}GB)")
        
        # Test-Inferenz
        print("Führe Test-Inferenz durch...")
        messages = [{"role": "user", "content": TEST_PROMPT}]
        
        # Chat-Template verwenden
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"User: {TEST_PROMPT}\nAssistant:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Inputs auf richtiges Device verschieben
        if hasattr(model, 'device'):
            target_device = model.device
        elif hasattr(model, 'hf_device_map') and model.hf_device_map:
            first_device = list(model.hf_device_map.values())[0]
            target_device = first_device
        else:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        
        # Generiere
        start_time = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.3,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start_time
        result.inference_time = inference_time
        
        # Decode
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        
        result.response = response
        result.inference_success = True
        
        print(f"✓ Inferenz in {inference_time:.2f}s")
        print(f"✓ Antwort: {response[:100]}...")
        
    except Exception as e:
        result.load_success = False
        result.error_message = str(e)
        print(f"✗ FEHLER: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        clear_gpu_memory()
        time.sleep(2)  # Warte auf Cleanup
    
    return result


def main():
    """Hauptfunktion - testet alle Methoden"""
    print("="*70)
    print("QWEN LADE-METHODEN TEST")
    print("="*70)
    print(f"Modell-Pfad: {QWEN_MODEL_PATH}")
    print(f"CUDA verfügbar: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU-Speicher: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB")
    print("="*70)
    
    # Definiere Test-Methoden
    test_methods = [
        {
            "name": "Method 1: device_map=cuda, float16 (AKTUELL)",
            "config": {
                "device_map": "cuda",
                "torch_dtype": "float16"
            }
        },
        {
            "name": "Method 2: device_map=auto, float16",
            "config": {
                "device_map": "auto",
                "torch_dtype": "float16"
            }
        },
        {
            "name": "Method 3: device_map=cuda, bfloat16",
            "config": {
                "device_map": "cuda",
                "torch_dtype": "bfloat16"
            }
        },
        {
            "name": "Method 4: device_map=auto, bfloat16",
            "config": {
                "device_map": "auto",
                "torch_dtype": "bfloat16"
            }
        },
        {
            "name": "Method 5: device_map=cuda, float16, 8-bit Quantisierung",
            "config": {
                "device_map": "auto",  # Quantisierung benötigt "auto"
                "torch_dtype": "float16",
                "use_quantization": True,
                "quantization_bits": 8
            }
        },
        {
            "name": "Method 6: device_map=cuda, float16, Flash Attention",
            "config": {
                "device_map": "cuda",
                "torch_dtype": "float16",
                "use_flash_attention": True
            }
        }
    ]
    
    results = []
    
    # Teste jede Methode
    for method in test_methods:
        try:
            result = test_loading_method(method["name"], method["config"])
            results.append(result)
        except KeyboardInterrupt:
            print("\n\nTest abgebrochen durch Benutzer")
            break
        except Exception as e:
            print(f"\n✗ Unerwarteter Fehler bei {method['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Zusammenfassung
    print("\n\n" + "="*70)
    print("ZUSAMMENFASSUNG")
    print("="*70)
    
    successful_results = [r for r in results if r.load_success and r.inference_success]
    
    if successful_results:
        print(f"\nErfolgreiche Tests: {len(successful_results)}/{len(results)}")
        print("\nVergleich:")
        print(f"{'Methode':<50} {'Ladezeit':<12} {'Inferenz':<12} {'GPU-Speicher':<15}")
        print("-"*70)
        
        for result in successful_results:
            print(f"{result.method_name[:48]:<50} {result.load_time:>8.2f}s   {result.inference_time:>8.2f}s   {result.gpu_memory_used_gb:>10.2f}GB")
        
        # Beste Methode finden
        # Gewichtung: Ladezeit 30%, Inferenzzeit 30%, Speicher 40%
        best_score = float('inf')
        best_result = None
        
        for result in successful_results:
            # Normalisiere Werte (kleiner = besser)
            # Annahme: Maximalwerte für Normalisierung
            max_load_time = max(r.load_time for r in successful_results)
            max_inference_time = max(r.inference_time for r in successful_results)
            max_memory = max(r.gpu_memory_used_gb for r in successful_results)
            
            normalized_load = (result.load_time / max_load_time) * 0.3
            normalized_inference = (result.inference_time / max_inference_time) * 0.3
            normalized_memory = (result.gpu_memory_used_gb / max_memory) * 0.4
            
            score = normalized_load + normalized_inference + normalized_memory
            
            if score < best_score:
                best_score = score
                best_result = result
        
        print(f"\n{'='*70}")
        print("BESTE METHODE:")
        print(f"{'='*70}")
        print(f"Methode: {best_result.method_name}")
        print(f"Ladezeit: {best_result.load_time:.2f}s")
        print(f"Inferenzzeit: {best_result.inference_time:.2f}s")
        print(f"GPU-Speicher: {best_result.gpu_memory_used_gb:.2f}GB")
        print(f"Config: {json.dumps(best_result.config, indent=2)}")
    else:
        print("\n✗ Keine erfolgreichen Tests!")
    
    # Speichere Ergebnisse
    results_file = "qwen_loading_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
    
    print(f"\nErgebnisse gespeichert in: {results_file}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✓ Test abgeschlossen!")
    except Exception as e:
        print(f"\n✗ Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
