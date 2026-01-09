"""
Minimaler Test um herauszufinden welcher Parameter model.generate() zum Hängen bringt
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

model_path = r"G:\KI Modelle\coding\qwen-2.5-7b-instruct"

print("=" * 80)
print("MINIMAL GENERATE TEST - Finde problematischen Parameter")
print("=" * 80)

print("\n[1/5] Lade Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("[2/5] Lade Modell...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("[3/5] Erstelle Test-Prompt...")
messages = [{"role": "user", "content": "2+2"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

# Move to correct device
if hasattr(model, 'hf_device_map') and model.hf_device_map:
    first_device = list(model.hf_device_map.values())[0]
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

print(f"[4/5] Input-Länge: {inputs['input_ids'].shape[1]} tokens")

# Teste verschiedene Konfigurationen
tests = [
    {
        "name": "TEST 1: Absolut minimal (nur max_new_tokens)",
        "params": {
            "max_new_tokens": 50
        }
    },
    {
        "name": "TEST 2: Mit pad_token",
        "params": {
            "max_new_tokens": 50,
            "pad_token_id": tokenizer.eos_token_id
        }
    },
    {
        "name": "TEST 3: Mit eos_token (single)",
        "params": {
            "max_new_tokens": 50,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    },
    {
        "name": "TEST 4: Mit do_sample=False (greedy)",
        "params": {
            "max_new_tokens": 50,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": False
        }
    },
    {
        "name": "TEST 5: Mit temperature (wie aktuell)",
        "params": {
            "max_new_tokens": 50,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        }
    },
    {
        "name": "TEST 6: Mit repetition_penalty",
        "params": {
            "max_new_tokens": 50,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }
    },
    {
        "name": "TEST 7: Mit no_repeat_ngram_size (VERDÄCHTIG)",
        "params": {
            "max_new_tokens": 50,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3
        }
    }
]

print("\n[5/5] Starte Tests...")
print("=" * 80)

# Leere GPU Cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

for test in tests:
    print(f"\n{test['name']}")
    print(f"Parameter: {test['params']}")
    print("BEFORE model.generate()...")
    
    start = time.time()
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **test['params']
            )
        elapsed = time.time() - start
        print(f"✅ SUCCESS nach {elapsed:.2f}s")
        
        # Decode
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Antwort: {response[:100]}...")
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FEHLER nach {elapsed:.2f}s: {e}")
    
    # Timeout-Protection
    if time.time() - start > 30:
        print("⚠️ Test dauert zu lange (>30s), abgebrochen")
        break
    
    # Kurze Pause zwischen Tests
    time.sleep(2)

print("\n" + "=" * 80)
print("TESTS ABGESCHLOSSEN")
print("=" * 80)

