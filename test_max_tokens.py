"""
Test: Unterschied zwischen max_new_tokens=50 vs 839
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_path = r"G:\KI Modelle\coding\qwen-2.5-7b-instruct"

print("Lade Modell...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

messages = [{"role": "user", "content": "test"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

if hasattr(model, 'hf_device_map') and model.hf_device_map:
    first_device = list(model.hf_device_map.values())[0]
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

# Teste verschiedene max_new_tokens
tests = [50, 100, 200, 400, 839]

for max_tok in tests:
    print(f"\n{'='*60}")
    print(f"TEST: max_new_tokens={max_tok}, no_repeat_ngram_size=3")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tok,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        elapsed = time.time() - start
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"✅ SUCCESS nach {elapsed:.2f}s")
        print(f"Generated tokens: {outputs[0].shape[0] - inputs['input_ids'].shape[1]}")
        print(f"Antwort (erste 100 chars): {response[:100]}")
        
        # Timeout nach 60s
        if elapsed > 60:
            print("⚠️ Zu langsam (>60s), stoppe weitere Tests")
            break
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FEHLER nach {elapsed:.2f}s: {e}")
        break

print("\n" + "="*60)
print("TESTS ABGESCHLOSSEN")
print("="*60)


