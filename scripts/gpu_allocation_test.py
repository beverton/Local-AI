import time
import sys
import requests

MODEL_SERVICE = "http://127.0.0.1:8001"
MAIN_SERVICE = "http://127.0.0.1:8000"


def wait_for(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def wait_for_model_loaded(url, model_key, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        status = requests.get(url, timeout=5).json()
        model_status = status.get(model_key, {})
        if model_status.get("loaded") and not model_status.get("loading"):
            return True, status
        if model_status.get("error"):
            return False, status
        time.sleep(5)
    return False, requests.get(url, timeout=5).json()


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "qwen-2.5-7b-instruct"
    print("Waiting for model service...")
    if not wait_for(f"{MODEL_SERVICE}/status", timeout=120):
        raise SystemExit("Model service not reachable")

    print("Waiting for main service...")
    if not wait_for(f"{MAIN_SERVICE}/status", timeout=120):
        raise SystemExit("Main service not reachable")

    print(f"Loading text model ({model_id})...")
    resp = requests.post(f"{MODEL_SERVICE}/models/text/load", json={"model_id": model_id})
    print("Text load:", resp.status_code, resp.text[:200])

    print("Loading audio model (whisper-large-v3)...")
    resp = requests.post(f"{MODEL_SERVICE}/models/audio/load", json={"model_id": "whisper-large-v3"})
    print("Audio load:", resp.status_code, resp.text[:200])

    print("Waiting for text model to finish loading...")
    text_ok, status = wait_for_model_loaded(f"{MODEL_SERVICE}/status", "text_model", timeout=300)
    print("Text model loaded:", text_ok)
    if not text_ok:
        print("Text model error:", status.get("text_model", {}).get("error"))
    
    print("Waiting for audio model to finish loading...")
    audio_ok, status = wait_for_model_loaded(f"{MODEL_SERVICE}/status", "audio_model", timeout=180)
    print("Audio model loaded:", audio_ok)
    if not audio_ok:
        print("Audio model error:", status.get("audio_model", {}).get("error"))

    status = requests.get(f"{MODEL_SERVICE}/status", timeout=5).json()
    print("Model service status:")
    print(status)

    stats = requests.get(f"{MAIN_SERVICE}/system/stats").json()
    print("System stats:")
    print(stats)

    print("GPU utilization:", stats.get("gpu_utilization"))
    print("GPU memory percent:", stats.get("gpu_memory_percent"))


if __name__ == "__main__":
    main()
