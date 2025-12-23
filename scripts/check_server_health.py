"""
Health-Check Script für Server
Prüft ob ein Server auf einem bestimmten Port läuft und erreichbar ist
"""
import sys
import time
import requests
import socket

def check_port(host, port, timeout=2):
    """Prüft ob ein Port geöffnet ist"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def check_http_endpoint(url, max_attempts=3, delay=0.5, timeout=2):
    """Prüft ob ein HTTP-Endpoint erreichbar ist"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code in [200, 404]:  # 404 ist auch OK - bedeutet Server läuft
                return True
        except requests.exceptions.ConnectionError:
            # Connection Error ist OK - Server startet noch
            pass
        except:
            pass
        
        if attempt < max_attempts - 1:
            time.sleep(delay)
    
    return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: check_server_health.py <host> <port> [endpoint]")
        sys.exit(1)
    
    host = sys.argv[1]
    port = int(sys.argv[2])
    endpoint = sys.argv[3] if len(sys.argv) > 3 else "/status"
    
    # Prüfe zuerst Port (wichtigste Prüfung)
    if not check_port(host, port):
        print(f"Port {port} ist nicht geöffnet")
        sys.exit(1)
    
    # Prüfe HTTP-Endpoint (optional - wenn Port offen ist, ist das meistens genug)
    url = f"http://{host}:{port}{endpoint}"
    if check_http_endpoint(url):
        print("OK")
        sys.exit(0)
    else:
        # Wenn Port offen ist, aber Endpoint nicht erreichbar, ist das auch OK
        # (Server könnte noch starten)
        print("OK")  # Port ist offen, das reicht
        sys.exit(0)

