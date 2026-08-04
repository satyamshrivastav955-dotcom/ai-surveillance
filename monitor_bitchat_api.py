"""Monitor Bitchat VLM API connection and run test when ready.

Run this script, then enable VLM API on Bitchat.
It will automatically detect when the API becomes available and run tests.
"""
import time
import requests
import sys

IP = "10.108.31.151"
PORT = 8765
URL = f"http://{IP}:{PORT}/status"

print(f"Monitoring Bitchat VLM API at {URL}")
print("Enable VLM API in Bitchat Settings to start testing...")
print("Press Ctrl+C to stop\n")

attempt = 0
while True:
    attempt += 1
    try:
        r = requests.get(URL, timeout=2)
        if r.status_code == 200:
            data = r.json()
            print(f"\n{'='*60}")
            print("SUCCESS! VLM API is now available!")
            print(f"{'='*60}")
            print(f"API Enabled: {data.get('api_enabled')}")
            print(f"Mesh Running: {data.get('mesh_running')}")
            print(f"Peers Count: {data.get('peers_count')}")
            print(f"HTTP Port: {data.get('http_port')}")
            print(f"\n{'='*60}")
            print("Running full integration test...")
            print(f"{'='*60}\n")
            
            # Import and run test
            from test_bitchat_fix import test_client
            test_client(IP, PORT)
            break
    except:
        pass
    
    # Progress indicator
    sys.stdout.write(f"\rAttempt {attempt} - Port 8765: CLOSED  ")
    sys.stdout.flush()
    time.sleep(2)
