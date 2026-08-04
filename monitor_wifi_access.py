"""
AUTO-DETECT: Monitor for Bitchat WiFi Access

This script continuously monitors for Bitchat VLM API on WiFi IP.
Run this script, then configure Bitchat to listen on WiFi.
It will automatically detect and run tests when ready.
"""
import requests
import time
import sys
import subprocess

IP = "10.108.31.151"
PORT = 8765
URL = f"http://{IP}:{PORT}/status"

print("="*60)
print("BITCHAT WIFI ACCESS MONITOR")
print("="*60)
print(f"\nMonitoring: {URL}")
print("Configure Bitchat to listen on this WiFi IP...")
print("The script will auto-detect and run tests when ready.\n")
print("Press Ctrl+C to stop\n")
print("="*60)

attempt = 0
check_interval = 2

try:
    while True:
        attempt += 1
        
        # Progress indicator
        sys.stdout.write(f"\rAttempt {attempt:4d} - Checking port {PORT}...")
        sys.stdout.flush()
        
        try:
            r = requests.get(URL, timeout=2)
            
            if r.status_code == 200:
                data = r.json()
                
                print(f"\n\n{'='*60}")
                print("SUCCESS! BITCHAT VLM API DETECTED!")
                print(f"{'='*60}")
                print(f"\nAPI Status:")
                print(f"  - API Enabled: {data.get('api_enabled')}")
                print(f"  - Mesh Running: {data.get('mesh_running')}")
                print(f"  - Peers Count: {data.get('peers_count')}")
                print(f"  - HTTP Port: {data.get('http_port')}")
                print(f"  - Silence Mode: {data.get('silence_enabled')}")
                
                print(f"\n{'='*60}")
                print("Running integration tests...")
                print(f"{'='*60}\n")
                
                # Run the test
                subprocess.run([
                    'python', 
                    'test_bitchat_fix.py', 
                    IP, 
                    str(PORT)
                ])
                
                print(f"\n{'='*60}")
                print("Test completed successfully!")
                print(f"{'='*60}")
                break
                
        except requests.exceptions.RequestException:
            pass
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            
        time.sleep(check_interval)

except KeyboardInterrupt:
    print(f"\n\nMonitoring stopped by user.")
    print(f"Attempted {attempt} connections over {attempt * check_interval:.0f} seconds")
    print(f"\nIf you're having trouble:")
    print(f"  1. Check Bitchat Settings -> VLM API Settings")
    print(f"  2. Look for 'Bind Address' or 'Network Access' setting")
    print(f"  3. Change to: 0.0.0.0 or 10.108.31.151")
    print(f"  4. Restart VLM API service")
    print(f"  5. Run this monitor again")
