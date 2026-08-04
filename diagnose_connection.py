"""Diagnose Bitchat VLM API connection issue.

The app shows 127.0.0.1:8765 (localhost on Android).
We need to connect via WiFi IP.
"""
import socket
import requests
import time

print("="*60)
print("BITCHAT VLM API CONNECTION DIAGNOSIS")
print("="*60)

# Test different possible IPs
test_ips = [
    ("127.0.0.1", "Local PC (not Android)"),
    ("10.108.31.151", "Android WiFi IP"),
    ("192.168.1.1", "Common router gateway"),
]

print("\nTesting ports on 10.108.31.151:")
print("-" * 60)

for port in [8765, 8080, 80, 443]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    result = s.connect_ex(('10.108.31.151', port))
    status = "OPEN" if result == 0 else "CLOSED"
    print(f"Port {port:5d}: {status}")
    s.close()

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("""
The Bitchat app shows: https://127.0.0.1:8765

This is the LOCALHOST address on the Android device itself.
To connect from your PC, you need:

SOLUTION 1: Use WiFi IP (RECOMMENDED)
  1. In Bitchat app, check Settings -> VLM API
  2. Look for "WiFi IP" or "Network IP" field
  3. It should show something like: 10.108.31.151
  4. Use that IP to connect

SOLUTION 2: ADB Port Forwarding
  1. Connect Android via USB
  2. Enable USB debugging on Android
  3. Run: adb forward tcp:8765 tcp:8765
  4. Then use: 127.0.0.1:8765

SOLUTION 3: Check Bitchat settings
  1. Open Bitchat app
  2. Go to Settings -> VLM API Settings
  3. Check if there's a "Bind Address" or "Network Access" setting
  4. Make sure it's set to WiFi IP, not just localhost
""")

print("\nChecking if device supports WiFi access...")
print("Attempt to ping the WiFi IP:")
import subprocess
result = subprocess.run(['ping', '-n', '1', '10.108.31.151'], 
                       capture_output=True, timeout=2)
if result.returncode == 0:
    print("OK - Device is reachable on WiFi network")
else:
    print("FAIL - Device not reachable - check WiFi connection")

print("\n" + "="*60)
