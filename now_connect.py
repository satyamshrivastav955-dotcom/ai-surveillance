"""Instructions for Bitchat VLM API Connection

PROBLEM:
The Bitchat app shows: https://127.0.0.1:8765
This is localhost on the Android device.
Port 8765 is CLOSED on WiFi IP 10.108.31.151

SOLUTIONS:
==================================================

OPTION 1: Setup ADB Port Forward (RECOMMENDED)
--------------------------------------------------
This makes the Android localhost available on your PC.

1. Connect Android device via USB
2. Enable USB debugging on Android:
   - Settings -> Developer Options -> USB Debugging
3. Run this command:
   adb forward tcp:8765 tcp:8765
4. Test: python -c "import requests; print(requests.get('http://127.0.0.1:8765/status').json())"


OPTION 2: Enable WiFi Access in Bitchat
-----------------------------------------
The Bitchat API might be bound to localhost only.

1. Open Bitchat app
2. Go to Settings -> VLM API Settings
3. Look for:
   - "Bind Address" setting
   - "Network Access" setting
   - "WiFi IP" setting
4. Make sure it's set to 0.0.0.0 (all interfaces)
   or specifically to the WiFi IP: 10.108.31.151


OPTION 3: Check Android Firewall
----------------------------------
Port 8765 might be blocked by Android firewall.

1. Check if any firewall app is running (NetGuard, AFWall+, etc.)
2. Allow Bitchat app to accept incoming connections
3. Or disable firewall temporarily for testing


CURRENT STATUS:
- Device 10.108.31.151 is pingable (WiFi OK)
- Port 8765 is CLOSED on WiFi interface
- Port forwarding via ADB is NOT active

NEXT STEPS:
==================================================
1. Try OPTION 1 (ADB forwarding) - easiest and most reliable
2. Or try OPTION 2 (check Bitchat WiFi settings)

Run this after setting up either option:
  python test_bitchat_fix.py
"""

print(__doc__)

# Try to test if ADB works
import subprocess
try:
    result = subprocess.run(['adb', 'devices'], capture_output=True, timeout=2)
    if 'device' in result.stdout.decode() and 'List of devices' in result.stdout.decode():
        print("\nADB DETECTED - Setting up port forwarding...")
        subprocess.run(['adb', 'forward', 'tcp:8765', 'tcp:8765'], timeout=5)
        print("Port forwarding enabled!")
        print("\nTesting connection...")
        import requests
        r = requests.get('http://127.0.0.1:8765/status', timeout=3)
        if r.status_code == 200:
            print("SUCCESS! Connected to Bitchat VLM API")
            print(f"Status: {r.json()}")
except Exception as e:
    print(f"\nADB not available or device not connected: {e}")
    print("\nPlease use one of the options above manually.")
