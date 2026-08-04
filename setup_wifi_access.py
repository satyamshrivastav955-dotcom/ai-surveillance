"""
SETUP GUIDE: Configure Bitchat to Listen on WiFi IP
=====================================================

The Bitchat app needs to bind to the WiFi IP address instead of just localhost.

STEP-BY-STEP INSTRUCTIONS:
==========================

1. Open Bitchat app on your Android device

2. Tap the menu icon (three horizontal lines) in top-left corner

3. Go to Settings

4. Scroll down to find "VLM API Settings"

5. Look for these settings:
   
   a) "Bind Address" or "Listen Address"
      - Change from: 127.0.0.1 (localhost)
      - Change to: 0.0.0.0 (all interfaces) OR 10.108.31.151 (WiFi IP)
   
   b) "Network Access" or "Allow WiFi Connections"
      - Enable this option if available
   
   c) "HTTP Port"
      - Should be: 8765 (default)
   
   d) If there's no "Bind Address" setting:
      - Look for a toggle: "Allow network connections" or "Listen on WiFi"
      - Enable it

6. Save/Apply settings

7. You may need to restart the VLM API service:
   - Toggle "Enable VLM API" OFF then ON again

8. Verify the change:
   - The app should now show: http://10.108.31.151:8765
   - Or it might show the WiFi IP in a "Network IP" field

9. Test from your PC:
   python -c "import requests; print(requests.get('http://10.108.31.151:8765/status').json())"


POSSIBLE SETTINGS LOCATIONS:
=============================
Depending on Bitchat version:

Option A: Settings -> VLM API Settings -> Bind Address
Option B: Settings -> VLM API Settings -> Network Access
Option C: Settings -> Network -> Allow external connections
Option D: Settings -> VLM API Settings -> Advanced -> Listen on all interfaces


IF NO SUCH SETTING EXISTS:
===========================
The app may only support localhost. In this case:

Alternative 1: Use ADB Port Forwarding
  adb forward tcp:8765 tcp:8765
  (Then use 127.0.0.1 in pipeline.yaml)

Alternative 2: Contact Bitchat developers
  Request network binding support


TROUBLESHOOTING:
================

Issue: Port 8765 still closed on WiFi IP
  - Check Android firewall apps (NetGuard, AFWall+, etc.)
  - Temporarily disable firewall to test
  - Make sure Bitchat has network permissions

Issue: Setting changes back to 127.0.0.1
  - This might be a security feature
  - Check if "Allow insecure network connections" needs to be enabled
  - Try restarting the app after changing settings

Issue: Connection fails from PC
  - Verify both devices on same WiFi network
  - Check antivirus/firewall on PC
  - Try: ping 10.108.31.151 from PC


AFTER SUCCESSFUL CONFIGURATION:
=================================
Run the test:
  python test_bitchat_fix.py 10.108.31.151 8765

Or run full test suite:
  python tests/bitchat_api_test.py --ip 10.108.31.151
"""

print(__doc__)

import requests
import time

print("\n" + "="*60)
print("TESTING CONNECTION...")
print("="*60)

ip = "10.108.31.151"
port = 8765
url = f"http://{ip}:{port}/status"

for attempt in range(3):
    try:
        print(f"\nAttempt {attempt + 1}/3: Testing {url}")
        r = requests.get(url, timeout=3)
        
        if r.status_code == 200:
            data = r.json()
            print("\n" + "="*60)
            print("SUCCESS! Bitchat VLM API is accessible!")
            print("="*60)
            print(f"API Enabled: {data.get('api_enabled')}")
            print(f"Mesh Running: {data.get('mesh_running')}")
            print(f"Peers Count: {data.get('peers_count')}")
            print(f"Port: {data.get('http_port')}")
            print(f"Silence Mode: {data.get('silence_enabled')}")
            
            print("\n" + "="*60)
            print("Running integration test...")
            print("="*60)
            import subprocess
            subprocess.run(['python', 'test_bitchat_fix.py', ip, str(port)])
            break
        else:
            print(f"HTTP {r.status_code}: {r.text[:200]}")
            
    except Exception as e:
        print(f"Connection failed: {str(e)[:100]}")
        if attempt < 2:
            print("Retrying in 2 seconds...")
            time.sleep(2)
    else:
        break
else:
    print("\n" + "="*60)
    print("CONNECTION FAILED")
    print("="*60)
    print("\nBitchat is not listening on WiFi IP yet.")
    print("Please follow the instructions above to configure it.")
    print("\nOnce configured, run: python test_bitchat_fix.py")
