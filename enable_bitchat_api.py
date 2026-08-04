"""
QUICK START - Enable Bitchat VLM API

Current Status:
- IP 10.108.31.151 is reachable (ping successful)
- Port 8765 is CLOSED (VLM API not enabled)

To Enable VLM API on Bitchat:
================================

1. Open Bitchat app on your Android device

2. Tap the menu icon (three lines) in top-left

3. Go to Settings

4. Scroll down to "VLM API Settings"

5. Toggle "Enable VLM API" to ON

6. Note the port number (default: 8765)

7. Verify WiFi IP matches: 10.108.31.151
   - Go to Android Settings -> WiFi
   - Tap your connected network
   - Check IP address

8. Once enabled, run:
   python test_bitchat_fix.py

The API should respond on http://10.108.31.151:8765/status
"""
print(__doc__)
