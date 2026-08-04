"""Quick test to verify Bitchat client fixes work correctly.

Run this after enabling VLM API in Bitchat on your Android device.

Usage:
    python test_bitchat_fix.py
"""
import time
import numpy as np
import cv2
from core.bitchat import BitchatAlertClient


def create_test_frame(text: str = "TEST") -> np.ndarray:
    """Create a simple test image."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (590, 430), (0, 255, 0), 3)
    cv2.putText(frame, text, (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, time.strftime("%H:%M:%S"), (220, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return frame


def test_client(ip: str = "172.20.77.176", port: int = 8765):
    """Test the BitchatAlertClient with fixed methods."""
    
    print("="*60)
    print("BITCHAT CLIENT FIX VERIFICATION")
    print("="*60)
    print(f"Target: http://{ip}:{port}")
    print()
    
    # Create client
    client = BitchatAlertClient(ip=ip, port=port, channel="test")
    
    # Test 1: Check connection
    print("Test 1: Checking connection...")
    if client.check_connection():
        print("OK - Connected to Bitchat API")
    else:
        print("FAILED - Cannot connect to Bitchat API")
        print("\nTroubleshooting:")
        print("  1. Open Bitchat app on Android")
        print("  2. Go to Settings -> VLM API Settings")
        print("  3. Enable 'VLM API' toggle")
        print("  4. Verify the IP matches: Settings -> WiFi -> tap network")
        return False
    
    print()
    
    # Test 2: Send text-only scene
    print("Test 2: Sending text-only scene...")
    client.send_scene("Test scene: Camera operational, no activity detected")
    print("OK - Text scene queued")
    time.sleep(6)
    
    # Test 3: Send scene with image
    print("\nTest 3: Sending scene with image...")
    frame = create_test_frame("SCENE TEST")
    client.send_scene("Test scene: Green rectangle visible", frame)
    print("OK - Scene with image queued")
    time.sleep(6)
    
    # Test 4: Send alert without image
    print("\nTest 4: Sending alert without image...")
    client.send_alert("TEST", "This is a test alert without image")
    print("OK - Text alert queued")
    time.sleep(6)
    
    # Test 5: Send alert with image
    print("\nTest 5: Sending alert with image...")
    frame2 = create_test_frame("ALERT TEST")
    client.send_alert("TEST", "Test alert with image attachment", frame=frame2)
    print("OK - Alert with image queued")
    time.sleep(2)
    
    # Test 6: Send priority alert with image
    print("\nTest 6: Sending PRIORITY alert with image...")
    frame3 = create_test_frame("PRIORITY")
    client.send_alert("CRITICAL", "Priority alert bypasses rate limit", frame=frame3, priority=True)
    print("OK - Priority alert queued")
    
    print()
    print("="*60)
    print("ALL TESTS QUEUED SUCCESSFULLY")
    print("="*60)
    print("\nCheck Bitchat app to verify messages arrived.")
    print("You should see:")
    print("  - Text scene description")
    print("  - Scene with image")
    print("  - Text alert")
    print("  - Alert with image")
    print("  - Priority alert with image")
    
    return True


if __name__ == "__main__":
    import sys
    
    ip = sys.argv[1] if len(sys.argv) > 1 else "10.108.31.151"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    
    success = test_client(ip, port)
    sys.exit(0 if success else 1)
