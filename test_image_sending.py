"""Test image sending with Bitchat API.

Quick test to verify images are encoded and sent correctly.
"""
import cv2
import numpy as np
import time
from core.bitchat import BitchatAlertClient

def create_test_frame():
    """Create a test frame with visible markers."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Add colored rectangles
    cv2.rectangle(frame, (50, 50), (300, 300), (0, 255, 0), -1)
    cv2.rectangle(frame, (350, 50), (600, 300), (255, 0, 0), -1)
    cv2.rectangle(frame, (650, 50), (900, 300), (0, 0, 255), -1)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"TEST IMAGE - {timestamp}", (200, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    return frame

def test_bitchat_image_sending(ip="10.99.220.148", port=8765):
    """Test Bitchat image sending with detailed logging."""
    print("="*60)
    print("BITCHAT IMAGE SENDING TEST")
    print("="*60)
    
    # Create client
    print(f"\n1. Creating BitchatAlertClient...")
    print(f"   URL: http://{ip}:{port}")
    
    client = BitchatAlertClient(
        ip=ip,
        port=port,
        channel="surveillance",
        rate_limit_s=2.0
    )
    
    # Check connection
    print(f"\n2. Checking connection...")
    if not client.check_connection():
        print("   ✗ Connection failed - aborting test")
        return False
    
    # Create test frame
    print(f"\n3. Creating test image...")
    frame = create_test_frame()
    print(f"   Frame shape: {frame.shape}")
    print(f"   Frame dtype: {frame.dtype}")
    
    # Test 1: Send scene with image
    print(f"\n4. Test 1: Sending scene with image...")
    client.send_scene("Test scene: AI surveillance system active", frame)
    print(f"   ✓ Scene queued")
    
    # Wait for rate limit
    print(f"\n5. Waiting for rate limit (2.5s)...")
    time.sleep(2.5)
    
    # Test 2: Send alert with image
    print(f"\n6. Test 2: Sending alert with image...")
    client.send_alert(
        event_type="TEST",
        detail="Test alert with image attachment",
        frame=frame,
        priority=True
    )
    print(f"   ✓ Alert queued")
    
    # Test 3: Send alert without image
    print(f"\n7. Waiting 3s for processing...")
    time.sleep(3)
    
    print(f"\n8. Test 3: Sending alert without image...")
    client.send_alert(
        event_type="TEST",
        detail="Test alert without image",
        frame=None,
        priority=False
    )
    print(f"   ✓ Alert queued")
    
    # Wait for completion
    print(f"\n9. Waiting 5s for all messages to send...")
    time.sleep(5)
    
    print(f"\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nCheck your Bitchat app for 3 messages:")
    print("  1. Scene message with green/blue/red rectangles")
    print("  2. TEST alert with same rectangles")
    print("  3. TEST alert without image")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Bitchat image sending")
    parser.add_argument("--ip", type=str, default="10.99.220.148",
                        help="Bitchat device IP")
    parser.add_argument("--port", type=int, default=8765,
                        help="Bitchat API port")
    
    args = parser.parse_args()
    
    try:
        test_bitchat_image_sending(args.ip, args.port)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
