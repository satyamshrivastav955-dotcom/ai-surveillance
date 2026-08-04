"""Comprehensive test for Bitchat VLM API integration.

Tests all endpoints according to API documentation:
- GET /status - Check connection
- POST /send/text - Send text messages
- POST /send/image - Send images with captions

Usage:
    python tests/bitchat_api_test.py --ip 172.20.77.176
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import cv2
import requests
from pathlib import Path


def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test image with colored rectangles."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(frame, (250, 50), (400, 200), (255, 0, 0), -1)
    cv2.rectangle(frame, (450, 50), (600, 200), (0, 0, 255), -1)
    
    cv2.putText(frame, "TEST IMAGE", (200, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(frame, time.strftime("%H:%M:%S"), (220, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    
    return frame


def test_status_endpoint(ip: str, port: int = 8765):
    """Test GET /status endpoint."""
    print("\n" + "="*60)
    print("TEST 1: GET /status")
    print("="*60)
    
    url = f"http://{ip}:{port}/status"
    print(f"URL: {url}")
    
    try:
        r = requests.get(url, timeout=5)
        print(f"Status Code: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            print(f"Response: {data}")
            
            assert data.get("status") == "ok", "Status not OK"
            assert data.get("api_enabled"), "API not enabled"
            
            print(f"✓ API is enabled")
            print(f"✓ HTTP Port: {data.get('http_port')}")
            print(f"✓ Mesh Running: {data.get('mesh_running')}")
            print(f"✓ Peers Count: {data.get('peers_count')}")
            print(f"✓ Silence Mode: {data.get('silence_enabled')}")
            
            return True
        else:
            print(f"✗ HTTP {r.status_code}: {r.text}")
            return False
            
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def test_send_text(ip: str, port: int = 8765):
    """Test POST /send/text endpoint."""
    print("\n" + "="*60)
    print("TEST 2: POST /send/text")
    print("="*60)
    
    url = f"http://{ip}:{port}/send/text"
    print(f"URL: {url}")
    
    payload = {
        "text": f"Test message from AI surveillance at {time.strftime('%H:%M:%S')}",
        "channel": "test"
    }
    
    print(f"Payload: {payload}")
    
    try:
        r = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")
        
        if r.status_code == 200:
            data = r.json()
            assert data.get("status") == "ok", "Response status not OK"
            print(f"✓ Text message sent successfully")
            print(f"✓ Message ID: {data.get('message_id')}")
            return True
        else:
            print(f"✗ HTTP {r.status_code}: {r.text}")
            return False
            
    except Exception as e:
        print(f"✗ Send text failed: {e}")
        return False


def test_send_image_with_caption(ip: str, port: int = 8765):
    """Test POST /send/image endpoint with image + caption."""
    print("\n" + "="*60)
    print("TEST 3: POST /send/image (with caption)")
    print("="*60)
    
    url = f"http://{ip}:{port}/send/image"
    print(f"URL: {url}")
    
    # Create test image
    frame = create_test_image()
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok, "Failed to encode test image"
    jpeg_bytes = buf.tobytes()
    
    print(f"Image size: {len(jpeg_bytes)} bytes")
    
    files = {"image": ("test_frame.jpg", jpeg_bytes, "image/jpeg")}
    data = {
        "caption": f"Test image with caption at {time.strftime('%H:%M:%S')}",
        "channel": "test"
    }
    
    print(f"Caption: {data['caption']}")
    
    try:
        r = requests.post(url, files=files, data=data, timeout=30)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")
        
        if r.status_code == 200:
            resp = r.json()
            assert resp.get("status") == "ok", "Response status not OK"
            print(f"✓ Image sent successfully")
            return True
        else:
            print(f"✗ HTTP {r.status_code}: {r.text}")
            return False
            
    except Exception as e:
        print(f"✗ Send image failed: {e}")
        return False


def test_send_image_no_caption(ip: str, port: int = 8765):
    """Test POST /send/image endpoint without caption."""
    print("\n" + "="*60)
    print("TEST 4: POST /send/image (no caption)")
    print("="*60)
    
    url = f"http://{ip}:{port}/send/image"
    print(f"URL: {url}")
    
    # Create test image
    frame = create_test_image(800, 600)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok, "Failed to encode test image"
    jpeg_bytes = buf.tobytes()
    
    print(f"Image size: {len(jpeg_bytes)} bytes")
    
    files = {"image": ("test_frame_no_caption.jpg", jpeg_bytes, "image/jpeg")}
    data = {"channel": "test"}
    
    try:
        r = requests.post(url, files=files, data=data, timeout=30)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")
        
        if r.status_code == 200:
            resp = r.json()
            assert resp.get("status") == "ok", "Response status not OK"
            print(f"✓ Image sent successfully (no caption)")
            return True
        else:
            print(f"✗ HTTP {r.status_code}: {r.text}")
            return False
            
    except Exception as e:
        print(f"✗ Send image failed: {e}")
        return False


def test_bitchat_client_integration(ip: str, port: int = 8765):
    """Test BitchatAlertClient class integration."""
    print("\n" + "="*60)
    print("TEST 5: BitchatAlertClient Integration")
    print("="*60)
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.bitchat import BitchatAlertClient
    
    client = BitchatAlertClient(ip=ip, port=port, channel="test")
    
    # Test connection
    print("\nTesting connection...")
    if not client.check_connection():
        print("✗ Connection check failed")
        return False
    
    print("✓ Connection successful")
    
    # Test send_text
    print("\nTesting send_scene (text only)...")
    client.send_scene("Test scene: No activity detected")
    time.sleep(6)
    
    # Test send_scene with image
    print("\nTesting send_scene (text + image)...")
    frame = create_test_image()
    client.send_scene("Test scene: Colored rectangles visible", frame)
    time.sleep(6)
    
    # Test send_alert without image
    print("\nTesting send_alert (text only)...")
    client.send_alert("TEST", "Alert without image", frame=None)
    time.sleep(6)
    
    # Test send_alert with image
    print("\nTesting send_alert (with image)...")
    frame2 = create_test_image(1280, 720)
    client.send_alert("TEST", "Alert with image attachment", frame=frame2, priority=True)
    time.sleep(2)
    
    print("\n✓ All BitchatAlertClient tests completed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test Bitchat VLM API endpoints"
    )
    parser.add_argument(
        "--ip",
        type=str,
        required=True,
        help="Bitchat device IP address (e.g., 172.20.77.176)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bitchat API port (default: 8765)"
    )
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip BitchatAlertClient integration test"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Bitchat VLM API Test Suite")
    print(f"{'='*60}")
    print(f"IP: {args.ip}")
    print(f"Port: {args.port}")
    print(f"Base URL: http://{args.ip}:{args.port}")
    
    results = []
    
    results.append(("GET /status", test_status_endpoint(args.ip, args.port)))
    time.sleep(1)
    
    results.append(("POST /send/text", test_send_text(args.ip, args.port)))
    time.sleep(6)
    
    results.append(("POST /send/image (with caption)", test_send_image_with_caption(args.ip, args.port)))
    time.sleep(6)
    
    results.append(("POST /send/image (no caption)", test_send_image_no_caption(args.ip, args.port)))
    
    if not args.skip_integration:
        time.sleep(6)
        results.append(("BitchatAlertClient", test_bitchat_client_integration(args.ip, args.port)))
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
