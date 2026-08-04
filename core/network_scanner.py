"""
Network scanner to auto-discover Bitchat VLM API endpoints.
Scans local subnet for devices running Bitchat on port 8765.
"""
from __future__ import annotations

import socket
import concurrent.futures
import time
from typing import Optional

import requests


def get_local_ip() -> str:
    """
    Get the local machine's IP address on the network.
    
    Returns:
        IP address as string (e.g., "192.168.1.100")
    
    Raises:
        RuntimeError: If unable to determine local IP
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        raise RuntimeError(f"Cannot determine local IP: {e}")


def get_local_subnet() -> str:
    """
    Get the PC's subnet prefix (first 3 octets).
    
    Returns:
        Subnet string (e.g., "192.168.1" or "172.20.77")
    """
    local_ip = get_local_ip()
    return ".".join(local_ip.split(".")[:3])


def test_ip_for_bitchat(
    ip: str,
    port: int = 8765,
    timeout: float = 1.0
) -> Optional[dict]:
    """
    Test if IP has Bitchat VLM API running.
    
    Args:
        ip: IP address to test
        port: Port number (default 8765)
        timeout: Request timeout in seconds
    
    Returns:
        Status dict if Bitchat found, None otherwise
        
    Example:
        >>> result = test_ip_for_bitchat("192.168.1.105")
        >>> if result and result.get("api_enabled"):
        ...     print("Bitchat found!")
    """
    url = f"http://{ip}:{port}/status"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "ok" and data.get("api_enabled"):
                return data
    except Exception:
        pass
    return None


def scan_for_bitchat_api(
    port: int = 8765,
    max_workers: int = 50,
    timeout: float = 1.0,
    verbose: bool = True
) -> Optional[str]:
    """
    Scan local subnet for Bitchat VLM API.
    
    Uses parallel scanning for speed (~10-30 seconds for full subnet).
    Scans all IPs from .1 to .254 in the local subnet.
    
    Args:
        port: Port number to scan (default 8765)
        max_workers: Number of parallel threads (default 50)
        timeout: Request timeout per IP in seconds (default 1.0)
        verbose: Print progress messages (default True)
    
    Returns:
        First valid IP address found, or None if not found
        
    Example:
        >>> ip = scan_for_bitchat_api()
        >>> if ip:
        ...     print(f"Found Bitchat at {ip}")
        ... else:
        ...     print("No Bitchat devices found")
    """
    try:
        subnet = get_local_subnet()
    except RuntimeError as e:
        if verbose:
            print(f"[scanner] ERROR: {e}")
        return None
    
    if verbose:
        print(f"[scanner] Scanning subnet {subnet}.x for Bitchat on port {port}...")
    
    target_ips = [f"{subnet}.{i}" for i in range(1, 255)]
    
    found_ip = None
    scanned = 0
    t0 = time.perf_counter()
    
    def check_ip(ip: str) -> Optional[str]:
        """Worker function for thread pool."""
        nonlocal scanned, found_ip
        
        result = test_ip_for_bitchat(ip, port, timeout)
        scanned += 1
        
        if result:
            if verbose and found_ip is None:
                elapsed = time.perf_counter() - t0
                print(
                    f"[scanner] Found Bitchat at {ip} "
                    f"(peers={result.get('peers_count', '?')}) "
                    f"[{scanned} IPs scanned in {elapsed:.1f}s]"
                )
            return ip
        return None
    
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = {
                executor.submit(check_ip, ip): ip
                for ip in target_ips
            }
            
            for future in concurrent.futures.as_completed(futures):
                ip_result = future.result()
                if ip_result and found_ip is None:
                    found_ip = ip_result
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                    
    except Exception as e:
        if verbose:
            print(f"[scanner] ERROR during scan: {e}")
    
    elapsed = time.perf_counter() - t0
    
    if verbose:
        if found_ip:
            print(f"[scanner] Discovery complete ({elapsed:.1f}s)")
        else:
            print(
                f"[scanner] No Bitchat devices found "
                f"({scanned} IPs scanned in {elapsed:.1f}s)"
            )
    
    return found_ip


def main():
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scan network for Bitchat VLM API devices"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port number to scan (default: 8765)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Request timeout per IP in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    ip = scan_for_bitchat_api(
        port=args.port,
        timeout=args.timeout,
        verbose=not args.quiet
    )
    
    if ip:
        print(f"\nFOUND: {ip}")
        print(f"URL: http://{ip}:{args.port}")
        print(f"Status endpoint: http://{ip}:{args.port}/status")
    else:
        print("\nNo Bitchat devices found on network")
        print("\nTroubleshooting:")
        print("  1. Ensure Bitchat app is running on Android device")
        print("  2. Enable VLM API in Bitchat settings")
        print("  3. Check that device is on same WiFi network")
        print(f"  4. Verify port {args.port} is correct")


if __name__ == "__main__":
    main()
