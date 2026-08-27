"""Auto-discover Bitchat VLM API devices on the local network.

Scans EVERY IPv4 subnet your PC is connected to (WiFi + Ethernet + more)
for a device listening on Bitchat's port (default 8765), then verifies it
by hitting /status. Prints the IP(s) to put in configs/pipeline.yaml.

Unlike core/network_scanner.py (which only scans the default-route subnet),
this handles a multi-homed PC (e.g. WiFi + Ethernet simultaneously).

Usage:
    python find_bitchat.py [--port 8765]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import socket
import time

import requests


def local_ipv4s():
    """Return the set of non-loopback, non-link-local IPv4 addresses."""
    import psutil
    ips = set()
    for _iface, addrs in psutil.net_if_addrs().items():
        for a in addrs:
            if a.family != socket.AF_INET:
                continue
            ip = a.address
            try:
                ifa = ipaddress.ip_address(ip)
            except ValueError:
                continue
            if ifa.is_loopback or ifa.is_link_local or ifa.is_unspecified:
                continue
            ips.add(ip)
    return ips


def subnets_from(ips, prefix=24):
    """Return the /24 subnets these IPs live in (deduped)."""
    out = set()
    for ip in ips:
        ifa = ipaddress.ip_interface(f"{ip}/{prefix}")
        out.add(str(ifa.network))
    return sorted(out)


def _port_open(ip: str, port: int, timeout: float) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((ip, port)) == 0
    finally:
        s.close()


def _verify_bitchat(ip: str, port: int, timeout: float) -> dict | None:
    try:
        r = requests.get(f"http://{ip}:{port}/status", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def scan(port: int = 8765, timeout: float = 1.0, max_workers: int = 128):
    ips = local_ipv4s()
    if not ips:
        print("[find] No usable IPv4 interface found.")
        return
    print(f"[find] Local interfaces: {', '.join(sorted(ips))}")

    nets = subnets_from(ips)
    print(f"[find] Scanning subnets: {', '.join(nets)}")

    hosts = [str(ip) for net in nets for ip in ipaddress.ip_network(net).hosts()]

    found = []

    def check(ip: str):
        if not _port_open(ip, port, timeout):
            return
        data = _verify_bitchat(ip, port, timeout)
        if data is not None:
            found.append((ip, data))
            print(f"[find] FOUND Bitchat at {ip}  peers={data.get('peers_count', '?')} "
                  f"api_enabled={data.get('api_enabled')}")

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(check, hosts))

    print(f"[find] Scan done in {time.perf_counter() - t0:.1f}s")

    if not found:
        print("[find] No Bitchat device found.")
        print("  Check:")
        print("  1. PC + phone are on the SAME WiFi network")
        print("  2. Bitchat -> Settings -> VLM API -> Enable is ON")
        print("  3. Bind address is 0.0.0.0 (not 127.0.0.1)")
        print(f"  4. Port {port} is correct")
        return

    print("\nTo use a phone, set in configs/pipeline.yaml:")
    for ip, data in found:
        print(f"    bitchat:\n      ip: \"{ip}\"   # peers={data.get('peers_count', '?')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--timeout", type=float, default=1.0)
    args = ap.parse_args()
    scan(port=args.port, timeout=args.timeout)


if __name__ == "__main__":
    main()
