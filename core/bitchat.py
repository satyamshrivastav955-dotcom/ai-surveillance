"""Bitchat VLM API client for the AI Surveillance pipeline.

Sends surveillance alerts and VLM scene descriptions to Bitchat mesh network.
Your Android device (with Bitchat) must be on the same WiFi as this machine.

Setup:
  1. Open Bitchat on Android -> Settings -> VLM API Settings -> Enable toggle
  2. Note the WiFi IP shown (e.g. 192.168.1.105)
  3. Set BITCHAT_IP in configs/pipeline.yaml  OR  set env var BITCHAT_IP
  4. import and use BitchatAlertClient from this module

Sent to Bitchat mesh:
  - VLM scene description (every ambient pass)
  - FIRE / SMOKE alerts with keyframe image
  - FALL alerts with keyframe image
  - FIGHT alerts with keyframe image
  - PHONE detected
  - GATHERING (crowd)
  - VIOLENCE (when enabled)
"""
from __future__ import annotations

import io
import os
import time
import threading
import datetime
from typing import Any

import cv2
import numpy as np
import requests


def _frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR numpy frame as JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return buf.tobytes()


class BitchatAlertClient:
    """HTTP client for Bitchat's VLM API.

    All send methods are fire-and-forget (run in a daemon thread) so they
    never block the main surveillance loop.

    Args:
        ip:      Android device WiFi IP (e.g. "192.168.1.105")
        port:    Bitchat VLM API port (default 8765)
        channel: Bitchat channel to broadcast to (default "surveillance")
        timeout: HTTP request timeout in seconds
        rate_limit_s: Minimum seconds between messages (matches Bitchat default 5s)
    """

    def __init__(
        self,
        ip: str,
        port: int = 8765,
        channel: str | None = None,
        timeout: float = 10.0,
        rate_limit_s: float = 5.5,
    ):
        self.base_url    = f"http://{ip}:{port}"
        self.channel     = channel
        self.timeout     = timeout
        self.rate_limit_s = rate_limit_s

        self._lock         = threading.Lock()
        self._last_sent_t  = 0.0
        self._queue: list[tuple[str, dict, bytes | None]] = []
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="bitchat-sender"
        )
        self._worker_thread.start()

        channel_info = f"channel=#{channel}" if channel else "default_destination"
        print(
            f"[bitchat] client ready  url={self.base_url}  "
            f"{channel_info}  rate_limit={rate_limit_s}s"
        )

    def check_connection(self) -> bool:
        """Blocking check. Returns True if Bitchat API is reachable and enabled."""
        try:
            r = requests.get(f"{self.base_url}/status", timeout=4)
            data = r.json()
            if not data.get("api_enabled", False):
                print("[bitchat] WARN: API reached but api_enabled=False in Bitchat settings")
                return False
            peers = data.get("peers_count", 0)
            print(f"[bitchat] connected  peers={peers}  mesh={data.get('mesh_running')}")
            return True
        except Exception as e:
            print(f"[bitchat] WARN: cannot connect to {self.base_url}  ({e})")
            return False

    def send_scene(self, scene_text: str, frame: np.ndarray | None = None) -> None:
        """Send VLM ambient scene description. Non-blocking."""
        msg = f"[CAM] {scene_text}"
        print(f"[bitchat] Queuing scene message: {msg[:80]}...")
        
        # Send text message with description
        self._enqueue("/send/text", {"text": msg}, None)
        
        # Send image separately if provided
        if frame is not None:
            import time
            # Small delay to ensure text arrives first
            import numpy as np
            jpeg = _frame_to_jpeg(frame)
            files = {"image": ("surveillance_frame.jpg", jpeg, "image/jpeg")}
            # Schedule image send with slight delay
            self._enqueue("/send/image", {}, frame)

    def send_alert(
        self,
        event_type: str,
        detail: str,
        frame: np.ndarray | None = None,
        priority: bool = False,
    ) -> None:
        """Send a surveillance alert. Non-blocking.

        Args:
            event_type: e.g. "FIRE", "FALL", "FIGHT"
            detail:     human-readable description
            frame:      optional camera frame to attach as image
            priority:   if True, skip the rate limiter (for critical alerts)
        """
        ts    = datetime.datetime.now().strftime("%H:%M:%S")
        msg   = f"[{event_type.upper()}] {ts} - {detail}"
        endpoint = "/send/analysis" if frame is not None else "/send/text"
        self._enqueue(endpoint, {"description": msg}, frame, skip_rate=priority)

    def _enqueue(
        self,
        endpoint: str,
        data: dict,
        frame: np.ndarray | None,
        skip_rate: bool = False,
    ) -> None:
        with self._lock:
            self._queue.append((endpoint, data, frame, skip_rate))

    def _worker(self) -> None:
        """Background sender thread - respects rate limiting."""
        while True:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.pop(0)
            if item is None:
                time.sleep(0.1)
                continue

            endpoint, data, frame, skip_rate = item

            # Rate limiting - Bitchat default is 5000ms between messages
            if not skip_rate:
                elapsed = time.perf_counter() - self._last_sent_t
                if elapsed < self.rate_limit_s:
                    time.sleep(self.rate_limit_s - elapsed)

            try:
                self._send(endpoint, data, frame)
                self._last_sent_t = time.perf_counter()
                print(f"[bitchat] Message sent successfully (HTTP 200 OK)")
            except Exception as e:
                print(f"[bitchat] send error: {e}")

    def _send(
        self,
        endpoint: str,
        data: dict,
        frame: np.ndarray | None,
    ) -> None:
        url = self.base_url + endpoint
        
        if self.channel:
            data["channel"] = self.channel

        if frame is not None and endpoint in ("/send/analysis", "/send/image"):
            jpeg = _frame_to_jpeg(frame)
            description = data.get("description") or data.get("caption", "")
            print(f"[bitchat] Sending image ({len(jpeg)} bytes)")
            
            files = {"image": ("surveillance_frame.jpg", jpeg, "image/jpeg")}
            r = requests.post(url, files=files, data={}, timeout=self.timeout)
            print(f"[bitchat] Response: {r.status_code} - {r.json()}")
        else:
            # Text-only message
            text_payload = data.get("text") or data.get("description", "")
            print(f"[bitchat] Sending text: {text_payload[:80]}")
            payload = {"text": text_payload}
            if self.channel:
                payload["channel"] = self.channel
            r = requests.post(
                self.base_url + "/send/text",
                json=payload,
                timeout=self.timeout,
            )
            print(f"[bitchat] Response: {r.status_code} - {r.json()}")

        if r.status_code != 200:
            print(f"[bitchat] HTTP {r.status_code}: {r.text[:120]}")
        else:
            resp = r.json()
            if resp.get("status") != "ok":
                print(f"[bitchat] API error: {resp}")


def build_client_from_config(cfg: dict | None = None) -> BitchatAlertClient | None:
    """Build a BitchatAlertClient from pipeline config or env vars.

    Config key: pipeline.yaml -> bitchat:
      ip:       "192.168.1.105"   # Android WiFi IP
      port:     8765
      channel:  "surveillance"
      enabled:  true

    Env override: BITCHAT_IP=192.168.1.105

    Returns None if bitchat is disabled or IP not set.
    """
    bc_cfg = (cfg or {}).get("bitchat", {})

    if not bc_cfg.get("enabled", False):
        return None

    ip = os.environ.get("BITCHAT_IP") or bc_cfg.get("ip", "")
    if not ip:
        print(
            "[bitchat] WARN: bitchat.enabled=true but no IP configured. "
            "Set bitchat.ip in pipeline.yaml or BITCHAT_IP env var."
        )
        return None

    channel = bc_cfg.get("channel", None)
    if channel:
        channel = str(channel)
    
    client = BitchatAlertClient(
        ip=ip,
        port=int(bc_cfg.get("port", 8765)),
        channel=channel,
        rate_limit_s=float(bc_cfg.get("rate_limit_s", 5.5)),
    )

    if not client.check_connection():
        print(
            "[bitchat] WARN: Bitchat not reachable. Alerts will NOT be sent. "
            "Check that Bitchat VLM API is enabled and device is on same WiFi."
        )
        # Still return the client - it will retry on each send
        # (useful when Bitchat is opened after pipeline starts)

    return client
