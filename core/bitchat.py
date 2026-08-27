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


def _prepare_image(frame: np.ndarray, max_dim: int = 640, quality: int = 70) -> bytes:
    """Downscale + JPEG-encode a frame for reliable mesh delivery.

    Bitchat transfers images as fragmented mesh packets; smaller images mean
    fewer fragments and a much higher chance of successful delivery.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
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
        msg = f"[CAM] {scene_text}"
        print(f"[bitchat] Queuing scene message: {msg[:80]}...")
        
        self._enqueue("/send/text", {"text": msg}, None, skip_rate=True)
        
        if frame is not None:
            print(f"[bitchat] Queuing scene image (downscaled for mesh)")
            # The phone's API silently drops an image sent within 5s of the
            # previous message (rate_limit_ms=5000). Delay the image so it
            # arrives after the phone's rate window is free.
            self._enqueue("/send/image", {"caption": msg}, frame,
                          skip_rate=True, delay_s=5.5)

    def send_alert(
        self,
        event_type: str,
        detail: str,
        frame: np.ndarray | None = None,
        priority: bool = False,
    ) -> None:
        ts    = datetime.datetime.now().strftime("%H:%M:%S")
        msg   = f"[{event_type.upper()}] {ts} - {detail}"
        
        if frame is not None:
            self._enqueue("/send/image", {"caption": msg}, frame, skip_rate=priority)
        else:
            self._enqueue("/send/text", {"text": msg}, None, skip_rate=priority)

    def _enqueue(
        self,
        endpoint: str,
        data: dict,
        frame: np.ndarray | None,
        skip_rate: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        with self._lock:
            frame_copy = frame.copy() if frame is not None else None
            self._queue.append((endpoint, data, frame_copy, skip_rate, delay_s))
            if endpoint == "/send/image":
                print(f"[bitchat] ENQUEUE IMAGE: queue size={len(self._queue)}, frame={'YES' if frame_copy is not None else 'NO'}")

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

            endpoint, data, frame, skip_rate, delay_s = item

            print(f"[bitchat] WORKER: Processing {endpoint}, has_frame={'YES' if frame is not None else 'NO'}")

            # Rate limiting - Bitchat default is 5000ms between messages
            if not skip_rate:
                elapsed = time.perf_counter() - self._last_sent_t
                if elapsed < self.rate_limit_s:
                    time.sleep(self.rate_limit_s - elapsed)

            # Delayed send — keeps the phone's own 5s API rate window free
            # (an image sent sooner is silently dropped by the phone).
            if delay_s > 0:
                time.sleep(delay_s)

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
        print(f"[bitchat] _send called: endpoint={endpoint}, frame={'YES' if frame is not None else 'NO'}")
        
        if endpoint == "/send/image" and frame is not None:
            try:
                url = self.base_url + "/send/image"
                jpeg = _prepare_image(frame)
                caption = data.get("caption") or data.get("description") or data.get("text", "")
                
                print(f"[bitchat] Sending image ({len(jpeg)} bytes) with caption: {caption[:50]}")
                
                files = {"image": ("surveillance_frame.jpg", jpeg, "image/jpeg")}
                form_data = {}
                if caption:
                    form_data["caption"] = caption
                if self.channel:
                    form_data["channel"] = self.channel

                r = None
                for attempt in range(2):
                    try:
                        r = requests.post(url, files=files, data=form_data, timeout=self.timeout)
                        if r.status_code == 200:
                            break
                    except Exception as _e:
                        print(f"[bitchat] image POST attempt {attempt + 1} failed: {_e}")
                        if attempt == 0:
                            time.sleep(1.0)
                if r is None:
                    raise RuntimeError("image POST failed after retries")
                print(f"[bitchat] Response: {r.status_code} - {r.json()}")
            except Exception as e:
                print(f"[bitchat] ERROR sending image: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            url = self.base_url + "/send/text"
            text_payload = data.get("text") or data.get("description", "")
            print(f"[bitchat] Sending text: {text_payload[:80]}")
            
            payload = {"text": text_payload}
            if self.channel:
                payload["channel"] = self.channel
                
            r = requests.post(url, json=payload, timeout=self.timeout)
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
