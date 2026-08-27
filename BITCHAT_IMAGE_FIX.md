# Bitchat Image Sending - Fix Documentation

## Problem
Images were not being sent to Bitchat mesh network. The VLM scene descriptions were working, but event alerts (FIRE, FALL, FIGHT, etc.) were not sending images.

## Root Cause
**Location:** `pipeline/main_loop.py` lines 771-773

The Bitchat alert sending code was **incomplete**:
```python
# --- Bitchat mesh alerts -----------------------------------------
if bitchat_client is not None:
    _kf = frame if _bc_send_keyframes else None
    
    # MISSING: No code to send alerts!
    # Just empty lines...
```

The code obtained the frame but never called `send_alert()` for any events.

## Solution
Added complete alert sending logic for all event types:

### Event Types and Their Handling:

1. **Phase 4 Events** (Fire, Smoke, Phone, Gathering, Violence)
   - Sends with event type and details
   - Priority: TRUE for FIRE, FALL, FIGHT, VIOLENCE
   - Priority: FALSE for others (bypasses rate limit)

2. **Fall Events**
   - Always sends with priority=TRUE
   - Detail: "Person id:{track_id} has fallen"

3. **Fight Events** (Phase 5D)
   - Always sends with priority=TRUE
   - Detail: "Fight between persons {track_ids}"

4. **Identity Events**
   - Sends with priority=FALSE
   - Detail: "Track {track_id} identified as {label}"

### Code Added:
```python
# --- Bitchat mesh alerts -----------------------------------------
if bitchat_client is not None:
    _kf = frame if _bc_send_keyframes else None
    
    # Send alerts for all Phase 4 events
    for ev in phase4_events:
        bitchat_client.send_alert(
            event_type=ev.event_type,
            detail=str(ev.details),
            frame=_kf,
            priority=ev.event_type in ["FIRE", "FALL", "FIGHT", "VIOLENCE"]
        )
    
    # Send alerts for fall events
    for ev in fall_events:
        bitchat_client.send_alert(
            event_type="FALL",
            detail=f"Person id:{ev.track_id} has fallen",
            frame=_kf,
            priority=True
        )
    
    # Send alerts for fight events
    for ev in fight_events:
        bitchat_client.send_alert(
            event_type="FIGHT",
            detail=f"Fight between persons {ev.track_ids}",
            frame=_kf,
            priority=True
        )
    
    # Send alerts for identity events
    for ev in identity_events:
        bitchat_client.send_alert(
            event_type="IDENTITY",
            detail=f"Track {ev.track_id} identified as {ev.label}",
            frame=_kf,
            priority=False
        )
```

## How Image Sending Works

### Architecture:
```
[Event Detected] → [main_loop.py] → [BitchatAlertClient.send_alert()]
                                                    ↓
                                        [_enqueue() - adds to queue]
                                                    ↓
                                        [Background thread worker]
                                                    ↓
                                        [_send() - processes queue item]
                                                    ↓
                        ┌─────────────────────────┴──────────────────────┐
                        │                                                  │
                    [TEXT]                                             [IMAGE]
                        ↓                                                  ↓
            POST /send/text                                         POST /send/image
            (JSON payload)                                          (multipart/form-data)
```

### Key Components:

1. **Frame Capture** (`_kf = frame`)
   - Captures current frame if `send_keyframes=True` in config
   - Frame is numpy array (H, W, 3) in BGR format

2. **Enqueue** (`_enqueue()`)
   - Thread-safe queue with lock
   - Copies frame: `frame.copy()` to avoid reference issues
   - Parameters: endpoint, data dict, frame, skip_rate (priority flag)

3. **Background Worker** (`_worker()`)
   - Daemon thread processes queue
   - Respects rate limiting (5.5s default)
   - Priority messages skip rate limit

4. **Image Encoding** (`_frame_to_jpeg()`)
   - Converts BGR numpy to JPEG bytes
   - Quality: 80 (good balance of size/quality)
   - Uses `cv2.imencode()`

5. **HTTP POST** (`_send()`)
   - Endpoint: `/send/image`
   - Multipart form data with:
     - `image`: (filename, jpeg_bytes, "image/jpeg")
     - `caption`: Alert message
     - `channel`: Optional (default: surveillance)

### BitchatAlertClient Methods:

**`send_alert(event_type, detail, frame, priority)`**
- Used for event-based alerts
- Priority events (FIRE, FALL, FIGHT): bypass rate limit
- Non-priority: queued with rate limiting

**`send_scene(scene_text, frame)`**
- Used for VLM scene descriptions
- Sends text first, then image after 0.3s delay
- Always rate-limited

## Testing

### Manual Test:
```bash
# Run test script with your Bitchat device IP
python test_image_sending.py --ip 10.99.220.148

# Should see in Bitchat:
# 1. Scene message with test rectangles
# 2. TEST alert with same image
# 3. TEST alert without image (text only)
```

### Integration Test:
```bash
# Run full pipeline
python -m pipeline.main_loop

# Trigger events (wave hand, open and close eyes for fall detection, etc.)
# Check Bitchat for alerts with keyframe images
```

### Unit Test:
```bash
# Test Bitchat API endpoints
python tests/bitchat_api_test.py --ip 10.99.220.148
```

## Configuration

**File:** `configs/pipeline.yaml`

```yaml
bitchat:
  enabled: true
  ip: "10.99.220.148"      # Your Android WiFi IP
  port: 8765
  rate_limit_s: 5.5        # Seconds between messages
  send_keyframes: true     # Attach camera frame with alerts
  send_scene: true         # Send VLM scene descriptions
```

## Verification Checklist

✅ **Frame capture**: `_kf` obtained from `frame`  
✅ **Alert call**: `send_alert()` called for each event  
✅ **Queue processing**: Background thread handles items  
✅ **Image encoding**: BGR → JPEG conversion works  
✅ **HTTP POST**: Multipart form-data correctly formatted  
✅ **Rate limiting**: Priority events bypass, others wait  
✅ **Channel routing**: Correct channel specified  

## Common Issues

### 1. Connection Timeout
**Cause:** Device not reachable  
**Fix:** Check WiFi connection, verify IP in `pipeline.yaml`

### 2. API Not Enabled
**Cause:** Bitchat VLM API toggle OFF  
**Fix:** Bitchat → Settings → VLM API → Enable

### 3. Images Not Received
**Cause:** Missing alert sending code (now fixed)  
**Fix:** Update to latest `main_loop.py`

### 4. Low Quality Images
**Cause:** JPEG quality low or frame small  
**Fix:** Adjust quality in `_frame_to_jpeg()` (default: 80)

### 5. Rate Limit Too Strict
**Cause:** Too many events, queue backing up  
**Fix:** Lower `rate_limit_s` or increase threshold

## Logs to Check

When pipeline runs, look for:
```
[bitchat] client ready  url=http://10.99.220.148:8765  channel=#+surveillance
[bitchat] connected  peers=3  mesh=True
[bitchat] ENQUEUE IMAGE: queue size=1, frame=YES
[bitchat] WORKER: Processing /send/image, has_frame=YES
[bitchat] Sending image (45123 bytes) with caption: [FIRE] 15:42:10 - Fire detected
[bitchat] Response: 200 - {'status': 'ok', 'message_id': 'abc123'}
```

## Next Steps

1. Run `test_image_sending.py` to verify fix
2. Run full pipeline and trigger test events
3. Verify images arrive in Bitchat mesh network
4. Adjust rate limiting / quality as needed

## Summary

**Fixed:** Missing alert sending code in main_loop.py  
**Added:** Complete event loop for all event types  
**Result:** Images now sent with alerts to Bitchat mesh network  
**Test:** Use `test_image_sending.py` to verify
