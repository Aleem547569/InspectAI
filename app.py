import cv2
import base64
import json
import threading
import time
import re
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import anthropic
from collections import deque
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='.', static_folder='static')
CORS(app)

# ── Models & API Client ────────────────────────────────────────────────────────
print("⚙  Loading YOLO model...")
model = YOLO("yolo11n.pt")
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Global State ───────────────────────────────────────────────────────────────
_lock = threading.Lock()
_last_verdict = None
_last_analysis_ts = 0

_log: deque = deque(maxlen=50)
_stats = {"total": 0, "PASS": 0, "REWORK": 0, "QUARANTINE": 0, "SCRAP": 0}
_stats_lock = threading.Lock()

CLAUDE_PROMPT = (
    "You are a world-class industrial quality control inspector with 20+ years of "
    "manufacturing experience. A robotic vision system has captured this object from a "
    "production line and needs your expert verdict.\n\n"
    "Inspect for: surface defects, cracks, contamination, color anomalies, structural "
    "damage, dimensional deviations, stains, scratches, dents, or any quality concern.\n\n"
    "Return ONLY valid JSON — no markdown fences, no explanation outside the JSON:\n"
    '{"verdict":"PASS","defect_type":"None detected","severity":"low",'
    '"reasoning":"The object appears intact with no visible defects. '
    'Surface quality meets production standards.","confidence":0.95}\n\n'
    "Verdict guide:\n"
    "  PASS        — meets quality standards, proceed to shipment\n"
    "  REWORK      — minor defect, salvageable with correction\n"
    "  QUARANTINE  — uncertain quality, hold for expert human review\n"
    "  SCRAP       — severe defect, must be discarded immediately"
)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    global _last_verdict, _last_analysis_ts

    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({"status": "error", "message": "No image provided"}), 400

    # Decode base64 image from browser
    try:
        img_bytes = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    # Run YOLO detection
    results = model(frame, verbose=False, conf=0.35)

    detections = []
    best_det = None
    best_area = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)

            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_name": cls_name,
                "conf": round(conf, 3),
            })

            if area > best_area:
                best_area = area
                pad = 15
                crop = frame[
                    max(0, y1 - pad): min(frame.shape[0], y2 + pad),
                    max(0, x1 - pad): min(frame.shape[1], x2 + pad)
                ]
                if crop.size > 0:
                    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    best_det = {
                        "crop_b64": base64.b64encode(buf).decode(),
                        "class_name": cls_name,
                        "conf": round(conf, 3),
                    }

    with _lock:
        cached_verdict = _last_verdict

    response = {
        "status": "ok",
        "detections": detections,
        "verdict": cached_verdict,
        "frame_w": frame.shape[1],
        "frame_h": frame.shape[0],
    }

    if not best_det:
        response["verdict"] = None
        return jsonify(response)

    # Claude analysis with cooldown
    now = time.time()
    if now - _last_analysis_ts < 2.5:
        return jsonify(response)

    _last_analysis_ts = now

    try:
        msg = claude.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": best_det["crop_b64"],
                        },
                    },
                    {"type": "text", "text": CLAUDE_PROMPT},
                ],
            }],
        )

        raw = msg.content[0].text.strip()
        try:
            verdict_data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                verdict_data = json.loads(m.group())
            else:
                verdict_data = {
                    "verdict": "QUARANTINE",
                    "defect_type": "Analysis parse error",
                    "severity": "medium",
                    "reasoning": raw[:300],
                    "confidence": 0.5,
                }

        verdict_data["class_name"] = best_det["class_name"]
        verdict_data["conf"]       = best_det["conf"]
        verdict_data["timestamp"]  = datetime.now().strftime("%H:%M:%S")

        with _lock:
            _last_verdict = verdict_data

        v = verdict_data.get("verdict", "SCRAP")
        with _stats_lock:
            _stats["total"] += 1
            if v in _stats:
                _stats[v] += 1

        _log.appendleft({
            "time":     verdict_data["timestamp"],
            "object":   best_det["class_name"],
            "verdict":  v,
            "severity": verdict_data.get("severity", "—"),
            "defect":   verdict_data.get("defect_type", ""),
        })

        response["verdict"] = verdict_data
        return jsonify(response)

    except anthropic.APIError as e:
        return jsonify({"status": "error", "message": f"Claude API: {e}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stats')
def stats():
    with _stats_lock:
        s = dict(_stats)
    t = s["total"]
    s["pass_rate"] = round(s["PASS"] / t * 100, 1) if t else 0
    return jsonify(s)


@app.route('/log')
def inspection_log():
    return jsonify(list(_log)[:10])


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n╔══════════════════════════════════════╗")
    print("║   InspectAI  —  RoboAI Hackathon     ║")
    print("║   AI-Powered Industrial Inspector    ║")
    print("╚══════════════════════════════════════╝")
    print("🤖  YOLO11n + Claude Vision API ready")
    print("🌐  Open → http://localhost:5001\n")
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
