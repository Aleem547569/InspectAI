import base64
import json
import threading
import time
import re
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import anthropic
from collections import deque
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='.', static_folder='static')
CORS(app)

print("⚙  Starting InspectAI (Claude Vision only)...")
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
print("✅  Ready")

# ── Global State ───────────────────────────────────────────────────────────────
_lock             = threading.Lock()
_last_verdict     = None
_last_analysis_ts = 0.0

_log: deque = deque(maxlen=50)
_stats = {"total": 0, "PASS": 0, "REWORK": 0, "QUARANTINE": 0, "SCRAP": 0}
_stats_lock = threading.Lock()

CLAUDE_PROMPT = (
    "You are a world-class industrial quality control inspector with 20+ years of "
    "manufacturing experience. Analyze this camera frame from a production line.\n\n"
    "1. Identify the main object visible in the frame.\n"
    "2. Inspect it for: surface defects, cracks, contamination, color anomalies, "
    "structural damage, scratches, dents, or any quality concern.\n\n"
    "If no clear object is visible, still return JSON with verdict QUARANTINE.\n\n"
    "Return ONLY valid JSON — no markdown, no explanation outside the JSON:\n"
    '{"class_name":"bottle","verdict":"PASS","defect_type":"None detected",'
    '"severity":"low","reasoning":"Object appears intact with no visible defects.",'
    '"confidence":0.92}\n\n'
    "Verdict guide:\n"
    "  PASS        — meets quality standards\n"
    "  REWORK      — minor defect, salvageable\n"
    "  QUARANTINE  — uncertain, hold for human review\n"
    "  SCRAP       — severe defect, discard immediately"
)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    global _last_verdict, _last_analysis_ts

    data = request.get_json(force=True)
    if not data or 'image' not in data or not data['image']:
        return jsonify({"status": "error", "message": "No image"}), 400

    with _lock:
        cached  = dict(_last_verdict) if _last_verdict else None
        last_ts = _last_analysis_ts

    now = time.time()

    # Return cached result if within cooldown
    if now - last_ts < 3.0:
        return jsonify({"status": "ok", "detections": [], "verdict": cached,
                        "frame_w": 640, "frame_h": 480})

    with _lock:
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
                            "data": data['image'],
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
            verdict_data = json.loads(m.group()) if m else {
                "class_name": "object",
                "verdict": "QUARANTINE",
                "defect_type": "Parse error",
                "severity": "medium",
                "reasoning": raw[:300],
                "confidence": 0.5,
            }

        verdict_data["timestamp"] = datetime.now().strftime("%H:%M:%S")

        with _lock:
            _last_verdict = verdict_data

        v = verdict_data.get("verdict", "SCRAP")
        with _stats_lock:
            _stats["total"] += 1
            if v in _stats:
                _stats[v] += 1

        _log.appendleft({
            "time":     verdict_data["timestamp"],
            "object":   verdict_data.get("class_name", "object"),
            "verdict":  v,
            "severity": verdict_data.get("severity", "—"),
            "defect":   verdict_data.get("defect_type", ""),
        })

        return jsonify({"status": "ok", "detections": [],
                        "verdict": verdict_data, "frame_w": 640, "frame_h": 480})

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
    port = int(os.environ.get("PORT", 5001))
    print(f"🌐  Open → http://localhost:{port}\n")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
