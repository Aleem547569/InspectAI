# InspectAI — AI-Powered Industrial Quality Inspector
### RoboAI Hackathon 2025

> **SDG 9 — Industry, Innovation and Infrastructure**
> InspectAI automates quality control in manufacturing facilities across Tunisia and emerging markets, replacing expensive human inspection labor with AI that is accessible, accurate, and scalable. It promotes inclusive industrialization by making world-class quality assurance available to small manufacturers who previously could not afford sophisticated inspection systems.

---

## What It Does

A live webcam quality-control station powered by YOLO object detection + Claude Vision AI:

1. **Detects** — YOLO11n detects objects in the live webcam feed in real time
2. **Inspects** — Claude Vision API (`claude-sonnet-4-5`) analyzes each detected object like a 20-year veteran QC inspector
3. **Decides** — Returns a structured verdict: `PASS` / `REWORK` / `QUARANTINE` / `SCRAP` with full expert reasoning
4. **Acts** — An animated robotic gantry arm on-screen moves to the correct bin

---

## Quick Start

```bash
# 1. Clone / enter the project
cd inspectai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 4. Run
python app.py

# 5. Open browser
open http://localhost:5000
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+ · Flask 3 |
| Computer Vision | OpenCV · Ultralytics YOLO11n |
| AI Brain | Anthropic Claude Vision API (`claude-sonnet-4-5`) |
| Frontend | Vanilla HTML/CSS/JS · SVG animations |
| Streaming | MJPEG via Flask Response generator |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard UI |
| `GET` | `/video_feed` | MJPEG webcam stream with YOLO boxes |
| `POST` | `/analyze` | Trigger Claude Vision analysis → JSON verdict |
| `GET` | `/stats` | Cumulative inspection statistics |
| `GET` | `/log` | Last 10 inspection records |

### Verdict JSON Schema
```json
{
  "verdict":    "PASS | REWORK | QUARANTINE | SCRAP",
  "defect_type": "None detected | specific defect description",
  "severity":   "low | medium | high",
  "reasoning":  "2-3 sentence expert analysis",
  "confidence": 0.0
}
```

---

## File Structure

```
inspectai/
├── app.py                  # Flask backend · YOLO · Claude API
├── requirements.txt        # Python dependencies
├── .env                    # ANTHROPIC_API_KEY (not committed)
├── templates/
│   └── index.html          # Full dashboard UI
├── static/
│   ├── style.css           # Industrial dark theme
│   └── app.js              # Webcam polling · arm animation · charts
└── README.md
```

---

## SDG 9 Impact Statement

Manufacturing quality control is a critical bottleneck for small and medium enterprises (SMEs) in developing economies. Traditional optical inspection systems cost $50,000–$200,000 — completely out of reach for most Tunisian or African manufacturers.

**InspectAI changes this equation:**
- Runs on a $200 webcam + any laptop
- Leverages Claude's reasoning to replicate expert human judgment
- Scales to any product category without retraining
- Provides audit-ready inspection logs
- Enables SMEs to meet international quality standards (ISO 9001 compliance support)

By democratizing AI-powered quality assurance, InspectAI directly advances **SDG 9 Target 9.3** — increasing the access of small-scale industrial enterprises to financial services and integration into value chains and markets.

---

## Notes

- YOLO model (`yolo11n.pt`) downloads automatically on first run (~6 MB)
- Claude is called at most every 2.5 seconds to stay within API rate limits
- All YOLO detected objects are analyzed — not just specific classes
- Bounding box colors update to reflect the latest Claude verdict
