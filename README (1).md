# Shadow Flicker Assessment Tool

Production-ready wind turbine shadow flicker impact assessment with interactive satellite map and AI-powered analytics dashboard.

---

## Architecture

```
shadow_flicker_frontend/   ← Pure HTML/JS (no build step needed)
  index.html               ← Complete app (Leaflet + dashboard)
  .env.example

shadow_flicker_backend/    ← Python FastAPI
  main.py
  requirements.txt
  .env.example
```

---

## Quick Start

### 1. Backend

```bash
cd shadow_flicker_backend
cp .env.example .env
# Edit .env → add your OPENAI_API_KEY
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Without an API key:** The backend returns realistic mock data automatically — great for development and demos.

### 2. Frontend

```bash
cd shadow_flicker_frontend
# Option A — no build step, just open in browser:
open index.html
# or serve locally:
python3 -m http.server 5173
# then visit http://localhost:5173
```

---

## Features

| Feature | Details |
|---|---|
| Satellite map | ESRI World Imagery (free, no API key) |
| Shadow zone | Elliptical overlay, dynamically sized by turbine specs |
| Turbine specs | Height (80–250m) and rotor diameter (60–200m) sliders |
| AI analysis | GPT-4o vision — counts buildings, detects sensitive locations |
| Risk gauge | 0–100 score with IEC 61400-11 reference |
| Building breakdown | Residential vs commercial with visual bar |
| Sensitive locations | Schools, hospitals, religious buildings flagged |
| Environmental context | Vegetation, water bodies, infrastructure quality |
| Recommendations | AI-generated action items |
| Analysis history | Stored in localStorage, clickable to revisit |
| Export | JSON download of full analysis result |
| Mock mode | Works without OpenAI API key (realistic random data) |

---

## Environment Variables

### Backend `.env`
```
OPENAI_API_KEY=sk-...         # Required for real analysis; mock used if absent
OPENAI_MODEL=gpt-4o           # or gpt-4-vision-preview
MAX_IMAGE_SIZE_MB=5
CORS_ORIGINS=http://localhost:5173
ANALYSIS_TIMEOUT_SECONDS=30
```

### Frontend (edit in index.html or create `.env` for Vite build)
```
API endpoint: http://localhost:8000   (line in index.html: const API = '...')
```

---

## API

### `POST /analyze`

**Request:**
```json
{
  "image_base64": "<base64 PNG>",
  "latitude": 28.6139,
  "longitude": 77.2090,
  "shadow_radius_m": 825,
  "turbine": { "height": 150, "rotor_diameter": 120 },
  "timestamp": "2026-03-30T10:00:00Z"
}
```

**Response:**
```json
{
  "building_count": { "total": 47, "residential": 39, "commercial": 8 },
  "density": { "score": 6.2, "classification": "Suburban" },
  "sensitive_locations": ["Primary school ~280m NE"],
  "vegetation_coverage": "Low",
  "water_bodies": false,
  "infrastructure_quality": "Fair",
  "risk_assessment": {
    "level": "Medium",
    "score": 58,
    "factors": ["Moderate residential density", "School within 300m", "Setback verification required"]
  },
  "recommendation": "Increase setback to minimum 600m and install curtailment system.",
  "action_items": [
    "Commission IEC 61400-11 shadow study",
    "Notify residents within 500m",
    "Install automated curtailment"
  ],
  "affected_population_estimate": 165,
  "shadow_zone_area_ha": 122.4,
  "land_use": "Residential",
  "success": true
}
```

### `GET /health`
Returns `{ "status": "ok", "version": "1.0.0" }`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML + CSS + JS (no framework required) |
| Maps | Leaflet.js + ESRI World Imagery |
| Screenshot | html2canvas |
| Backend | FastAPI (Python 3.10+) |
| AI Vision | OpenAI GPT-4o |
| Persistence | localStorage (frontend history) |

---

## Upgrading to Vite + React

The frontend is plain HTML for simplicity. To upgrade:

```bash
npm create vite@latest shadow-flicker-app -- --template react-ts
cd shadow-flicker-app
npm install leaflet react-leaflet html2canvas zustand recharts jspdf
```

Copy the logic from `index.html` into React components as described in the implementation guide.
