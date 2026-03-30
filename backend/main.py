"""
Shadow Flicker Assessment Tool - FastAPI Backend
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import base64
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "building_count": {
            "type": "object",
            "properties": {
                "total": {"type": "integer"},
                "residential": {"type": "integer"},
                "commercial": {"type": "integer"},
            },
            "required": ["total", "residential", "commercial"],
        },
        "density": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "classification": {"type": "string"},
            },
            "required": ["score", "classification"],
        },
        "sensitive_locations": {"type": "array", "items": {"type": "string"}},
        "vegetation_coverage": {"type": "string"},
        "water_bodies": {"type": "boolean"},
        "infrastructure_quality": {"type": "string"},
        "risk_assessment": {
            "type": "object",
            "properties": {
                "level": {"type": "string"},
                "score": {"type": "integer"},
                "factors": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["level", "score", "factors"],
        },
        "recommendation": {"type": "string"},
        "action_items": {"type": "array", "items": {"type": "string"}},
        "affected_population_estimate": {"type": "integer"},
        "land_use": {"type": "string"},
    },
    "required": [
        "building_count",
        "density",
        "sensitive_locations",
        "vegetation_coverage",
        "water_bodies",
        "infrastructure_quality",
        "risk_assessment",
        "recommendation",
        "action_items",
        "affected_population_estimate",
        "land_use",
    ],
}

app = FastAPI(title="Shadow Flicker Assessment API", version="1.0.0")


def _parse_cors_origins() -> list[str]:
    """Allow common local frontend origins by default for easier dev setup."""
    configured = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,null",
    )
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ──────────────────────────────────────────────────

class TurbineSpecs(BaseModel):
    height: float = Field(default=150, description="Turbine hub height in meters")
    rotor_diameter: float = Field(default=120, description="Rotor diameter in meters")

class AnalysisRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded PNG/JPEG of map screenshot")
    latitude: float
    longitude: float
    shadow_radius_m: float = Field(default=850)
    turbine: TurbineSpecs = TurbineSpecs()
    timestamp: Optional[str] = None

class BuildingCount(BaseModel):
    total: int
    residential: int
    commercial: int

class DensityInfo(BaseModel):
    score: float  # 1-10
    classification: str  # Rural / Suburban / Urban

class RiskAssessment(BaseModel):
    level: str   # Low / Medium / High
    score: int   # 0-100
    factors: list[str]

class AnalysisResponse(BaseModel):
    building_count: BuildingCount
    density: DensityInfo
    sensitive_locations: list[str]
    vegetation_coverage: str   # Low / Moderate / High
    water_bodies: bool
    infrastructure_quality: str  # Poor / Fair / Good
    risk_assessment: RiskAssessment
    recommendation: str
    action_items: list[str]
    affected_population_estimate: int
    shadow_zone_area_ha: float
    land_use: str
    success: bool = True
    error: Optional[str] = None


ANALYSIS_PROMPT = """
You are an expert wind energy environmental impact analyst. You are analyzing a satellite image of a proposed wind turbine location with a shadow flicker zone overlay (an elliptical shaded area).

Analyze the satellite image carefully and return ONLY a valid JSON object with this EXACT structure (no markdown, no explanation):

{
  "building_count": {
    "total": <integer>,
    "residential": <integer>,
    "commercial": <integer>
  },
  "density": {
    "score": <float 1.0-10.0>,
    "classification": "<Rural|Suburban|Urban>"
  },
  "sensitive_locations": ["<description with approximate direction and distance>"],
  "vegetation_coverage": "<Low|Moderate|High>",
  "water_bodies": <true|false>,
  "infrastructure_quality": "<Poor|Fair|Good>",
  "risk_assessment": {
    "level": "<Low|Medium|High>",
    "score": <integer 0-100>,
    "factors": ["<factor1>", "<factor2>", "<factor3>"]
  },
  "recommendation": "<specific recommendation for turbine placement or mitigation>",
  "action_items": ["<action1>", "<action2>", "<action3>"],
  "affected_population_estimate": <integer>,
  "land_use": "<Agricultural|Residential|Commercial|Mixed|Industrial|Forest|Barren>"
}

Guidelines:
- Count ALL visible structures within the elliptical shadow zone
- Classify buildings as residential (houses, apartments) or commercial (shops, factories, offices)
- Identify sensitive locations: schools, hospitals, religious buildings, playgrounds, care homes
- Vegetation: Low (<20% green cover), Moderate (20-60%), High (>60%)
- Risk score: 0-30 = Low, 31-65 = Medium, 66-100 = High
- Population estimate: total buildings × 3.5 average occupancy
- Be conservative and precise in counts
- OVERRIDE: Count a building ONLY if a distinct roofed man-made structure is clearly visible inside the ellipse.
- OVERRIDE: Do NOT count field boundaries, crop patterns, tree clusters, bushes, shadows, small dark dots, vehicles, roads, or image artifacts as buildings.
- OVERRIDE: If a possible structure is ambiguous or low-confidence, exclude it.
- OVERRIDE: If no clear buildings are visible, set total = 0, residential = 0, commercial = 0, sensitive_locations = [], and affected_population_estimate = 0.
- OVERRIDE: Only report sensitive locations if the image itself clearly shows a specific sensitive facility. Never infer a school, religious building, clinic, playground, or house from nearby settlement patterns alone.
- OVERRIDE: If the zone is mostly open fields, scrubland, or bare soil, prefer land_use = "Agricultural" or "Barren".
- OVERRIDE: 0 clear buildings usually means Low risk with score 0-20.
- OVERRIDE: Prefer undercounting over hallucinating and base every field only on direct visual evidence from the image.
- OVERRIDE: In sparse rural scenes, stay conservative and prefer lower counts when structures are ambiguous.
- OVERRIDE: In dense urban scenes with continuous rooftops across much of the ellipse, do NOT undercount by trying to enumerate only a few clearly isolated roofs.
- OVERRIDE: For dense urban scenes, return a realistic estimate of total visible buildings/structures within the ellipse, even if exact counting is impossible.
- OVERRIDE: If the ellipse covers a dense city-core or tightly packed urban neighborhood, totals may reasonably be in the hundreds or thousands rather than tens.
- OVERRIDE: If density.classification is "Urban" and the ellipse is visually packed with buildings, building_count.total should usually be at least several hundred.
- OVERRIDE: For dense urban scenes, set risk_assessment.score near the top of the range, and use 95-100 when the zone is clearly unsuitable.
"""


def _extract_json_from_text(raw: str) -> dict:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


async def _call_gemini(request: AnalysisRequest) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    timeout = int(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "30"))
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": ANALYSIS_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": request.image_base64,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": ANALYSIS_SCHEMA,
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini returned {resp.status_code}: {resp.text}")

    body = resp.json()
    try:
        raw = body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Gemini response shape unexpected: {exc}") from exc

    try:
        return _extract_json_from_text(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini JSON parse failed: {exc}") from exc


async def _call_openai(request: AnalysisRequest) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-placeholder"):
        raise RuntimeError("OPENAI_API_KEY missing or placeholder")

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    timeout = int(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "30"))
    payload = {
        "model": model,
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYSIS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{request.image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI returned {resp.status_code}: {resp.text}")

    body = resp.json()
    try:
        raw = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"OpenAI response shape unexpected: {exc}") from exc

    try:
        return _extract_json_from_text(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI JSON parse failed: {exc}") from exc


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_location(request: AnalysisRequest):
    provider_errors: list[str] = []
    data: Optional[dict] = None
    preferred_provider = os.getenv("VISION_PROVIDER", "gemini").strip().lower()

    # Validate image
    max_mb = float(os.getenv("MAX_IMAGE_SIZE_MB", "5"))
    img_bytes = len(request.image_base64) * 3 / 4
    if img_bytes > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {max_mb}MB limit")

    providers = ["gemini", "openai"] if preferred_provider != "openai" else ["openai", "gemini"]

    for provider in providers:
        try:
            if provider == "gemini":
                data = await _call_gemini(request)
            else:
                data = await _call_openai(request)
            break
        except (httpx.HTTPError, RuntimeError) as exc:
            provider_errors.append(f"{provider}: {exc}")

    if data is None:
        return _mock_analysis(request, " | ".join(provider_errors) + "; using mock analysis.")

    # Calculate shadow zone area (ellipse: π × a × b, a=850m, b=500m approx)
    import math
    a = request.shadow_radius_m
    b = a * 0.58
    area_ha = round(math.pi * a * b / 10000, 1)

    return AnalysisResponse(
        building_count=BuildingCount(**data["building_count"]),
        density=DensityInfo(**data["density"]),
        sensitive_locations=data.get("sensitive_locations", []),
        vegetation_coverage=data.get("vegetation_coverage", "Moderate"),
        water_bodies=data.get("water_bodies", False),
        infrastructure_quality=data.get("infrastructure_quality", "Fair"),
        risk_assessment=RiskAssessment(**data["risk_assessment"]),
        recommendation=data.get("recommendation", "Further assessment required"),
        action_items=data.get("action_items", []),
        affected_population_estimate=data.get("affected_population_estimate", 0),
        shadow_zone_area_ha=area_ha,
        land_use=data.get("land_use", "Mixed"),
        success=True
    )


def _mock_analysis(request: AnalysisRequest, error_message: Optional[str] = None) -> AnalysisResponse:
    """Returns realistic mock data for demo/development purposes."""
    import math, random
    a = request.shadow_radius_m
    b = a * 0.58
    area_ha = round(math.pi * a * b / 10000, 1)

    total = random.randint(8, 72)
    residential = int(total * 0.82)
    commercial = total - residential

    score = random.randint(20, 85)
    level = "Low" if score < 31 else ("Medium" if score < 66 else "High")
    density_score = round(random.uniform(1.5, 8.5), 1)
    classification = "Rural" if density_score < 3 else ("Suburban" if density_score < 6.5 else "Urban")

    return AnalysisResponse(
        building_count=BuildingCount(total=total, residential=residential, commercial=commercial),
        density=DensityInfo(score=density_score, classification=classification),
        sensitive_locations=random.choice([
            ["Primary school ~320m NE", "Health clinic ~480m SE"],
            ["Religious building ~200m W"],
            [],
            ["Community hall ~150m N", "Kindergarten ~400m NW"]
        ]),
        vegetation_coverage=random.choice(["Low", "Moderate", "High"]),
        water_bodies=random.choice([True, False]),
        infrastructure_quality=random.choice(["Poor", "Fair", "Good"]),
        risk_assessment=RiskAssessment(
            level=level,
            score=score,
            factors=[
                f"{classification} area with {total} structures",
                "Setback compliance requires verification",
                "Shadow hours potentially exceeding 30h/year threshold"
            ]
        ),
        recommendation=(
            "Site appears suitable with standard mitigation measures. "
            "Automated shadow curtailment system strongly recommended."
            if score < 50 else
            "High sensitivity detected. Increase turbine setback to minimum 700m. "
            "Full environmental impact assessment required before proceeding."
        ),
        action_items=[
            "Commission detailed shadow flicker study (IEC 61400-11)",
            "Notify affected residents within 500m radius",
            "Install automated curtailment if >30h/year limit exceeded",
            "Consult local planning authority for noise & visual impact"
        ],
        affected_population_estimate=int(total * 3.5),
        shadow_zone_area_ha=area_ha,
        land_use=random.choice(["Agricultural", "Residential", "Mixed", "Commercial"]),
        success=True,
        error=error_message
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
