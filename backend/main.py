"""
Shadow Flicker Assessment Tool - FastAPI Backend
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import json
import math
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

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
    provider_used: str = "unknown"
    fallback_used: bool = False
    error: Optional[str] = None


ANALYSIS_PROMPT = """
You are an expert wind energy environmental impact analyst. You are analyzing a satellite image that shows only the area inside the shadow flicker impact zone for a proposed wind turbine.

Important constraints:
- The image is already cropped or masked to the ellipse.
- Analyze only what is visible inside the ellipse.
- Ignore everything outside the ellipse, including blacked-out regions.
- Base every field only on direct visual evidence from the image.
- Do not infer hidden buildings or sensitive sites from nearby context.

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
- OVERRIDE: Include partially visible buildings if any portion lies inside the ellipse.
- OVERRIDE: Do NOT count field boundaries, crop patterns, tree clusters, bushes, shadows, small dark dots, vehicles, roads, or image artifacts as buildings.
- OVERRIDE: If many small buildings are tightly clustered in a visible settlement, estimate the count realistically instead of counting only isolated roofs.
- OVERRIDE: If a possible structure is ambiguous or low-confidence, exclude it.
- OVERRIDE: If no clear buildings are visible, set total = 0, residential = 0, commercial = 0, sensitive_locations = [], and affected_population_estimate = 0.
- OVERRIDE: Only report sensitive locations if the image itself clearly shows a specific sensitive facility. Never infer a school, religious building, clinic, playground, or house from nearby settlement patterns alone.
- OVERRIDE: If the zone is mostly open fields, scrubland, or bare soil, prefer land_use = "Agricultural" or "Barren".
- OVERRIDE: 0 clear buildings usually means Low risk with score 0-20.
- OVERRIDE: Base every field only on direct visual evidence from the image.
- OVERRIDE: In sparse rural scenes, stay conservative when structures are ambiguous, but do not undercount clearly visible village clusters.
- OVERRIDE: In dense urban scenes with continuous rooftops across much of the ellipse, do NOT undercount by trying to enumerate only a few clearly isolated roofs.
- OVERRIDE: For dense urban scenes, return a realistic estimate of total visible buildings/structures within the ellipse, even if exact counting is impossible.
- OVERRIDE: If the ellipse covers a dense city-core or tightly packed urban neighborhood, totals may reasonably be in the hundreds or thousands rather than tens.
- OVERRIDE: If density.classification is "Urban" and the ellipse is visually packed with buildings, building_count.total should usually be at least several hundred.
- OVERRIDE: For dense urban scenes, set risk_assessment.score near the top of the range, and use 95-100 when the zone is clearly unsuitable.
"""


def _extract_json_from_text(raw: str) -> dict:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _refresh_env() -> None:
    """Reload .env so API key/account switches are picked up without stale process state."""
    load_dotenv(override=True)


def _summarize_provider_error(provider: str, exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()

    if "429" in text and ("quota" in lowered or "resource_exhausted" in lowered or "insufficient_quota" in lowered):
        return f"{provider} quota exceeded"
    if "api_key" in lowered and ("missing" in lowered or "invalid" in lowered or "placeholder" in lowered):
        return f"{provider} API key missing or invalid"
    if "json parse failed" in lowered:
        return f"{provider} returned an unreadable response"
    if "timed out" in lowered or "timeout" in lowered:
        return f"{provider} request timed out"
    return f"{provider} request failed"


def _normalize_analysis(data: dict) -> dict:
    """Sanity-check model output so the dashboard does not show inconsistent results."""
    building_count = data.get("building_count") or {}
    total = max(0, int(round(building_count.get("total", 0))))
    residential = max(0, int(round(building_count.get("residential", 0))))
    commercial = max(0, int(round(building_count.get("commercial", 0))))

    known_total = residential + commercial
    if known_total > total:
        total = known_total

    if total == 0:
        residential = 0
        commercial = 0
    elif known_total == 0:
        residential = max(total - 1, 0)
        commercial = total - residential

    density = data.get("density") or {}
    density_score = round(_clamp(float(density.get("score", 1)), 1, 10), 1)
    classification = str(density.get("classification", "")).title()
    if classification not in {"Rural", "Suburban", "Urban"}:
        classification = "Rural" if density_score < 3 else ("Suburban" if density_score < 6.5 else "Urban")

    if total == 0:
        density_score = 1.0
    elif total <= 12:
        density_score = max(density_score, 2.0)
    elif total <= 40:
        density_score = max(density_score, 3.0)
    elif total <= 120:
        density_score = max(density_score, 4.0)
    elif total <= 300:
        density_score = max(density_score, 6.0)
    else:
        density_score = max(density_score, 8.0)

    if classification == "Rural":
        density_score = min(density_score, 4.9)
    elif classification == "Suburban":
        density_score = _clamp(density_score, 3.0, 7.4)
    else:
        density_score = max(density_score, 7.0)

    density_score = round(_clamp(density_score, 1, 10), 1)
    classification = "Rural" if density_score < 3 else ("Suburban" if density_score < 6.5 else "Urban")

    sensitive_locations = [str(item).strip() for item in data.get("sensitive_locations", []) if str(item).strip()]

    vegetation = str(data.get("vegetation_coverage", "Moderate")).title()
    if vegetation == "Medium":
        vegetation = "Moderate"
    if vegetation not in {"Low", "Moderate", "High"}:
        vegetation = "Moderate"

    water_bodies = bool(data.get("water_bodies", False))

    infrastructure_quality = str(data.get("infrastructure_quality", "Fair")).title()
    if infrastructure_quality not in {"Poor", "Fair", "Good"}:
        infrastructure_quality = "Fair"

    land_use = str(data.get("land_use", "Mixed")).title()
    if land_use not in {"Agricultural", "Residential", "Commercial", "Mixed", "Industrial", "Forest", "Barren"}:
        land_use = "Mixed"

    affected_population_estimate = int(round(data.get("affected_population_estimate", total * 3.5)))
    if total == 0:
        affected_population_estimate = 0
    elif affected_population_estimate < int(total * 2):
        affected_population_estimate = int(round(total * 3.5))

    risk_score = 5
    risk_score += min(total * 0.75, 45)
    risk_score += (density_score - 1) * 4
    risk_score += min(len(sensitive_locations) * 12, 24)
    if water_bodies:
        risk_score += 4
    if land_use in {"Residential", "Commercial", "Industrial"}:
        risk_score += 6
    elif land_use == "Mixed":
        risk_score += 3

    risk_score = int(round(_clamp(risk_score, 0, 100)))
    if classification == "Rural":
        risk_score = min(risk_score, 65)
    if total == 0:
        risk_score = min(risk_score, 20)

    risk_level = "Low" if risk_score < 31 else ("Medium" if risk_score < 66 else "High")

    factors = [str(item).strip() for item in (data.get("risk_assessment") or {}).get("factors", []) if str(item).strip()]
    if not factors:
        factors = [
            f"{classification} area with approximately {total} visible structures",
            "Assessment uses only features visible inside the ellipse",
            "Site verification is recommended before final setback decisions",
        ]

    recommendation = str(data.get("recommendation", "")).strip()
    if not recommendation:
        recommendation = (
            "Proceed with standard verification, as the visible impact zone appears lightly developed."
            if risk_level == "Low" else
            "Verify all visible dwellings in the ellipse before finalizing shadow flicker mitigation and setback compliance."
            if risk_level == "Medium" else
            "High visible exposure is indicated; review turbine placement and require a detailed shadow flicker mitigation plan."
        )

    action_items = [str(item).strip() for item in data.get("action_items", []) if str(item).strip()]
    if len(action_items) < 2:
        action_items = [
            "Validate visible building count with higher-resolution imagery",
            "Confirm any sensitive buildings before final siting approval",
            "Re-run shadow flicker compliance using verified structures",
        ]

    return {
        "building_count": {
            "total": total,
            "residential": residential,
            "commercial": commercial,
        },
        "density": {
            "score": density_score,
            "classification": classification,
        },
        "sensitive_locations": sensitive_locations,
        "vegetation_coverage": vegetation,
        "water_bodies": water_bodies,
        "infrastructure_quality": infrastructure_quality,
        "risk_assessment": {
            "level": risk_level,
            "score": risk_score,
            "factors": factors[:5],
        },
        "recommendation": recommendation,
        "action_items": action_items[:4],
        "affected_population_estimate": affected_population_estimate,
        "land_use": land_use,
    }


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
    _refresh_env()
    provider_errors: list[str] = []
    data: Optional[dict] = None
    preferred_provider = os.getenv("VISION_PROVIDER", "gemini").strip().lower()
    provider_used = "unknown"

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
            provider_used = provider
            break
        except (httpx.HTTPError, RuntimeError) as exc:
            provider_errors.append(_summarize_provider_error(provider, exc))

    if data is None:
        return _mock_analysis(request, ", ".join(provider_errors) + "; using conservative fallback analysis.")

    data = _normalize_analysis(data)

    # Calculate shadow zone area (ellipse: π × a × b, a=850m, b=500m approx)
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
        success=True,
        provider_used=provider_used,
        fallback_used=False,
        error=None,
    )


def _mock_analysis(request: AnalysisRequest, error_message: Optional[str] = None) -> AnalysisResponse:
    """Returns conservative deterministic fallback data when vision providers are unavailable."""
    a = request.shadow_radius_m
    b = a * 0.58
    area_ha = round(math.pi * a * b / 10000, 1)
    fallback = _normalize_analysis(
        {
            "building_count": {
                "total": 0,
                "residential": 0,
                "commercial": 0,
            },
            "density": {
                "score": 1,
                "classification": "Rural",
            },
            "sensitive_locations": [],
            "vegetation_coverage": "Moderate",
            "water_bodies": False,
            "infrastructure_quality": "Fair",
            "risk_assessment": {
                "level": "Low",
                "score": 10,
                "factors": [
                    "Vision provider unavailable, so no reliable structure count was produced",
                    "Fallback result is intentionally conservative",
                    "A fresh analysis should be run once the model connection is restored",
                ],
            },
            "recommendation": "Vision analysis is unavailable. Re-run the assessment with the AI provider enabled before using the result for planning decisions.",
            "action_items": [
                "Restore the configured vision API connection",
                "Re-run the image analysis",
                "Validate the returned building count before acting on the risk score",
            ],
            "affected_population_estimate": 0,
            "land_use": "Mixed",
        }
    )

    return AnalysisResponse(
        building_count=BuildingCount(**fallback["building_count"]),
        density=DensityInfo(**fallback["density"]),
        sensitive_locations=fallback["sensitive_locations"],
        vegetation_coverage=fallback["vegetation_coverage"],
        water_bodies=fallback["water_bodies"],
        infrastructure_quality=fallback["infrastructure_quality"],
        risk_assessment=RiskAssessment(**fallback["risk_assessment"]),
        recommendation=fallback["recommendation"],
        action_items=fallback["action_items"],
        affected_population_estimate=fallback["affected_population_estimate"],
        shadow_zone_area_ha=area_ha,
        land_use=fallback["land_use"],
        success=False,
        provider_used="fallback",
        fallback_used=True,
        error=error_message
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
