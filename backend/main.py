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

app = FastAPI(title="Shadow Flicker Assessment API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
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
"""


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_location(request: AnalysisRequest):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-placeholder"):
        # Return mock data when no real API key
        return _mock_analysis(request)

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    timeout = int(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "30"))

    # Validate image
    max_mb = float(os.getenv("MAX_IMAGE_SIZE_MB", "5"))
    img_bytes = len(request.image_base64) * 3 / 4
    if img_bytes > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {max_mb}MB limit")

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
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {resp.text}")

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    # Strip any markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {e}")

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


def _mock_analysis(request: AnalysisRequest) -> AnalysisResponse:
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
        success=True
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
