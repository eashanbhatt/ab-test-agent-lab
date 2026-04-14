"""
AB Test Agent Lab - FastAPI Backend
====================================
Run with: uvicorn app:app --reload
Open:      http://localhost:8000
"""

import os
import json
import math
import random
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="AB Test Agent Lab")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic client — reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


# ── Request / Response models ──────────────────────────────────────────────
class GenerateVariantsRequest(BaseModel):
    test_name: str
    component_type: str
    metric: str
    traffic_split: str
    control_description: str
    hypothesis: str


class SimulateRequest(BaseModel):
    daily_visitors: int
    base_ctr: float          # e.g. 0.10 for 10%
    effect_size: float        # e.g. 0.20 for +20% relative lift


class InsightRequest(BaseModel):
    test_name: str
    variant_a_rate: float
    variant_b_rate: float
    confidence: float
    visitors: int
    duration_days: int


# ── In-memory test store (replace with a real DB for production) ───────────
TESTS = [
    {
        "id": "t1",
        "name": "Onboarding step 2 layout",
        "component": "Onboarding flow",
        "metric": "Conversion rate",
        "status": "winner",
        "visitors": 12440,
        "day": 7,
        "variant_a": 11.2,
        "variant_b": 14.9,
        "confidence": 96.3,
        "started": (datetime.now() - timedelta(days=7)).isoformat(),
    },
    {
        "id": "t2",
        "name": "Hero headline copy",
        "component": "Headline copy",
        "metric": "Time on page",
        "status": "stopped",
        "visitors": 8100,
        "day": 5,
        "variant_a": 62.4,
        "variant_b": 63.1,
        "confidence": 44.2,
        "started": (datetime.now() - timedelta(days=5)).isoformat(),
    },
]


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML."""
    with open("index.html", "r") as f:
        return f.read()


@app.get("/api/tests")
async def get_tests():
    """Return all tests."""
    return {"tests": TESTS}


@app.post("/api/tests/{test_id}/stop")
async def stop_test(test_id: str):
    """Stop a running test."""
    for t in TESTS:
        if t["id"] == test_id:
            t["status"] = "stopped"
            return {"ok": True, "test": t}
    raise HTTPException(status_code=404, detail="Test not found")


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    """
    Calculate days to significance and return simulated daily data.
    Uses standard two-proportion z-test sample size formula.
    """
    p1 = req.base_ctr
    p2 = p1 * (1 + req.effect_size)
    z_alpha = 1.96   # 95% confidence
    z_beta  = 0.84   # 80% power

    pooled = (p1 + p2) / 2
    if p2 == p1:
        return {"error": "Effect size too small"}

    n_per_variant = math.ceil(
        2 * pooled * (1 - pooled) * (z_alpha + z_beta) ** 2
        / (p2 - p1) ** 2
    )
    total_needed = n_per_variant * 2
    days_needed  = math.ceil(total_needed / req.daily_visitors)

    # Build simulated cumulative conversion curves
    days = min(days_needed + 3, 21)
    per_day = req.daily_visitors / 2           # split 50/50
    chart_labels = [f"Day {i}" for i in range(days + 1)]
    chart_a, chart_b = [0], [0]

    for _ in range(days):
        daily_conv_a = random.gauss(per_day * p1, per_day * p1 * 0.08)
        daily_conv_b = random.gauss(per_day * p2, per_day * p2 * 0.08)
        chart_a.append(round(chart_a[-1] + max(0, daily_conv_a)))
        chart_b.append(round(chart_b[-1] + max(0, daily_conv_b)))

    return {
        "days_to_significance": days_needed,
        "sample_per_variant":   n_per_variant,
        "expected_lift_pct":    round(req.effect_size * 100, 1),
        "variant_b_rate":       round(p2 * 100, 1),
        "chart": {
            "labels":    chart_labels,
            "variant_a": chart_a,
            "variant_b": chart_b,
        },
    }


@app.post("/api/generate-variants")
async def generate_variants(req: GenerateVariantsRequest):
    """
    Stream AI-generated variant suggestions using Claude.
    Returns SSE (Server-Sent Events) so the UI can show real-time typing.
    """
    prompt = f"""You are a senior UX researcher and conversion rate optimisation expert.

A product team wants to A/B test the following UI component:

Test name:       {req.test_name}
Component type:  {req.component_type}
Primary metric:  {req.metric}
Traffic split:   {req.traffic_split}
Control (A):     {req.control_description}
Hypothesis:      {req.hypothesis}

Your task:
1. Briefly explain your reasoning (2–3 sentences).
2. Confirm or refine the control variant A description.
3. Write a concrete challenger variant B description (visual properties, copy, sizing, colour, micro-interactions).
4. Give a JSON block at the end in this exact format — no extra keys:

```json
{{
  "variant_a": {{
    "label": "Control",
    "description": "...",
    "predicted_ctr_change": "baseline"
  }},
  "variant_b": {{
    "label": "Challenger",
    "description": "...",
    "predicted_ctr_change": "+X% to +Y%"
  }},
  "agent_rationale": "One-sentence reason B should outperform A."
}}
```"""

    async def event_stream():
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                # Send each chunk as an SSE event
                yield f"data: {json.dumps({'chunk': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/insights")
async def get_insights(req: InsightRequest):
    """Ask Claude to analyse completed test results and recommend next steps."""
    uplift = round((req.variant_b_rate - req.variant_a_rate) / req.variant_a_rate * 100, 1)

    prompt = f"""You are a data-driven UX analyst. A/B test results:

Test:       {req.test_name}
Variant A:  {req.variant_a_rate}% conversion
Variant B:  {req.variant_b_rate}% conversion  ({'+' if uplift >= 0 else ''}{uplift}% relative lift)
Confidence: {req.confidence}%
Visitors:   {req.visitors:,}
Duration:   {req.duration_days} days

Provide a concise analysis (3 short paragraphs):
1. What the result means and whether to ship Variant B.
2. Likely UX reason for the difference.
3. One concrete follow-up test to run next.

Be direct. No bullet lists — prose only. Max 120 words total."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"insight": response.content[0].text}


# ── Run directly ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
