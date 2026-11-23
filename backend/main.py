"""
FastAPI Backend Main Application
Serves REST API endpoints for energy simulation, ML forecasts, PPO inference,
and mounts static frontend web assets.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.system import router as system_router
from backend.routes.forecast import router as forecast_router
from backend.routes.simulation import router as simulation_router
from backend.routes.comparison import router as comparison_router
from backend.routes.scenario import router as scenario_router

app = FastAPI(
    title="AI-Driven Smart Energy Management System",
    description="Forecast-Assisted Proximal Policy Optimization (PPO) vs Rule-Based EMS",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path.lower()
    if path.endswith((".html", ".js", ".css", ".json")) or path == "/" or "/api/" in path:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Include API Routers
app.include_router(system_router)
app.include_router(forecast_router)
app.include_router(simulation_router)
app.include_router(comparison_router)
app.include_router(scenario_router)

# Mount static frontend directory
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
