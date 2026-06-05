from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from asktell_service import ASKTELL
from hilo_service import SERVICE

DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class SettingsPayload(BaseModel):
    settings: Dict[str, Any] = Field(default_factory=dict)


class RunPayload(BaseModel):
    steps: int = 1
    settings: Dict[str, Any] = Field(default_factory=dict)


class ReadoutPayload(BaseModel):
    readout: Dict[str, Any]


class TranslatePayload(BaseModel):
    transcript: str
    model: str = "claude-opus-4-5-20251101"
    temperature: float = 0.0
    api_key: Optional[str] = None


class AskTellConfig(BaseModel):
    objective_name: str = "objective"
    goal: str = "maximize"
    parameters: list = Field(default_factory=list)
    batch_size: int = 3
    n_init: int = 5
    seed: int = 0


class AskTellTell(BaseModel):
    results: list = Field(default_factory=list)


class AskTellReadout(BaseModel):
    readout: Dict[str, Any]


class AskTellTranslate(BaseModel):
    transcript: str
    model: str = "claude-opus-4-5-20251101"
    temperature: float = 0.0
    api_key: Optional[str] = None


app = FastAPI(title="HILO UGI API", version="0.1.0")
# Allow any origin: the built frontend is served same-origin from this app, and a
# networked dev server (Vite on :5173) should also be able to reach the API.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    return SERVICE.state()


@app.post("/api/reset")
def reset(payload: Optional[SettingsPayload] = None) -> Dict[str, Any]:
    try:
        return SERVICE.reset((payload.settings if payload else {}) or {})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/run")
def run(payload: RunPayload) -> Dict[str, Any]:
    try:
        return SERVICE.run_steps(payload.steps, payload.settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/clear-readout")
def clear_readout() -> Dict[str, Any]:
    try:
        return SERVICE.clear_readout()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/api/readout")
def update_readout(payload: ReadoutPayload) -> Dict[str, Any]:
    try:
        return SERVICE.update_readout(payload.readout)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/translate-readout")
def translate_readout(payload: TranslatePayload) -> Dict[str, Any]:
    try:
        readout = SERVICE.translate_expert_text(
            transcript=payload.transcript,
            model=payload.model,
            temperature=payload.temperature,
            api_key=payload.api_key,
        )
        return {"readout": readout}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/prior-surface")
def prior_surface(n_grid: int = 48) -> Dict[str, Any]:
    try:
        return SERVICE.prior_surface(n_grid=n_grid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/prior-surfaces")
def prior_surfaces(n_grid: int = 40) -> Dict[str, Any]:
    try:
        return SERVICE.prior_surfaces(n_grid=n_grid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --- Ask-Tell: domain-agnostic manual optimizer ----------------------------
@app.get("/api/asktell/state")
def asktell_state() -> Dict[str, Any]:
    return ASKTELL.state()


@app.post("/api/asktell/init")
def asktell_init(payload: AskTellConfig) -> Dict[str, Any]:
    try:
        return ASKTELL.configure(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/asktell/tell")
def asktell_tell(payload: AskTellTell) -> Dict[str, Any]:
    try:
        return ASKTELL.tell(payload.results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/asktell/suggest")
def asktell_suggest() -> Dict[str, Any]:
    try:
        return ASKTELL.suggest_again()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/asktell/reset")
def asktell_reset() -> Dict[str, Any]:
    return ASKTELL.reset_all()


@app.put("/api/asktell/readout")
def asktell_readout(payload: AskTellReadout) -> Dict[str, Any]:
    try:
        return ASKTELL.set_readout(payload.readout)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/asktell/clear-readout")
def asktell_clear_readout() -> Dict[str, Any]:
    try:
        return ASKTELL.clear_readout()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/asktell/translate")
def asktell_translate(payload: AskTellTranslate) -> Dict[str, Any]:
    try:
        readout = ASKTELL.translate(
            transcript=payload.transcript,
            model=payload.model,
            temperature=payload.temperature,
            api_key=payload.api_key,
        )
        return {"readout": readout}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/asktell/prior-surfaces")
def asktell_prior_surfaces(n_grid: int = 40) -> Dict[str, Any]:
    try:
        return ASKTELL.prior_surfaces(n_grid=n_grid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --- static frontend (built bundle) ---------------------------------------
# Serving the compiled SPA from the same origin as the API means no proxy and
# no CORS are needed when the app is reached over the network.
if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def index() -> FileResponse:
    if not (DIST / "index.html").is_file():
        raise HTTPException(status_code=404, detail="Frontend not built. Run: cd frontend && npm run build")
    return FileResponse(DIST / "index.html")


@app.get("/{full_path:path}")
def spa_fallback(full_path: str) -> FileResponse:
    # API routes are registered above and take precedence; this only catches
    # browser navigation / static files for the single-page app.
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    candidate = (DIST / full_path).resolve()
    if candidate.is_file() and str(candidate).startswith(str(DIST.resolve())):
        return FileResponse(candidate)
    if not (DIST / "index.html").is_file():
        raise HTTPException(status_code=404, detail="Frontend not built. Run: cd frontend && npm run build")
    return FileResponse(DIST / "index.html")


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("HILO_HOST", "0.0.0.0"),
        port=int(os.environ.get("HILO_PORT", "8765")),
        reload=False,
    )
