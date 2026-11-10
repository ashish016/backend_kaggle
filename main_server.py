"""
FastAPI main server for the YouTube Shorts AI backend.

Run on Kaggle:
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public = ngrok.connect(8000)
    print("Public URL:", public.public_url)

Then:
    uvicorn main_server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /api/health             -> { ok: True, config: {...} }
    POST /api/generate            -> kicks off a generation job; returns {job_id}
    GET  /api/jobs/{job_id}       -> { status, progress, stage, error, video_path? }
    GET  /api/jobs                -> list all jobs
    GET  /api/output/{filename}   -> streams the rendered mp4
    POST /api/upload/{job_id}      -> uploads the rendered video to YouTube
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# Make `services` and `config` importable when run as `uvicorn main_server:app`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from services import (
    ffmpeg_composer,
    llm_scriptwriter,
    media_fetcher,
    tts_generator,
    youtube_uploader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_server")


# ---------------------------------------------------------------------------
# Job registry (in-memory; sufficient for single-instance Kaggle use)
# ---------------------------------------------------------------------------
class JobStore:
    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create(self, payload: Dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:12]
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "init",
            "progress": 0,
            "error": None,
            "payload": payload,
            "script": None,
            "tts": None,
            "clips": None,
            "video_path": None,
            "youtube": None,
            "created_at": asyncio.get_event_loop().time(),
        }
        return job_id

    def update(self, job_id: str, **kwargs: Any) -> None:
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def all(self) -> list[Dict[str, Any]]:
        return list(self.jobs.values())


JOBS = JobStore()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)
    num_scenes: int = Field(6, ge=3, le=10)
    voice: Optional[str] = None
    visuals_mode: Optional[str] = None  # overrides config.VISUALS_MODE
    auto_upload_youtube: bool = False
    add_background_music: bool = False
    extra_instructions: Optional[str] = None

    # Subtitle styling
    subtitle_font: str = "Inter"
    subtitle_font_size: int = 18
    subtitle_primary_color: str = "#FFFFFF"
    subtitle_highlight_color: str = "#FFFF00"
    subtitle_outline_color: str = "#000000"
    subtitle_bold: bool = True
    subtitle_italic: bool = False
    subtitle_alignment: int = 2  # bottom center
    subtitle_margin_v: int = 100


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optionally open an Ngrok tunnel so the public URL is logged on boot."""
    if config.NGROK_AUTH_TOKEN:
        try:
            from pyngrok import ngrok

            ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)
            tunnel = ngrok.connect(config.NGROK_PORT, "http", region=config.NGROK_REGION)
            logger.info("Ngrok public URL: %s", tunnel.public_url)
            app.state.ngrok_url = tunnel.public_url
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ngrok tunnel failed: %s", exc)
            app.state.ngrok_url = None
    else:
        logger.warning(
            "NGROK_AUTH_TOKEN not set. The server will only be reachable on localhost."
        )
        app.state.ngrok_url = None

    yield


app = FastAPI(
    title="YouTube Shorts AI API",
    version="1.0.0",
    description="GPU-accelerated YouTube Shorts generation backend.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "config": config.as_dict(),
        "ngrok_url": getattr(app.state, "ngrok_url", None),
    }


@app.get("/api/ngrok")
async def ngrok_url() -> dict:
    return {"url": getattr(app.state, "ngrok_url", None)}


# ---------------------------------------------------------------------------
# Generation pipeline (async background task)
# ---------------------------------------------------------------------------
async def _run_pipeline(job_id: str, req: GenerateRequest) -> None:
    """The full multi-stage pipeline. Updates JOBS[job_id] at each step."""
    try:
        job_dir = config.OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Stage 1: Script ----------
        JOBS.update(job_id, status="running", stage="scripting", progress=5)
        try:
            if config.LLM_PROVIDER == "gemini" and not config.GEMINI_API_KEY:
                logger.info("No LLM key; using mock scriptwriter.")
                script = await llm_scriptwriter.generate_mock_script(req.prompt, req.num_scenes)
            elif config.LLM_PROVIDER == "openai" and not config.OPENAI_API_KEY:
                logger.info("No LLM key; using mock scriptwriter.")
                script = await llm_scriptwriter.generate_mock_script(req.prompt, req.num_scenes)
            else:
                script = await llm_scriptwriter.generate_script(
                    req.prompt,
                    extra_instructions=req.extra_instructions,
                    num_scenes=req.num_scenes,
                )
        except Exception as exc:
            logger.warning("LLM failed (%s); falling back to mock.", exc)
            script = await llm_scriptwriter.generate_mock_script(req.prompt, req.num_scenes)
        JOBS.update(job_id, progress=15, script=script)

        # ---------- Stage 2: TTS ----------
        JOBS.update(job_id, stage="voice_synthesis", progress=25)
        tts = await tts_generator.generate_for_script(
            scenes=script["scenes"],
            out_dir=job_dir,
            job_id=job_id,
            voice=req.voice,
        )
        JOBS.update(job_id, progress=45, tts={"audio_path": tts["audio_path"], "duration": tts["duration"]})

        # ---------- Stage 3: Visuals ----------
        JOBS.update(job_id, stage="asset_fetching", progress=55)
        if req.visuals_mode in ("stock", "diffusers", "both"):
            media_fetcher.VISUALS_MODE = req.visuals_mode  # per-request override
        clips = await media_fetcher.fetch_clips_for_scenes(script["scenes"], job_dir)
        JOBS.update(job_id, progress=70, clips=[c["clip_path"] for c in clips])

        # ---------- Stage 4: Subtitles ----------
        JOBS.update(job_id, stage="subtitle_rendering", progress=78)
        style = ffmpeg_composer.SubtitleStyle(
            font_name=req.subtitle_font,
            font_size=req.subtitle_font_size,
            primary_color=req.subtitle_primary_color,
            highlight_color=req.subtitle_highlight_color,
            outline_color=req.subtitle_outline_color,
            bold=req.subtitle_bold,
            italic=req.subtitle_italic,
            alignment=req.subtitle_alignment,
            margin_v=req.subtitle_margin_v,
        )
        ass_path = ffmpeg_composer.build_ass_subtitle(
            boundaries=tts["boundaries"],
            out_path=config.SUBTITLE_DIR / f"{job_id}.ass",
            style=style,
        )

        # ---------- Stage 5: FFmpeg compose ----------
        JOBS.update(job_id, stage="rendering", progress=85)
        out_path = job_dir / f"{job_id}.mp4"
        ffmpeg_composer.compose_video(
            clip_paths=[Path(c["clip_path"]) for c in clips],
            tts_audio_path=Path(tts["audio_path"]),
            subtitle_path=ass_path,
            out_path=out_path,
            background_music_path=Path(config.BACKGROUND_MUSIC_PATH)
            if (req.add_background_music and config.BACKGROUND_MUSIC_PATH)
            else None,
        )
        JOBS.update(job_id, progress=95, video_path=str(out_path))

        # ---------- Stage 6: Optional YouTube upload ----------
        if req.auto_upload_youtube:
            JOBS.update(job_id, stage="publishing", progress=98)
            try:
                yt = youtube_uploader.upload(
                    video_path=out_path,
                    title=script["title"],
                    description=script["description"],
                    tags=script["hashtags"],
                )
                JOBS.update(job_id, youtube=yt)
            except Exception as exc:  # noqa: BLE001
                logger.warning("YouTube upload failed: %s", exc)
                JOBS.update(job_id, youtube={"error": str(exc)})

        JOBS.update(job_id, status="completed", stage="done", progress=100)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed for job %s", job_id)
        JOBS.update(job_id, status="failed", stage="error", error=str(exc), progress=100)


@app.post("/api/generate")
async def generate(req: GenerateRequest, bg: BackgroundTasks) -> dict:
    job_id = JOBS.create(req.model_dump())
    bg.add_task(_run_pipeline, job_id, req)
    return {"job_id": job_id, "status": "queued", "poll": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs")
async def list_jobs() -> list[dict]:
    return JOBS.all()


@app.get("/api/output/{job_id}")
async def download_output(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job or not job.get("video_path"):
        raise HTTPException(status_code=404, detail="Output not ready")
    path = Path(job["video_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.post("/api/upload/{job_id}")
async def trigger_youtube_upload(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job or not job.get("video_path"):
        raise HTTPException(status_code=400, detail="No rendered video for this job")
    if not job.get("script"):
        raise HTTPException(status_code=400, detail="No script available for upload metadata")
    try:
        result = youtube_uploader.upload(
            video_path=Path(job["video_path"]),
            title=job["script"]["title"],
            description=job["script"]["description"],
            tags=job["script"]["hashtags"],
        )
        JOBS.update(job_id, youtube=result)
        return result
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/youtube/quota")
async def youtube_quota() -> dict:
    return youtube_uploader.quota_status()


# ---------------------------------------------------------------------------
# Quick local run: `python main_server.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", str(config.NGROK_PORT)))
    uvicorn.run("main_server:app", host="0.0.0.0", port=port, reload=False)
