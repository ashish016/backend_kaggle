"""
Central configuration for the YouTube Shorts AI backend.

All secrets are read from environment variables so the repo can be committed
without leaking keys. On Kaggle, use the "Add-ons -> Secrets" panel and
` KaggleTeX ` style secrets access, OR set them via os.environ in the notebook
cell before launching the server.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ---------- Paths ----------
BASE_DIR: Path = Path(__file__).resolve().parent
STATIC_DIR: Path = BASE_DIR / "static"
OUTPUT_DIR: Path = BASE_DIR / "output"
ASSET_DIR: Path = BASE_DIR / "static" / "assets"
SUBTITLE_DIR: Path = STATIC_DIR / "subtitles"

for _p in (STATIC_DIR, OUTPUT_DIR, ASSET_DIR, SUBTITLE_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------- LLM ----------
# Either Gemini or OpenAI can be used. Gemini is the default because the free
# tier is generous; OpenAI is a drop-in fallback.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()  # "gemini" | "openai"
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ---------- TTS ----------
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "edge-tts").lower()  # "edge-tts" | "elevenlabs"
ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE: str = os.getenv("ELEVENLABS_VOICE", "21m00Tcm4TlvDq8ikWAM")
EDGE_TTS_VOICE: str = os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")


# ---------- Visuals ----------
# "stock" uses Pexels/Pixabay APIs. "diffusers" uses local GPU generation.
VISUALS_MODE: str = os.getenv("VISUALS_MODE", "stock").lower()  # "stock" | "diffusers" | "both"
PEXELS_API_KEY: Optional[str] = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY: Optional[str] = os.getenv("PIXABAY_API_KEY")
HF_MODEL: str = os.getenv("HF_MODEL", "cerspense/zeroscope_v2_576w")
DIFFUSERS_STEPS: int = int(os.getenv("DIFFUSERS_STEPS", "25"))
DIFFUSERS_FPS: int = int(os.getenv("DIFFUSERS_FPS", "8"))
DIFFUSERS_NUM_FRAMES: int = int(os.getenv("DIFFUSERS_NUM_FRAMES", "24"))


# ---------- FFmpeg / Output ----------
VIDEO_WIDTH: int = 1080
VIDEO_HEIGHT: int = 1920
VIDEO_FPS: int = 30
VIDEO_CODEC: str = "libx264"
AUDIO_CODEC: str = "aac"
BACKGROUND_MUSIC_PATH: Optional[str] = os.getenv("BACKGROUND_MUSIC_PATH")  # optional mp3


# ---------- YouTube Upload ----------
YOUTUBE_CLIENT_SECRETS_FILE: Optional[str] = os.getenv(
    "YOUTUBE_CLIENT_SECRETS_FILE", str(STATIC_DIR / "youtube_client_secret.json")
)
YOUTUBE_TOKEN_FILE: Optional[str] = os.getenv(
    "YOUTUBE_TOKEN_FILE", str(STATIC_DIR / "youtube_token.json")
)
YOUTUBE_PRIVACY: str = os.getenv("YOUTUBE_PRIVACY", "private")  # public | unlisted | private


# ---------- Ngrok ----------
NGROK_AUTH_TOKEN: Optional[str] = os.getenv("NGROK_AUTH_TOKEN")
NGROK_PORT: int = int(os.getenv("NGROK_PORT", "8000"))
NGROK_REGION: str = os.getenv("NGROK_REGION", "us")  # us | eu | ap | au | sa | jp | in


# ---------- CORS ----------
# Comma-separated list of allowed origins, "*" allows all (dev only).
CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]


def as_dict() -> dict:
    """Return a safe (non-secret) summary of the config for /api/health."""
    return {
        "llm_provider": LLM_PROVIDER,
        "llm_ready": bool(GEMINI_API_KEY or OPENAI_API_KEY),
        "tts_provider": TTS_PROVIDER,
        "visuals_mode": VISUALS_MODE,
        "pexels_ready": bool(PEXELS_API_KEY),
        "pixabay_ready": bool(PIXABAY_API_KEY),
        "ngrok_configured": bool(NGROK_AUTH_TOKEN),
        "youtube_configured": bool(
            YOUTUBE_CLIENT_SECRETS_FILE and Path(YOUTUBE_CLIENT_SECRETS_FILE).exists()
        ),
        "cors_origins": CORS_ORIGINS,
        "output_dir": str(OUTPUT_DIR),
    }
