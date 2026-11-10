"""
Media Fetcher.

Two modes:
  - Mode A (stock): query Pexels / Pixabay APIs for short stock clips matching
    each scene's `search_query`, download them, and resize/crop to 1080x1920.
  - Mode B (diffusers): generate short AI video clips using Hugging Face
    `diffusers` (ZeroScope / Stable Video Diffusion) on the local GPU.

All returned clip paths are 1080x1920 H.264 mp4s ready for FFmpeg stitching.
"""

from __future__ import annotations

import asyncio
import logging
import os
import os.path
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config import (
    ASSET_DIR,
    DIFFUSERS_FPS,
    DIFFUSERS_NUM_FRAMES,
    DIFFUSERS_STEPS,
    HF_MODEL,
    PIXABAY_API_KEY,
    PEXELS_API_KEY,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    VISUALS_MODE,
)

logger = logging.getLogger("media_fetcher")


# ---------------------------------------------------------------------------
# Stock footage (Mode A)
# ---------------------------------------------------------------------------
async def _pexels_fetch(query: str, out_path: Path) -> Optional[Path]:
    """Fetch the first vertical-friendly stock video from Pexels matching `query`."""
    if not PEXELS_API_KEY:
        return None
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": 10, "orientation": "portrait"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://api.pexels.com/videos/search", headers=headers, params=params
        )
        resp.raise_for_status()
        data = resp.json()

    for v in data.get("videos", []):
        # Prefer HD files, prefer vertical/aspect ratios close to 9:16
        for f in v.get("video_files", []):
            if f.get("width", 0) >= 720 and f.get("height", 0) >= 1280:
                url = f["link"]
                # Download
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.get(url)
                    r.raise_for_status()
                    out_path.write_bytes(r.content)
                return out_path
    return None


async def _pixabay_fetch(query: str, out_path: Path) -> Optional[Path]:
    """Fetch a stock video from Pixabay as a fallback to Pexels."""
    if not PIXABAY_API_KEY:
        return None
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "video_type": "all",
        "per_page": 10,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get("https://pixabay.com/api/videos/", params=params)
        resp.raise_for_status()
        data = resp.json()
    for v in data.get("hits", []):
        # Try large -> medium -> small downloads
        for size_key in ("large", "medium", "small", "tiny"):
            video = v.get("videos", {}).get(size_key)
            if video and video.get("url"):
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.get(video["url"])
                    r.raise_for_status()
                    out_path.write_bytes(r.content)
                return out_path
    return None


async def _fetch_stock_clip(query: str, out_path: Path) -> Optional[Path]:
    """Try Pexels first, then Pixabay."""
    try:
        if await _pexels_fetch(query, out_path):
            return out_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pexels fetch failed for %r: %s", query, exc)
    try:
        if await _pixabay_fetch(query, out_path):
            return out_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pixabay fetch failed for %r: %s", query, exc)
    return None


# ---------------------------------------------------------------------------
# AI video generation (Mode B — Kaggle GPU only)
# ---------------------------------------------------------------------------
_DIFFUSER_PIPELINE = None


def _get_diffuser_pipeline():
    """Lazily load the Hugging Face diffusers pipeline (heavy import)."""
    global _DIFFUSER_PIPELINE
    if _DIFFUSER_PIPELINE is not None:
        return _DIFFUSER_PIPELINE
    try:
        import torch
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    except ImportError as exc:
        raise RuntimeError(
            "diffusers / torch not installed. On Kaggle, run:\n"
            "pip install diffusers transformers accelerate torch imageio imageio-ffmpeg"
        ) from exc

    logger.info("Loading diffusers model: %s", HF_MODEL)
    pipe = DiffusionPipeline.from_pretrained(
        HF_MODEL, torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    _DIFFUSER_PIPELINE = pipe
    return pipe


def _generate_diffuser_clip(query: str, out_path: Path) -> Path:
    """Synchronous diffusers generation — wrapped in run_in_executor by callers."""
    import torch
    import imageio

    pipe = _get_diffuser_pipeline()
    prompt = f"{query}, vertical 9:16, cinematic, high quality"
    negative = "low quality, blurry, watermark, text, distorted"
    logger.info("Generating diffusers clip for %r (frames=%d)", query, DIFFUSERS_NUM_FRAMES)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=DIFFUSERS_STEPS,
        num_frames=DIFFUSERS_NUM_FRAMES,
        height=576,  # zeroscope native; we upscale later
        width=324,
    )
    frames = result.frames[0]  # numpy array (T, H, W, C)
    # Save as mp4 via imageio
    tmp_mp4 = out_path.with_suffix(".raw.mp4")
    imageio.mimwrite(
        tmp_mp4, frames, fps=DIFFUSERS_FPS, codec="libx264", quality=8
    )
    # Upscale & crop to 1080x1920 with FFmpeg
    _ffmpeg_crop_to_vertical(tmp_mp4, out_path)
    tmp_mp4.unlink(missing_ok=True)
    return out_path


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------
def _ffmpeg_crop_to_vertical(src: Path, dst: Path) -> Path:
    """Scale + center-crop any video to 1080x1920 (9:16)."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920"
        ),
        "-r", "30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def fetch_clip_for_scene(
    scene: Dict[str, Any], index: int, job_dir: Path
) -> Dict[str, Any]:
    """Return a clip path for one scene.

    Falls back to a synthetic colored gradient clip if both modes fail. That
    way the pipeline always produces *something* — useful for offline testing
    on a sandbox with no API keys / GPU.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    out_path = job_dir / f"scene_{index:02d}.mp4"
    query = scene.get("search_query", "abstract motion background")

    path: Optional[Path] = None
    mode = VISUALS_MODE

    if mode in ("stock", "both"):
        path = await _fetch_stock_clip(query, out_path)

    if path is None and mode in ("diffusers", "both"):
        # Run blocking diffusers generation in a thread pool
        try:
            loop = asyncio.get_running_loop()
            path = await loop.run_in_executor(None, _generate_diffuser_clip, query, out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Diffusers failed for %r: %s", query, exc)

    if path is None:
        # Final fallback: generate a gradient clip with ffmpeg's `geq` filter.
        logger.info("Generating synthetic fallback clip for scene %d", index)
        path = _generate_fallback_clip(out_path, query)

    return {
        "scene_index": index,
        "search_query": query,
        "clip_path": str(path),
        "mode": "stock" if path == out_path and mode != "diffusers" else mode,
    }


def _generate_fallback_clip(out_path: Path, query: str) -> Path:
    """Generate a 5s gradient mp4 as an absolute fallback (no API / GPU)."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x1e1b4b:s=1080x1920:d=5:r=30",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf", (
            f"drawtext=text='{query[:30]}':"
            "fontcolor=white:fontsize=48:"
            "x=(w-text_w)/2:y=(h-text_h)/2:"
            "box=1:boxcolor=0x4338ca@0.6:boxborderw=20"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


async def fetch_clips_for_scenes(
    scenes: List[Dict[str, Any]], job_dir: Path
) -> List[Dict[str, Any]]:
    """Fetch all scene clips concurrently (with a small semaphore to avoid
    hammering the stock APIs)."""
    sem = asyncio.Semaphore(4)

    async def _wrapped(i: int, scene: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            return await fetch_clip_for_scene(scene, i, job_dir)

    results = await asyncio.gather(
        *[_wrapped(i, s) for i, s in enumerate(scenes)]
    )
    return list(results)
