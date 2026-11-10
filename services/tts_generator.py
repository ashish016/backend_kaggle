"""
Text-to-Speech generator.

Primary: edge-tts (free, neural Microsoft Edge TTS, supports word boundaries).
Fallback: ElevenLabs API.

Output: MP3 file + a list of word-level timestamps used by the subtitle burner.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from config import (
    EDGE_TTS_VOICE,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE,
    TTS_PROVIDER,
    OUTPUT_DIR,
)

logger = logging.getLogger("tts_generator")


@dataclass
class WordBoundary:
    """A single word with its start/end time in seconds."""

    text: str
    start: float  # seconds
    end: float    # seconds

    def to_dict(self) -> dict:
        return asdict(self)


async def _edge_tts(text: str, out_path: Path, voice: str) -> List[WordBoundary]:
    """Generate TTS using edge-tts and return word boundaries.

    edge-tts emits WordBoundary events with offset in 100-nanosecond units.
    """
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice)
    boundaries: List[WordBoundary] = []

    # We must iterate over the stream to receive WordBoundary events.
    with open(out_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # offset and duration are in 100-nanosecond units (HNS)
                start_s = chunk["offset"] / 10_000_000
                duration_s = chunk["duration"] / 10_000_000
                word = chunk["text"].strip()
                if word:
                    boundaries.append(
                        WordBoundary(text=word, start=start_s, end=start_s + duration_s)
                    )

    if not boundaries:
        # Fallback: estimate even-spaced words based on audio length.
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(out_path)
            duration = len(audio) / 1000.0
            words = text.split()
            per = duration / max(len(words), 1)
            for i, w in enumerate(words):
                boundaries.append(WordBoundary(text=w, start=i * per, end=(i + 1) * per))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not estimate word timings: %s", exc)

    return boundaries


async def _elevenlabs(text: str, out_path: Path, voice: str) -> List[WordBoundary]:
    """Generate TTS using ElevenLabs. ElevenLabs free tier doesn't return word
    timestamps, so we estimate them from the audio length evenly."""
    import httpx
    from pydub import AudioSegment

    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

    # Estimate even word timings
    audio = AudioSegment.from_file(out_path)
    duration = len(audio) / 1000.0
    words = text.split()
    per = duration / max(len(words), 1)
    return [WordBoundary(text=w, start=i * per, end=(i + 1) * per) for i, w in enumerate(words)]


async def generate(
    text: str,
    out_dir: Optional[Path] = None,
    *,
    voice: Optional[str] = None,
    job_id: str = "tts",
) -> dict:
    """Synthesize speech for a single text block.

    Returns:
        {
            "audio_path": str,
            "duration": float (seconds),
            "boundaries": list[dict],
        }
    """
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / f"{job_id}.mp3"

    text = text.strip()
    if not text:
        raise ValueError("TTS text is empty")

    provider = TTS_PROVIDER
    if provider == "elevenlabs" and not ELEVENLABS_API_KEY:
        logger.warning("ElevenLabs configured but no API key; falling back to edge-tts")
        provider = "edge-tts"

    if provider == "edge-tts":
        boundaries = await _edge_tts(text, audio_path, voice or EDGE_TTS_VOICE)
    elif provider == "elevenlabs":
        boundaries = await _elevenlabs(text, audio_path, voice or ELEVENLABS_VOICE)
    else:
        raise ValueError(f"Unknown TTS provider: {provider!r}")

    # Compute total duration
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0
    except Exception:  # noqa: BLE001
        duration = boundaries[-1].end if boundaries else 0.0

    return {
        "audio_path": str(audio_path),
        "duration": round(duration, 3),
        "boundaries": [b.to_dict() for b in boundaries],
    }


async def generate_for_script(
    scenes: list[dict],
    out_dir: Path,
    job_id: str,
    voice: Optional[str] = None,
) -> dict:
    """Generate TTS for each scene's `speech`, concatenating into one MP3.

    Returns:
        {
            "audio_path": str,        # single merged mp3
            "duration": float,
            "boundaries": list[dict],  # word boundaries with scene_index
        }
    """
    import pydub
    from pydub import AudioSegment

    merged: AudioSegment = AudioSegment.empty()
    all_boundaries: list[dict] = []
    cursor = 0.0

    for i, scene in enumerate(scenes):
        speech = scene.get("speech", "").strip()
        if not speech:
            continue
        per = await generate(speech, out_dir, voice=voice, job_id=f"{job_id}_scene_{i}")
        seg = AudioSegment.from_file(per["audio_path"])
        merged += seg
        # Shift boundaries by current cursor and tag with scene_index
        for b in per["boundaries"]:
            all_boundaries.append(
                {
                    "text": b["text"],
                    "start": round(b["start"] + cursor, 3),
                    "end": round(b["end"] + cursor, 3),
                    "scene_index": i,
                }
            )
        cursor += per["duration"]

    final_path = out_dir / f"{job_id}_tts.mp3"
    merged.export(final_path, format="mp3")

    return {
        "audio_path": str(final_path),
        "duration": round(len(merged) / 1000.0, 3),
        "boundaries": all_boundaries,
    }
