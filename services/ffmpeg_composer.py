"""
FFmpeg Composer.

Stitches scene clips together, mixes TTS audio with optional background music,
and burns kinetic word-by-word ASS subtitles with custom styling.
"""

from __future__ import annotations

import logging
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import (
    BACKGROUND_MUSIC_PATH,
    OUTPUT_DIR,
    SUBTITLE_DIR,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)

logger = logging.getLogger("ffmpeg_composer")


# ---------------------------------------------------------------------------
# ASS subtitle generation with kinetic word highlighting
# ---------------------------------------------------------------------------
@dataclass
class SubtitleStyle:
    font_name: str = "Inter"
    font_size: int = 18
    primary_color: str = "&H00FFFFFF"   # white (ASS uses &H00BBGGRR)
    secondary_color: str = "&H000000FF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000"
    highlight_color: str = "&H0000FFFF"  # yellow (BGR)
    bold: bool = True
    italic: bool = False
    outline_width: int = 3
    shadow: int = 2
    alignment: int = 2  # bottom center
    margin_v: int = 100
    margin_l: int = 80
    margin_r: int = 80


def _hex_to_ass_color(hex_color: str) -> str:
    """Convert a #RRGGBB hex string into an ASS &H00BBGGRR color string."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "&H00FFFFFF"
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H00{b.upper()}{g.upper()}{r.upper()}"


def build_ass_subtitle(
    boundaries: List[dict],
    out_path: Path,
    style: SubtitleStyle,
) -> Path:
    """Write an ASS subtitle file with kinetic word-by-word highlighting.

    We implement highlighting using two simultaneous dialogue lines:
      - A "base" line showing the full sentence in `primary_color`.
      - A "active" line clipped to the active word's x-extent using
        `\\clip(x1,y1,x2,y2)` and drawn in `highlight_color`.

    Because computing pixel-exact clip rectangles requires measuring font widths
    (which ffmpeg's libass won't expose), we approximate by using the `\\k` karaoke
    effect: each word gets a karaoke timing matching its start/end time, and the
    ASS engine will color the "active" word with `secondary_color` automatically.

    This gives reliable word-by-word highlighting across all libass builds.
    """
    style.primary_color = _hex_to_ass_color(style.primary_color if style.primary_color.startswith("&") else "#FFFFFF")
    style.highlight_color = _hex_to_ass_color(style.highlight_color if style.highlight_color.startswith("&") else "#FFFF00")
    style.outline_color = _hex_to_ass_color(style.outline_color if style.outline_color.startswith("&") else "#000000")

    header = textwrap.dedent(
        f"""
        [Script Info]
        ScriptType: v4.00+
        PlayResX: {VIDEO_WIDTH}
        PlayResY: {VIDEO_HEIGHT}
        ScaledBorderAndShadow: yes
        WrapStyle: 2

        [V4+ Styles]
        Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        Style: Base,{style.font_name},{style.font_size},{style.primary_color},{style.highlight_color},{style.outline_color},{style.back_color},{'1' if style.bold else '0'},{'1' if style.italic else '0'},0,0,100,100,0,0,1,{style.outline_width},{style.shadow},{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},1

        [Events]
        Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        """
    ).strip() + "\n"

    # Group boundaries by scene_index for one sentence per dialogue line.
    lines: list[str] = []
    if not boundaries:
        # Empty subtitle file
        out_path.write_text(header, encoding="utf-8")
        return out_path

    # Group words into sentences by scene_index
    by_scene: dict[int, list[dict]] = {}
    for b in boundaries:
        by_scene.setdefault(b["scene_index"], []).append(b)

    for scene_idx, words in by_scene.items():
        if not words:
            continue
        start_s = words[0]["start"]
        end_s = words[-1]["end"]
        # Build karaoke-tagged text: \\k durations are in centiseconds (1/100 s)
        # for each word. The sum of k durations should equal the line duration.
        text_parts = []
        for w in words:
            dur_cs = max(1, int((w["end"] - w["start"]) * 100))
            text_parts.append(f"\\k{dur_cs}{w['text']}")
        karaoke_text = " ".join(text_parts)
        # Wrap into the sentence-level dialogue line
        line = (
            f"Dialogue: 0,"
            f"{_ass_time(start_s)},{_ass_time(end_s)},"
            f"Base,,0,0,0,,"
            f"{karaoke_text}"
        )
        lines.append(line)

    body = "\n".join(lines) + "\n"
    out_path.write_text(header + body, encoding="utf-8")
    return out_path


def _ass_time(seconds: float) -> str:
    """Format seconds as H:MM:SS.cs (centiseconds, ASS spec)."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------
# Video composition
# ---------------------------------------------------------------------------
def _concat_clips(clip_paths: List[Path], concat_list_path: Path) -> None:
    """Write an ffmpeg concat demuxer list file."""
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            # ffmpeg concat demuxer requires forward slashes and absolute paths
            # Workaround: use relative file names if same dir.
            f.write(f"file '{p.resolve().as_posix()}'\n")


def compose_video(
    clip_paths: List[Path],
    tts_audio_path: Path,
    subtitle_path: Path,
    out_path: Path,
    *,
    background_music_path: Optional[Path] = None,
    music_volume: float = 0.10,
) -> Path:
    """Stitch clips, attach audio (TTS + optional background music), and burn
    the ASS subtitles. Output: H.264 / AAC MP4 at 1080x1920 30fps.

    Args:
        clip_paths: ordered list of scene clip paths.
        tts_audio_path: merged TTS mp3 path.
        subtitle_path: ASS subtitle path (already styled).
        out_path: final mp4 destination.
        background_music_path: optional background music mp3.
        music_volume: 0.0 - 1.0 music gain relative to TTS.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Concatenate scene clips into a single vertical video.
    concat_list = out_path.parent / "concat_list.txt"
    _concat_clips(clip_paths, concat_list)
    concat_video = out_path.parent / "concat_video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(concat_video),
        ],
        check=True,
        capture_output=True,
    )

    # 2) Build the audio graph: TTS always; mix in music if provided.
    audio_filter = "anullsrc=channel_layout=stereo:sample_rate=44100"
    inputs = ["-i", str(concat_video), "-i", str(tts_audio_path)]
    audio_input_index = 1

    if background_music_path and Path(background_music_path).exists():
        inputs += ["-i", str(background_music_path)]
        audio_input_index = 1  # TTS index in this build is still 1
        music_index = 2
        audio_filter = (
            f"[{music_index}:a]volume={music_volume},aloop=loop=-1:size=2e09[bgm];"
            f"[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
        amap = "[aout]"
    else:
        audio_filter = "[1:a]aformat=channel_layouts=stereo[aout]"
        amap = "[aout]"

    # 3) Final pass: take concat video, mix audio, burn ASS subtitles, scale to 1080x1920.
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-vf", (
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},"
            f"ass='{subtitle_path.resolve().as_posix()}'"
        ),
        "-map", "0:v",
        "-map", amap,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(out_path),
    ]
    # The filter graph needs -filter_complex when there are multiple audio inputs
    if background_music_path and Path(background_music_path).exists():
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", audio_filter,
            "-vf", (
                f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},"
                f"ass='{subtitle_path.resolve().as_posix()}'"
            ),
            "-map", "0:v",
            "-map", amap,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path),
        ]

    logger.info("Running final ffmpeg compose: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)

    # Clean up intermediates
    for f in (concat_video, concat_list):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    return out_path


# ---------------------------------------------------------------------------
# Style bridge from request payload -> SubtitleStyle
# ---------------------------------------------------------------------------
def style_from_payload(payload: dict) -> SubtitleStyle:
    s = SubtitleStyle()
    if "font_name" in payload:
        s.font_name = payload["font_name"]
    if "font_size" in payload:
        s.font_size = int(payload["font_size"])
    if "primary_color" in payload:
        s.primary_color = payload["primary_color"]  # hex like #FFFFFF
    if "highlight_color" in payload:
        s.highlight_color = payload["highlight_color"]
    if "outline_color" in payload:
        s.outline_color = payload["outline_color"]
    if "bold" in payload:
        s.bold = bool(payload["bold"])
    if "italic" in payload:
        s.italic = bool(payload["italic"])
    if "alignment" in payload:
        s.alignment = int(payload["alignment"])
    if "margin_v" in payload:
        s.margin_v = int(payload["margin_v"])
    return s
