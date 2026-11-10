"""
LLM Script Writer.

Takes a user prompt and returns a structured JSON scene breakdown using either
Google Gemini or OpenAI. The schema is:

{
  "title": str,
  "description": str,
  "hashtags": list[str],
  "scenes": [
    {"timestamp": "00:00", "speech": str, "search_query": str}
  ]
}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger("llm_scriptwriter")

SCHEMA_EXAMPLE = """{
  "title": "5 Mind-Blowing Space Facts",
  "description": "Short, punchy space facts narrated for YouTube Shorts.",
  "hashtags": ["#shorts", "#space", "#facts"],
  "scenes": [
    {"timestamp": "00:00", "speech": "Did you know a day on Venus is longer than its year?", "search_query": "venus planet rotating in space"},
    {"timestamp": "00:04", "speech": "Neutron stars can spin 600 times a second!", "search_query": "neutron star pulsar spinning"},
    {"timestamp": "00:09", "speech": "There are more stars than grains of sand on Earth.", "search_query": "starry night sky galaxy"}
  ]
}"""

SYSTEM_PROMPT = (
    "You are a scriptwriter for viral YouTube Shorts. Given a topic, you produce a "
    "JSON document with a title, description, hashtags, and 4-7 short scenes. Each "
    "scene has a 'timestamp' (MM:SS from clip start), a 'speech' line (<=12 words, "
    "conversational, hook-first), and a 'search_query' (a concrete visual description "
    "suitable for stock footage search, 2-5 words). Total runtime should be 25-45 seconds. "
    "Respond with VALID JSON ONLY, no markdown fences, no commentary."
)


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` style fences if the model included them."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce the LLM output into the expected schema with sensible defaults."""
    payload.setdefault("title", "Untitled Short")
    payload.setdefault("description", "")
    payload.setdefault("hashtags", ["#shorts"])
    if not isinstance(payload["hashtags"], list):
        payload["hashtags"] = ["#shorts"]
    # Always include #shorts
    if "#shorts" not in payload["hashtags"]:
        payload["hashtags"].insert(0, "#shorts")

    scenes = payload.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        scenes = [
            {
                "timestamp": "00:00",
                "speech": payload.get("title", "Welcome to my channel."),
                "search_query": "abstract background motion",
            }
        ]
    cleaned: List[Dict[str, Any]] = []
    for s in scenes:
        if not isinstance(s, dict):
            continue
        cleaned.append(
            {
                "timestamp": str(s.get("timestamp", "00:00")),
                "speech": str(s.get("speech", "")).strip(),
                "search_query": str(s.get("search_query", "abstract background motion")).strip(),
            }
        )
    payload["scenes"] = cleaned
    return payload


async def generate_script(
    prompt: str,
    *,
    extra_instructions: Optional[str] = None,
    num_scenes: int = 6,
) -> Dict[str, Any]:
    """Generate a structured script for a YouTube Short.

    Args:
        prompt: The user's topic / idea.
        extra_instructions: Optional style / tone modifiers.
        num_scenes: Target number of scenes (4-7 recommended).

    Returns:
        Validated dict matching the schema described in the module docstring.
    """
    user_msg = (
        f"Topic: {prompt}\n"
        f"Number of scenes: {num_scenes}\n"
        f"Extra instructions: {extra_instructions or 'none'}\n"
        f"Return ONLY valid JSON matching this exact schema:\n{SCHEMA_EXAMPLE}"
    )

    if LLM_PROVIDER == "gemini":
        return await _generate_gemini(user_msg)
    elif LLM_PROVIDER == "openai":
        return _generate_openai(user_msg)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


async def _generate_gemini(user_msg: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to env / Kaggle secrets.")

    # Imported lazily so the server can start even if the SDK is not installed
    # (e.g. in pure-frontend preview environments).
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. Run: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    logger.info("Calling Gemini model=%s", GEMINI_MODEL)
    resp = await model.generate_content_async(user_msg)
    text = resp.text or ""
    text = _strip_code_fences(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Gemini returned non-JSON: %s", text[:500])
        raise RuntimeError("Gemini did not return valid JSON") from exc
    return _validate(payload)


def _generate_openai(user_msg: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is not installed. Run: pip install openai") from exc

    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("Calling OpenAI model=%s", OPENAI_MODEL)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.8,
    )
    text = resp.choices[0].message.content or "{}"
    text = _strip_code_fences(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("OpenAI returned non-JSON: %s", text[:500])
        raise RuntimeError("OpenAI did not return valid JSON") from exc
    return _validate(payload)


# ---------------------------------------------------------------------------
# Mock provider (used when no LLM API key is configured, e.g. local preview)
# ---------------------------------------------------------------------------
async def generate_mock_script(prompt: str, num_scenes: int = 5) -> Dict[str, Any]:
    """Generate a deterministic placeholder script so the pipeline can be tested
    end-to-end without an LLM API key. Used automatically when no key is set."""
    base = prompt.strip() or "Untitled Short"
    title = base[:60]
    scenes = []
    for i in range(max(3, min(num_scenes, 6))):
        ts = i * 5
        scenes.append(
            {
                "timestamp": f"00:{ts:02d}",
                "speech": f"This is scene {i + 1} about {base[:30]}.",
                "search_query": "abstract motion background",
            }
        )
    return {
        "title": title,
        "description": f"Mock script for: {base}",
        "hashtags": ["#shorts", "#ai", "#mock"],
        "scenes": scenes,
    }
