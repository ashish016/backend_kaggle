"""
YouTube Data API v3 uploader.

Performs an OAuth 2.0 device- or browser-flow authentication on first use,
then uploads the rendered MP4 as a YouTube video with title/description/tags
from the scriptwriter output.

NOTE: On Kaggle, the browser flow can't be used (no display). Use the local
OAuth flow once to generate `youtube_token.json`, then upload that file as a
Kaggle dataset and mount it. We provide a small CLI helper for that.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from config import (
    YOUTUBE_CLIENT_SECRETS_FILE,
    YOUTUBE_PRIVACY,
    YOUTUBE_TOKEN_FILE,
)

logger = logging.getLogger("youtube_uploader")


SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _build_client():
    """Authenticate and return an authenticated YouTube service client."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
    except ImportError as exc:
        raise RuntimeError(
            "YouTube upload libs not installed. Run:\n"
            "pip install google-api-python-client google-auth-oauthlib google-auth"
        ) from exc

    if not YOUTUBE_CLIENT_SECRETS_FILE or not Path(YOUTUBE_CLIENT_SECRETS_FILE).exists():
        raise RuntimeError(
            f"YouTube client secret not found at {YOUTUBE_CLIENT_SECRETS_FILE}. "
            "Download it from Google Cloud Console and place it there."
        )

    creds: Optional[Credentials] = None
    if YOUTUBE_TOKEN_FILE and Path(YOUTUBE_TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(str(YOUTUBE_TOKEN_FILE), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(YOUTUBE_CLIENT_SECRETS_FILE), SCOPES
        )
        # On a server / Kaggle, use run_console (deprecated but works without browser).
        # Locally, use run_local_server for a nicer flow.
        try:
            creds = flow.run_local_server(port=0)
        except Exception:  # noqa: BLE001
            logger.warning("Local server auth failed; falling back to console flow.")
            creds = flow.run_console()

    if YOUTUBE_TOKEN_FILE:
        Path(YOUTUBE_TOKEN_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(YOUTUBE_TOKEN_FILE).write_text(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def upload(
    video_path: Path,
    title: str,
    description: str,
    tags: list[str],
    *,
    privacy: Optional[str] = None,
    category_id: str = "22",  # People & Blogs
) -> Dict[str, Any]:
    """Upload a video to YouTube.

    Returns the YouTube videoId and metadata.
    """
    from googleapiclient.http import MediaFileUpload

    privacy = privacy or YOUTUBE_PRIVACY
    youtube = _build_client()

    body = {
        "snippet": {
            "title": title[:100],
            "description": (description or "")[:4900],
            "tags": tags[:500],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(video_path), mimetype="video/mp4", resumable=True, chunksize=-1
    )
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response: Optional[Dict[str, Any]] = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logger.info("YouTube upload progress: %.1f%%", status.progress() * 100)

    if "id" not in response:
        raise RuntimeError(f"YouTube upload failed: {response}")
    return {
        "video_id": response["id"],
        "url": f"https://youtu.be/{response['id']}",
        "privacy": privacy,
        "raw": response,
    }


def quota_status() -> Dict[str, Any]:
    """Return basic info about quota usage. The YouTube API doesn't expose
    remaining quota directly, so this is informational only."""
    # Could be extended to fetch channel info via youtube.channels().list
    return {
        "default_quota_per_day": 10000,
        "upload_cost_per_video": 1600,
        "note": "Quota resets daily at 00:00 PT.",
    }
