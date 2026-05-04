"""Lightweight on-disk cache for expensive Whisper runs.

Whisper-large-v3 transcription of a one-hour file takes several
minutes. Speaker diarization, in contrast, runs in ~10 seconds. So
during iteration we cache the transcription keyed by a hash of the
(audio file content + model name + language) tuple, allowing repeated
runs to skip the slow stage.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .transcribe import TranscriptSegment


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "mac-meeting-transcriber"


def compute_cache_key(
    audio_path: Path,
    model: str,
    language: str | None,
    initial_prompt: str | None = None,
) -> str:
    """Hash audio content + key params to make a reproducible cache key.

    We hash the first & last 1 MiB plus the file size; this is far
    faster than hashing the whole file and collisions are astronomically
    unlikely for our use case.

    ``initial_prompt`` is part of the key because it meaningfully changes
    Whisper's output (vocabulary bias, language priors); two runs with
    different prompts must not share a cache entry.
    """
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update((language or "").encode("utf-8"))
    h.update(b"\0")
    h.update((initial_prompt or "").encode("utf-8"))
    h.update(b"\0")

    size = audio_path.stat().st_size
    h.update(str(size).encode("utf-8"))

    with open(audio_path, "rb") as f:
        head = f.read(1024 * 1024)
        h.update(head)
        if size > 2 * 1024 * 1024:
            f.seek(-1024 * 1024, 2)
            tail = f.read(1024 * 1024)
            h.update(tail)

    return h.hexdigest()


def load_transcript_cache(
    cache_key: str,
    cache_dir: Path | None = None,
) -> list[TranscriptSegment] | None:
    cache_dir = cache_dir or _default_cache_dir()
    cache_file = cache_dir / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return [TranscriptSegment(**item) for item in data.get("segments", [])]


def save_transcript_cache(
    cache_key: str,
    segments: list[TranscriptSegment],
    cache_dir: Path | None = None,
) -> Path:
    cache_dir = cache_dir or _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"
    payload = {
        "segments": [asdict(seg) for seg in segments],
    }
    cache_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cache_file
