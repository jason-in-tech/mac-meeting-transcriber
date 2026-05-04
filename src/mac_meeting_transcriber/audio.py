"""Audio preprocessing utilities.

Senko requires 16 kHz mono 16-bit WAV input. This module handles
converting arbitrary audio formats (m4a, mp3, mp4, etc.) into that
canonical form using ffmpeg, which is provided via `static-ffmpeg`
so no system-level install is required.
"""

import subprocess
import tempfile
from pathlib import Path

import static_ffmpeg

_FFMPEG_INITIALIZED = False


def _ensure_ffmpeg():
    global _FFMPEG_INITIALIZED
    if not _FFMPEG_INITIALIZED:
        static_ffmpeg.add_paths()
        _FFMPEG_INITIALIZED = True


def to_canonical_wav(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convert any audio file into 16 kHz mono 16-bit PCM WAV.

    Args:
        input_path: Source audio (m4a, mp3, mp4, wav, ...).
        output_path: Target WAV path. If None, a temp file is used.

    Returns:
        Path to the converted WAV.
    """
    _ensure_ffmpeg()

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Audio not found: {input_path}")

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        output_path = Path(tmp.name)
    else:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-vn",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    return output_path
