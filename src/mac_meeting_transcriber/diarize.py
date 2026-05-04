"""Speaker diarization via Senko.

Senko is a CoreML/ANE-accelerated pipeline (Pyannote segmentation-3.0
VAD + CAM++ embeddings + spectral clustering). On Apple M-series it
processes one hour of audio in ~8 seconds, which is ~40x faster than
Pyannote on CPU while matching its 13.5% DER on VoxConverse.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SpeakerSegment:
    """A chunk of audio attributed to a single speaker."""
    start: float
    end: float
    speaker: str


def diarize_audio(wav_path: str | Path, warmup: bool = False, quiet: bool = True) -> list[SpeakerSegment]:
    """Run speaker diarization on a 16kHz mono 16-bit WAV file.

    Args:
        wav_path: Path to WAV file in the required canonical format
            (see mac_meeting_transcriber.audio.to_canonical_wav).
        warmup: Warm CoreML models before timing — shaves latency off
            the first real call at the cost of a few seconds upfront.
        quiet: Suppress Senko's per-stage logs.

    Returns:
        Speaker segments ordered by start time, labelled as
        "SPEAKER_00", "SPEAKER_01", ... consistent with pyannote's
        RTTM conventions.
    """
    import senko

    diarizer = senko.Diarizer(device="auto", warmup=warmup, quiet=quiet)
    result = diarizer.diarize(str(wav_path), generate_colors=False)

    segments: list[SpeakerSegment] = []
    for seg in result["merged_segments"]:
        speaker = _normalize_speaker_label(seg["speaker"])
        segments.append(
            SpeakerSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=speaker,
            )
        )

    segments.sort(key=lambda s: s.start)
    return segments


def _normalize_speaker_label(raw) -> str:
    """Coerce Senko's speaker field into a canonical "SPEAKER_NN" string.

    Senko has shipped both integer indices and pre-formatted "SPEAKER_NN"
    strings across versions, so we handle both.
    """
    if isinstance(raw, (int, float)):
        return f"SPEAKER_{int(raw):02d}"
    text = str(raw).strip()
    if text.startswith("SPEAKER_"):
        return text
    try:
        return f"SPEAKER_{int(text):02d}"
    except ValueError:
        return text
