"""Mac Meeting Transcriber.

Fast on-device meeting transcription with speaker diarization,
optimized for Apple Silicon.

Pipeline:
    1. mlx-whisper (large-v3) — Apple Silicon native transcription
    2. senko — CoreML/ANE-accelerated speaker diarization
    3. merge — align segments with speaker labels
    4. formatter — render Markdown meeting transcript
"""

__version__ = "0.1.0"

from .diarize import diarize_audio
from .formatter import render_markdown, save_markdown
from .identify import (
    Candidate,
    IdentificationResult,
    LLMConfig,
    identify_speakers,
    parse_candidates,
    resolve_llm_config,
)
from .merge import collapse_consecutive, merge_transcript_with_speakers
from .polish import PolishResult, polish_transcript, resolve_glossary
from .transcribe import (
    filter_hallucinations,
    is_hallucination,
    raw_transcribe,
    transcribe_audio,
)

__all__ = [
    "__version__",
    "transcribe_audio",
    "raw_transcribe",
    "filter_hallucinations",
    "is_hallucination",
    "diarize_audio",
    "merge_transcript_with_speakers",
    "collapse_consecutive",
    "render_markdown",
    "save_markdown",
    "identify_speakers",
    "parse_candidates",
    "resolve_llm_config",
    "polish_transcript",
    "resolve_glossary",
    "Candidate",
    "IdentificationResult",
    "LLMConfig",
    "PolishResult",
]
