"""End-to-end example: audio file → speaker-attributed Markdown.

Demonstrates the modern pipeline:
  1. convert audio to canonical WAV
  2. transcribe with Whisper (via the on-disk cache for repeat runs)
  3. diarize with Senko
  4. merge + collapse
  5. polish paragraphs with an LLM + optional glossary
  6. (optional) ask an LLM to map SPEAKER_XX labels to real names
  7. render Markdown

Run with:
    uv run python examples/basic_usage.py /path/to/audio.m4a
    uv run python examples/basic_usage.py /path/to/audio.m4a out.md "Jason:ic,Leo:manager"
"""

import sys
import tempfile
from pathlib import Path

from mac_meeting_transcriber import (
    collapse_consecutive,
    diarize_audio,
    filter_hallucinations,
    identify_speakers,
    merge_transcript_with_speakers,
    parse_candidates,
    polish_transcript,
    raw_transcribe,
    render_markdown,
    resolve_glossary,
    resolve_llm_config,
    save_markdown,
)
from mac_meeting_transcriber.audio import to_canonical_wav
from mac_meeting_transcriber.cache import (
    compute_cache_key,
    load_transcript_cache,
    save_transcript_cache,
)


def transcribe_meeting(
    audio_path: Path,
    output_path: Path | None = None,
    speakers_spec: str | None = None,
) -> str:
    model = "mlx-community/whisper-large-v3-mlx"
    language: str | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        to_canonical_wav(audio_path, wav_path)

        cache_key = compute_cache_key(audio_path, model, language)
        raw = load_transcript_cache(cache_key)
        if raw is None:
            raw = raw_transcribe(wav_path, model=model, language=language)
            save_transcript_cache(cache_key, raw)

        transcript = filter_hallucinations(raw)
        speakers = diarize_audio(wav_path)

        attributed = collapse_consecutive(
            merge_transcript_with_speakers(transcript, speakers)
        )

    llm = resolve_llm_config()
    glossary_text, _ = resolve_glossary()
    try:
        polish_result = polish_transcript(attributed, llm, glossary=glossary_text)
        attributed = polish_result.segments
    except Exception as exc:
        print(f"warning: polish failed ({exc}); keeping raw transcript")

    aliases: dict[str, str] | None = None
    if speakers_spec:
        candidates = parse_candidates(speakers_spec)
        try:
            result = identify_speakers(attributed, candidates, llm)
            aliases = result.mapping
        except Exception as exc:
            print(f"warning: LLM speaker ID failed ({exc}); keeping SPEAKER_XX labels")

    markdown = render_markdown(
        attributed,
        audio_name=audio_path.stem,
        speaker_aliases=aliases,
    )

    if output_path is not None:
        save_markdown(markdown, output_path)

    return markdown


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: python basic_usage.py <audio-file> [output.md] [\"Name:role,Name:role\"]"
        )
        sys.exit(1)

    audio = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else None
    speakers_spec = sys.argv[3] if len(sys.argv) > 3 else None

    md = transcribe_meeting(audio, out, speakers_spec)
    if out is None:
        print(md)
    else:
        print(f"Wrote {out}")
