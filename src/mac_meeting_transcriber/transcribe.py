"""Speech-to-text transcription via mlx-whisper.

Runs OpenAI's whisper-large-v3 natively on Apple Silicon (Metal)
through the MLX framework, which is the fastest available CPU/GPU
path for Whisper on M-series chips.
"""

from dataclasses import dataclass
from pathlib import Path

# Known hallucinations we strip out. These strings are common artifacts
# when Whisper encounters long silences or low-quality audio.
_HALLUCINATION_MARKERS = (
    "请不吝点赞",
    "訂閱",
    "转发",
    "打赏支持",
    "明镜与点点",
    "響鐘",
    "字幕由",
    "字幕组",
    "感谢观看",
    "謝謝觀看",
    "谢谢观看",
    "Subscribe to my channel",
    "Thanks for watching",
    "Please subscribe",
    "like and subscribe",
)


def _is_repetitive(text: str, coverage_threshold: float = 0.8) -> bool:
    """Detect Whisper hallucinations that repeat a phrase during silence.

    Two classes of repetition get caught:

    1. **Head-anchored periodic repetition.** The entire segment is the
       same fragment N times over, starting from position 0:
         - "呼吸呼吸"              (2-char × 2)
         - "嗯嗯嗯嗯嗯嗯"          (1-char × many)
         - "我可以做点什么吗?" × 29 (long phrase × many)

    2. **Embedded repetition-degeneration.** A short prefix of real
       content is followed by a single-character loop that dominates the
       segment — a common failure mode on silence for Chinese fine-tuned
       checkpoints like BELLE:
         - "放弃中文异数数数数数数数数数..."  (one char × 100+)
         - "还有一个弹弹弹弹弹弹弹..."        (one char × 100+)
       These slip past (1) because the head isn't itself repetitive, but
       any run of the same character covering most of the segment is an
       unambiguous decoder artifact.

    Returns True if the segment qualifies under either heuristic.
    """
    stripped = text.strip()
    n = len(stripped)
    if n < 4:
        return False

    # Class 1: head-anchored periodic repetition.
    for frag_len in range(1, n // 2 + 1):
        fragment = stripped[:frag_len]
        reps = 1
        pos = frag_len
        while pos + frag_len <= n and stripped[pos:pos + frag_len] == fragment:
            reps += 1
            pos += frag_len
        if reps >= 2 and reps * frag_len >= n * coverage_threshold:
            return True

    # Class 2: embedded single-character runs. Slide through the string
    # and look for the longest run of an identical character; if the
    # longest run alone covers most of the segment (or a big absolute
    # chunk), the segment is dominated by a loop and should be dropped.
    # Threshold pair is deliberately conservative so real content like
    # "嗯嗯嗯嗯，那我们开始吧" (short lead-in + substance) survives.
    max_run = 1
    run = 1
    for i in range(1, n):
        if stripped[i] == stripped[i - 1]:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1
    if max_run >= 20 and max_run >= n * 0.5:
        return True

    return False


# Characters that are "content-free" padding: punctuation, whitespace,
# Chinese interjections. A segment made of only these tokens is noise.
_NOISE_CHARS = set(" \t\n!?.,;:。，、？！：；…—~·'\"“”‘’()[]{}<>《》「」『』")


def _is_trivial(text: str) -> bool:
    """Segment has no meaningful content — just punctuation or a stray mark."""
    cleaned = "".join(ch for ch in text if ch not in _NOISE_CHARS)
    return len(cleaned) <= 1


@dataclass
class TranscriptSegment:
    """One contiguous transcribed chunk, produced by Whisper."""
    start: float
    end: float
    text: str


def is_hallucination(text: str) -> bool:
    """Detect common Whisper-artifact strings.

    Public so downstream stages (e.g. post-merge filtering in
    ``merge.collapse_consecutive``) and user code can call it directly
    without reaching into a private symbol.
    """
    t = text.strip()
    if not t:
        return True
    if _is_trivial(t):
        return True
    if any(marker in t for marker in _HALLUCINATION_MARKERS):
        return True
    if _is_repetitive(t):
        return True
    return False


def raw_transcribe(
    audio_path: str | Path,
    model: str = "mlx-community/whisper-large-v3-mlx",
    language: str | None = None,
    initial_prompt: str | None = None,
    verbose: bool = False,
) -> list[TranscriptSegment]:
    """Transcribe without filtering hallucinations. Intended for caching.

    ``initial_prompt`` is fed to Whisper as a decoding bias. Use it to:
      - Anchor domain vocabulary (proper nouns, acronyms, technical terms)
        so Whisper picks the right homophone / casing.
      - Hint the expected language mix for a multilingual meeting, e.g.
        "This is a Mandarin (Putonghua) technical meeting mixed with
        English terms. Transcribe each speaker's words verbatim; do NOT
        translate." — this alone dramatically reduces the failure mode
        where Whisper silently translates Mandarin audio into English
        when language detection is confused.
    """
    import mlx_whisper

    kwargs = {
        "path_or_hf_repo": model,
        "word_timestamps": False,
        "verbose": verbose,
    }
    if language is not None:
        kwargs["language"] = language
    if initial_prompt is not None:
        kwargs["initial_prompt"] = initial_prompt

    result = mlx_whisper.transcribe(str(audio_path), **kwargs)

    return [
        TranscriptSegment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"].strip(),
        )
        for seg in result["segments"]
    ]


def filter_hallucinations(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Remove segments whose text looks like Whisper hallucination noise."""
    return [s for s in segments if not is_hallucination(s.text)]


def transcribe_audio(
    audio_path: str | Path,
    model: str = "mlx-community/whisper-large-v3-mlx",
    language: str | None = None,
    initial_prompt: str | None = None,
    verbose: bool = False,
) -> list[TranscriptSegment]:
    """Transcribe an audio file and strip known hallucinations.

    Convenience wrapper around ``raw_transcribe`` + ``filter_hallucinations``
    for quick one-shot use. **Does not use the on-disk transcript cache** —
    every call re-runs Whisper end-to-end.

    For iterative workflows, prefer:

        raw = load_transcript_cache(key) or raw_transcribe(...)
        transcript = filter_hallucinations(raw)

    which hits the cache on repeat runs (see ``mac_meeting_transcriber.cache``).
    """
    raw = raw_transcribe(
        audio_path,
        model=model,
        language=language,
        initial_prompt=initial_prompt,
        verbose=verbose,
    )
    return filter_hallucinations(raw)
