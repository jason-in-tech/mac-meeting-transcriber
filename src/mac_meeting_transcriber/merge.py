"""Align Whisper transcript segments with Senko speaker segments.

Whisper produces text segments with timestamps. Senko produces
speaker segments with timestamps. Here we attribute each transcript
segment to the speaker who held the floor during the majority of that
segment's time range.
"""

from dataclasses import dataclass, replace

from .diarize import SpeakerSegment
from .transcribe import TranscriptSegment, is_hallucination


@dataclass
class AttributedSegment:
    """A transcript segment with its assigned speaker label."""
    start: float
    end: float
    text: str
    speaker: str


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _attribute_speaker(
    seg: TranscriptSegment,
    speaker_segments: list[SpeakerSegment],
) -> str:
    """Pick the speaker with the most time overlap with `seg`.

    Falls back to nearest-in-time if a transcript segment happens to
    land entirely inside a gap Senko didn't label (rare, but possible
    when Whisper's VAD and Senko's VAD disagree slightly).
    """
    best_speaker = "UNKNOWN"
    best_overlap = 0.0
    for sp in speaker_segments:
        ov = _overlap(seg.start, seg.end, sp.start, sp.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = sp.speaker

    if best_overlap > 0.0:
        return best_speaker

    if not speaker_segments:
        return "UNKNOWN"
    mid = (seg.start + seg.end) / 2
    nearest = min(
        speaker_segments,
        key=lambda sp: min(abs(sp.start - mid), abs(sp.end - mid)),
    )
    return nearest.speaker


def merge_transcript_with_speakers(
    transcript: list[TranscriptSegment],
    speakers: list[SpeakerSegment],
) -> list[AttributedSegment]:
    """Combine transcript + diarization into speaker-attributed segments."""
    attributed: list[AttributedSegment] = []
    for seg in transcript:
        speaker = _attribute_speaker(seg, speakers)
        attributed.append(
            AttributedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=speaker,
            )
        )
    return attributed


def collapse_consecutive(
    segments: list[AttributedSegment],
    max_gap: float = 1.5,
) -> list[AttributedSegment]:
    """Merge neighbouring same-speaker segments into paragraphs.

    Whisper often splits a single spoken sentence into many short
    segments. Once we know the speaker, we can glue consecutive
    same-speaker segments back into a readable paragraph.

    Args:
        segments: Attributed transcript segments.
        max_gap: Max silent gap (seconds) to still be considered the
            same utterance. Beyond this, a new paragraph is started
            even for the same speaker.
    """
    if not segments:
        return []

    merged: list[AttributedSegment] = []
    current = replace(segments[0])

    for seg in segments[1:]:
        gap = seg.start - current.end
        same_speaker = seg.speaker == current.speaker
        if same_speaker and gap <= max_gap:
            current.end = seg.end
            current.text = f"{current.text}{seg.text}".strip()
        else:
            merged.append(current)
            current = replace(seg)

    merged.append(current)
    # Post-merge sanity pass: a merged paragraph made entirely of a
    # repeated hallucinated phrase (e.g. 29× "我可以做点什么吗?")
    # only becomes visible after collapse, so we filter here too.
    return [seg for seg in merged if not is_hallucination(seg.text)]
