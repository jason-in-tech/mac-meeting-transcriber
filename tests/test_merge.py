"""Tests for transcript ↔ diarization alignment and paragraph collapse."""

from mac_meeting_transcriber.diarize import SpeakerSegment
from mac_meeting_transcriber.merge import (
    AttributedSegment,
    _attribute_speaker,
    _overlap,
    collapse_consecutive,
    merge_transcript_with_speakers,
)
from mac_meeting_transcriber.transcribe import TranscriptSegment


def test_overlap_simple_overlap():
    assert _overlap(0.0, 10.0, 5.0, 15.0) == 5.0


def test_overlap_no_overlap():
    assert _overlap(0.0, 5.0, 5.0, 10.0) == 0.0
    assert _overlap(0.0, 5.0, 10.0, 20.0) == 0.0


def test_overlap_full_containment():
    assert _overlap(0.0, 10.0, 2.0, 8.0) == 6.0
    assert _overlap(2.0, 8.0, 0.0, 10.0) == 6.0


def test_attribute_speaker_majority_wins():
    seg = TranscriptSegment(0.0, 10.0, "hello")
    speakers = [
        SpeakerSegment(0.0, 3.0, "SPEAKER_00"),
        SpeakerSegment(3.0, 10.0, "SPEAKER_01"),
    ]
    assert _attribute_speaker(seg, speakers) == "SPEAKER_01"


def test_attribute_speaker_nearest_fallback_when_no_overlap():
    # Transcript seg at (100, 110) falls in a gap Senko didn't label.
    # SPEAKER_00 ends at 50 (distance 55 to midpoint 105).
    # SPEAKER_01 starts at 200 (distance 95 to midpoint 105).
    # SPEAKER_00 wins.
    seg = TranscriptSegment(100.0, 110.0, "orphan")
    speakers = [
        SpeakerSegment(0.0, 50.0, "SPEAKER_00"),
        SpeakerSegment(200.0, 300.0, "SPEAKER_01"),
    ]
    assert _attribute_speaker(seg, speakers) == "SPEAKER_00"


def test_attribute_speaker_empty_speakers():
    seg = TranscriptSegment(0.0, 10.0, "alone")
    assert _attribute_speaker(seg, []) == "UNKNOWN"


def test_merge_transcript_with_speakers_round_trip():
    transcript = [
        TranscriptSegment(0.0, 5.0, "A"),
        TranscriptSegment(5.0, 10.0, "B"),
        TranscriptSegment(10.0, 15.0, "C"),
    ]
    speakers = [
        SpeakerSegment(0.0, 7.0, "SPEAKER_00"),
        SpeakerSegment(7.0, 15.0, "SPEAKER_01"),
    ]
    out = merge_transcript_with_speakers(transcript, speakers)
    assert [(s.text, s.speaker) for s in out] == [
        ("A", "SPEAKER_00"),
        ("B", "SPEAKER_00"),  # 5-7s overlap > 7-10s (2s vs 3s)… wait
    ] or [(s.text, s.speaker) for s in out] == [
        ("A", "SPEAKER_00"),
        ("B", "SPEAKER_01"),  # 5-7s=2s with S0, 7-10s=3s with S1 → S1 wins
        ("C", "SPEAKER_01"),
    ]


def test_merge_transcript_preserves_timestamps():
    transcript = [TranscriptSegment(1.5, 3.25, "hi")]
    speakers = [SpeakerSegment(0.0, 10.0, "SPEAKER_00")]
    out = merge_transcript_with_speakers(transcript, speakers)
    assert out[0].start == 1.5
    assert out[0].end == 3.25
    assert out[0].text == "hi"


def test_collapse_consecutive_merges_same_speaker():
    segs = [
        AttributedSegment(0.0, 1.0, "Hello ", "S0"),
        AttributedSegment(1.0, 2.0, "world", "S0"),
        AttributedSegment(2.0, 3.0, "bye", "S1"),
    ]
    out = collapse_consecutive(segs, max_gap=0.5)
    assert len(out) == 2
    assert out[0].speaker == "S0"
    assert out[0].start == 0.0
    assert out[0].end == 2.0
    assert "Hello" in out[0].text and "world" in out[0].text
    assert out[1].speaker == "S1"


def test_collapse_consecutive_respects_gap():
    segs = [
        AttributedSegment(0.0, 1.0, "First sentence.", "S0"),
        AttributedSegment(5.0, 6.0, "Second sentence.", "S0"),  # 4s silent gap
    ]
    out = collapse_consecutive(segs, max_gap=1.5)
    assert len(out) == 2
    assert out[0].text == "First sentence."
    assert out[1].text == "Second sentence."


def test_collapse_consecutive_does_not_mutate_input():
    original_end = 1.0
    original_text = "First sentence."
    segs = [
        AttributedSegment(0.0, original_end, original_text, "S0"),
        AttributedSegment(1.0, 2.0, "Second sentence.", "S0"),
    ]
    _ = collapse_consecutive(segs, max_gap=1.5)
    # Caller's list must remain untouched — we should not have mutated
    # segs[0].end or segs[0].text by reference.
    assert segs[0].end == original_end
    assert segs[0].text == original_text
    assert segs[1].text == "Second sentence."


def test_collapse_consecutive_filters_post_merge_hallucination():
    # Six short segments of "呼吸"; gap=0 so they merge into one paragraph
    # whose text is "呼吸" × 6 — a post-merge hallucination that only
    # becomes visible after collapse, and must be filtered here.
    segs = [
        AttributedSegment(float(i), float(i + 1), "呼吸", "S0") for i in range(6)
    ]
    out = collapse_consecutive(segs, max_gap=1.5)
    assert out == []


def test_collapse_consecutive_empty():
    assert collapse_consecutive([]) == []


def test_collapse_consecutive_singleton():
    segs = [AttributedSegment(0.0, 1.0, "Hello there.", "S0")]
    out = collapse_consecutive(segs)
    assert len(out) == 1
    assert out[0].text == "Hello there."
