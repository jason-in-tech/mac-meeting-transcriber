"""Tests for LLM-free parts of speaker identification.

Everything here runs without contacting the LLM — we're testing the
parsing, scoring, and validation scaffolding around the LLM call.
"""

import pytest

from mac_meeting_transcriber.identify import (
    Candidate,
    _distinct_speakers,
    _expand_hint,
    _score_utterance,
    _validate_mapping,
    parse_candidates,
    sample_speaker_utterances,
    strip_code_fence,
)
from mac_meeting_transcriber.merge import AttributedSegment


def test_parse_candidates_bare_names():
    out = parse_candidates("Jason,Leo,Kashish")
    assert out == [
        Candidate("Jason", ""),
        Candidate("Leo", ""),
        Candidate("Kashish", ""),
    ]


def test_parse_candidates_with_hints():
    out = parse_candidates("Jason:IC, Leo : manager ,Kashish")
    assert out[0] == Candidate("Jason", "IC")
    assert out[1] == Candidate("Leo", "manager")
    assert out[2] == Candidate("Kashish", "")


def test_parse_candidates_skips_empties():
    assert parse_candidates("") == []
    assert parse_candidates(" , , ") == []
    assert parse_candidates(" , ,Jason , ") == [Candidate("Jason", "")]


def test_parse_candidates_freeform_hint():
    out = parse_candidates("Leo:Jason's direct report")
    assert out == [Candidate("Leo", "Jason's direct report")]


def test_expand_hint_known_shorthand():
    assert "individual contributor" in _expand_hint("ic").lower()
    assert "individual contributor" in _expand_hint("IC").lower()
    assert "manager" in _expand_hint("mgr").lower()
    assert "manager" in _expand_hint("manager").lower()
    assert "skip-level" in _expand_hint("skip").lower()
    assert "peer" in _expand_hint("peer").lower()
    assert "direct report" in _expand_hint("report").lower()


def test_expand_hint_passthrough_freeform():
    assert _expand_hint("Jason's direct report") == "Jason's direct report"


def test_expand_hint_empty():
    assert _expand_hint("") == ""
    assert _expand_hint("   ") == ""


def test_score_utterance_filler_is_zero():
    assert _score_utterance("yeah") == 0
    assert _score_utterance("right") == 0
    assert _score_utterance("okay") == 0
    assert _score_utterance("嗯嗯嗯") == 0
    assert _score_utterance("对") == 0
    assert _score_utterance("mm-hmm") == 0


def test_score_utterance_too_short_is_zero():
    assert _score_utterance("ok") == 0
    assert _score_utterance("sure thing.") == 0  # len < 20


def test_score_utterance_substantive_has_score():
    score = _score_utterance("We should ship the new eval protocol by EOW.")
    assert score > 0


def test_distinct_speakers_preserves_first_appearance_order():
    segs = [
        AttributedSegment(0.0, 1.0, "a", "SPEAKER_02"),
        AttributedSegment(1.0, 2.0, "b", "SPEAKER_00"),
        AttributedSegment(2.0, 3.0, "c", "SPEAKER_02"),
        AttributedSegment(3.0, 4.0, "d", "UNKNOWN"),
        AttributedSegment(4.0, 5.0, "e", "SPEAKER_01"),
    ]
    assert _distinct_speakers(segs) == ["SPEAKER_02", "SPEAKER_00", "SPEAKER_01"]


def test_sample_speaker_utterances_returns_per_speaker_samples():
    segs = []
    for i in range(8):
        segs.append(
            AttributedSegment(
                float(i), float(i + 1),
                f"Substantive utterance number {i} from speaker zero that "
                "is long enough to pass the information-density filter.",
                "SPEAKER_00",
            )
        )
    for i in range(8, 16):
        segs.append(
            AttributedSegment(
                float(i), float(i + 1),
                f"Another long utterance {i} from the second speaker, with "
                "plenty of content and words to score above the threshold.",
                "SPEAKER_01",
            )
        )
    samples = sample_speaker_utterances(segs, max_per_speaker=3)
    assert set(samples.keys()) == {"SPEAKER_00", "SPEAKER_01"}
    assert all(1 <= len(v) <= 3 for v in samples.values())


def test_sample_speaker_utterances_drops_filler():
    segs = [
        AttributedSegment(0.0, 1.0, "yeah", "SPEAKER_00"),
        AttributedSegment(1.0, 2.0, "ok", "SPEAKER_00"),
    ]
    assert sample_speaker_utterances(segs) == {}


def test_validate_mapping_valid():
    out = _validate_mapping(
        {"SPEAKER_00": "Jason", "SPEAKER_01": "Leo"},
        ["SPEAKER_00", "SPEAKER_01"],
        ["Jason", "Leo"],
    )
    assert out == {"SPEAKER_00": "Jason", "SPEAKER_01": "Leo"}


def test_validate_mapping_rejects_duplicate_candidate():
    with pytest.raises(ValueError, match="duplicate"):
        _validate_mapping(
            {"SPEAKER_00": "Jason", "SPEAKER_01": "Jason"},
            ["SPEAKER_00", "SPEAKER_01"],
            ["Jason", "Leo"],
        )


def test_validate_mapping_rejects_unknown_name():
    with pytest.raises(ValueError, match="not in candidates"):
        _validate_mapping(
            {"SPEAKER_00": "Alice"},
            ["SPEAKER_00"],
            ["Jason", "Leo"],
        )


def test_validate_mapping_rejects_missing_speaker():
    with pytest.raises(ValueError, match="no name assigned"):
        _validate_mapping(
            {"SPEAKER_00": "Jason"},
            ["SPEAKER_00", "SPEAKER_01"],
            ["Jason", "Leo"],
        )


def test_validate_mapping_rejects_non_dict():
    with pytest.raises(ValueError, match="not a dict"):
        _validate_mapping("not a dict", ["SPEAKER_00"], ["Jason"])


def test_strip_code_fence_plain_passthrough():
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_strip_code_fence_with_json_fence():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_with_bare_fence():
    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_preserves_internal_whitespace():
    assert strip_code_fence('```json\n{\n  "a": 1\n}\n```') == '{\n  "a": 1\n}'
