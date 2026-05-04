"""Tests for transcript hallucination filtering.

These are pure-function tests — no Whisper invocation, no audio,
no network. They exercise the same heuristics the pipeline relies on
to strip obvious ASR artifacts.
"""

from mac_meeting_transcriber.transcribe import (
    TranscriptSegment,
    _is_repetitive,
    _is_trivial,
    filter_hallucinations,
    is_hallucination,
)


def test_is_hallucination_known_markers():
    assert is_hallucination("请不吝点赞订阅转发打赏支持")
    assert is_hallucination("字幕由 XYZ 制作")
    assert is_hallucination("Subscribe to my channel for more content")
    assert is_hallucination("Thanks for watching!")


def test_is_hallucination_empty_or_whitespace():
    assert is_hallucination("")
    assert is_hallucination("   ")
    assert is_hallucination("\n\t")


def test_is_hallucination_trivial_punctuation():
    assert is_hallucination("。")
    assert is_hallucination(",,,")
    assert is_hallucination("?!")
    assert is_hallucination("「」")


def test_is_hallucination_repetitive():
    # 1-char × many
    assert is_hallucination("嗯嗯嗯嗯嗯嗯嗯嗯")
    # 2-char × many
    assert is_hallucination("呼吸呼吸呼吸呼吸")
    # long phrase × many — the 29× case that motivated the post-merge filter
    assert is_hallucination("我可以做点什么吗？" * 10)


def test_is_hallucination_embedded_degeneration():
    # BELLE-style failure mode observed on the Voice Memos silent pre-roll:
    # short real prefix, then one character looped many times. The
    # head-anchored classifier doesn't catch this because "放弃中文" is
    # itself non-repetitive, but the trailing run of 100+ identical
    # glyphs is unambiguous decoder noise.
    assert is_hallucination("放弃中文异" + "数" * 120)
    assert is_hallucination("还有一个" + "弹" * 80)
    assert is_hallucination("放" + "热" * 60 + "烵" * 40)


def test_is_hallucination_short_repetition_is_ok():
    # Real speech often starts with a short filler like "嗯嗯嗯嗯好". We
    # must NOT drop those — only the long, dominating loops.
    assert not is_hallucination("嗯嗯嗯嗯，那我们开始吧，这是第一点。")
    assert not is_hallucination("对对对，我觉得这个方向没问题。")


def test_is_hallucination_normal_text():
    assert not is_hallucination("Hello, this is a normal utterance.")
    assert not is_hallucination("今天我们讨论了项目的进展。")
    assert not is_hallucination("I think we should try FastViT first.")


def test_is_repetitive_threshold():
    # Below 80% coverage → not flagged
    assert not _is_repetitive("abab trailing real content here", coverage_threshold=0.8)
    # Exactly the repeated fragment → flagged
    assert _is_repetitive("ababababab")


def test_is_trivial_boundary():
    assert _is_trivial("")
    assert _is_trivial("  ... ")
    assert _is_trivial("a")  # single content char — still trivial (<=1)
    assert not _is_trivial("ab")
    assert not _is_trivial("嗯 okay")


def test_filter_hallucinations_drops_junk_keeps_content():
    segs = [
        TranscriptSegment(0.0, 1.0, "Real content one."),
        TranscriptSegment(1.0, 2.0, "嗯嗯嗯嗯嗯嗯嗯"),
        TranscriptSegment(2.0, 3.0, "Subscribe to my channel"),
        TranscriptSegment(3.0, 4.0, "Real content two."),
        TranscriptSegment(4.0, 5.0, "   "),
    ]
    out = filter_hallucinations(segs)
    assert [s.text for s in out] == ["Real content one.", "Real content two."]


def test_filter_hallucinations_empty_input():
    assert filter_hallucinations([]) == []
