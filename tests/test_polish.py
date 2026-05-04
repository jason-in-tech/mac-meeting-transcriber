"""Tests for LLM-free parts of the polish pipeline."""

import json
from pathlib import Path

import pytest

from mac_meeting_transcriber.merge import AttributedSegment
from mac_meeting_transcriber.polish import (
    MAX_CHARS_PER_BATCH,
    MAX_SEGMENTS_PER_BATCH,
    _batch_segments,
    _build_messages,
    _parse_polished,
    normalize_chinese_script,
    resolve_glossary,
)


def _seg(i: int, chars: int = 50, speaker: str = "S0") -> AttributedSegment:
    return AttributedSegment(
        start=float(i),
        end=float(i + 1),
        text="x" * chars,
        speaker=speaker,
    )


def test_batch_segments_hits_segment_cap():
    segs = [_seg(i, 10) for i in range(50)]
    batches = _batch_segments(segs, max_segments=20, max_chars=10_000)
    assert len(batches) == 3
    assert len(batches[0]) == 20
    assert len(batches[1]) == 20
    assert len(batches[2]) == 10


def test_batch_segments_hits_char_cap():
    # 10 × 2000-char segments with a 5000-char cap: each batch can
    # hold at most 2 before flushing (2×2000 < 5000, 3×2000 > 5000).
    segs = [_seg(i, 2000) for i in range(10)]
    batches = _batch_segments(segs, max_segments=20, max_chars=5000)
    for batch in batches:
        total = sum(len(seg.text) for _, seg in batch)
        assert total <= 5000 or len(batch) == 1  # a single giant seg can exceed


def test_batch_segments_preserves_order_and_indices():
    segs = [_seg(i, 5) for i in range(10)]
    batches = _batch_segments(segs, max_segments=3, max_chars=10_000)
    flat = [(idx, seg) for batch in batches for idx, seg in batch]
    assert [idx for idx, _ in flat] == list(range(10))
    assert all(flat[i][1] is segs[i] for i in range(10))


def test_batch_segments_empty():
    assert _batch_segments([]) == []


def test_batch_segments_singleton_oversize():
    # A single segment larger than max_chars still gets its own batch —
    # we don't split mid-segment.
    segs = [_seg(0, 20_000)]
    batches = _batch_segments(segs, max_segments=20, max_chars=10_000)
    assert len(batches) == 1
    assert len(batches[0]) == 1


def test_batch_module_defaults_are_reasonable():
    # Guard against accidental regression of the tuning knobs.
    assert MAX_SEGMENTS_PER_BATCH >= 10
    assert MAX_CHARS_PER_BATCH >= 4000


def test_parse_polished_accepts_valid_json():
    raw = json.dumps({
        "polished": [
            {"id": 0, "text": "A polished."},
            {"id": 1, "text": "B polished."},
        ]
    })
    out = _parse_polished(raw, [0, 1])
    assert out == {0: "A polished.", 1: "B polished."}


def test_parse_polished_strips_code_fence():
    raw = '```json\n{"polished":[{"id":0,"text":"A"}]}\n```'
    assert _parse_polished(raw, [0]) == {0: "A"}


def test_parse_polished_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        _parse_polished("", [0])
    with pytest.raises(ValueError, match="empty"):
        _parse_polished("   ", [0])


def test_parse_polished_rejects_missing_ids():
    raw = json.dumps({"polished": [{"id": 0, "text": "A"}]})
    with pytest.raises(ValueError, match="missing"):
        _parse_polished(raw, [0, 1, 2])


def test_parse_polished_rejects_non_list_field():
    raw = json.dumps({"polished": {"id": 0, "text": "A"}})
    with pytest.raises(ValueError, match="not a list"):
        _parse_polished(raw, [0])


def test_parse_polished_skips_malformed_items_but_still_errors_if_expected_missing():
    # Bad items (non-dict, missing keys) are silently dropped, so if a
    # real expected id ends up missing we still raise.
    raw = json.dumps({
        "polished": [
            "not a dict",
            {"id": "not-an-int", "text": "A"},
            {"id": 0},  # missing text
            {"id": 1, "text": "kept"},
        ]
    })
    with pytest.raises(ValueError, match="missing"):
        _parse_polished(raw, [0, 1])

    # But when every expected id IS present, we accept the batch and
    # silently ignore the malformed siblings.
    raw_ok = json.dumps({
        "polished": [
            "not a dict",
            {"id": 1, "text": "kept"},
        ]
    })
    assert _parse_polished(raw_ok, [1]) == {1: "kept"}


def test_resolve_glossary_explicit_path(tmp_path: Path):
    g = tmp_path / "gloss.md"
    g.write_text("# Glossary\n- FastViT\n", encoding="utf-8")
    text, path = resolve_glossary(str(g))
    assert "FastViT" in text
    assert path == str(g)


def test_resolve_glossary_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    g = tmp_path / "env_gloss.md"
    g.write_text("env-glossary", encoding="utf-8")
    monkeypatch.setenv("MMT_GLOSSARY", str(g))
    text, path = resolve_glossary(None)
    assert text == "env-glossary"
    assert path == str(g)


def test_resolve_glossary_missing_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Redirect HOME so the default ~/.config/... path can't accidentally
    # hit a real user glossary, and clear the env override.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MMT_GLOSSARY", raising=False)
    text, path = resolve_glossary(None)
    assert text == ""
    assert path is None


def test_resolve_glossary_explicit_beats_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    explicit = tmp_path / "explicit.md"
    explicit.write_text("explicit", encoding="utf-8")
    from_env = tmp_path / "env.md"
    from_env.write_text("from-env", encoding="utf-8")
    monkeypatch.setenv("MMT_GLOSSARY", str(from_env))

    text, path = resolve_glossary(str(explicit))
    assert text == "explicit"
    assert path == str(explicit)


def test_normalize_chinese_script_auto_is_noop():
    text = "我哋嘅會議很長"
    assert normalize_chinese_script(text, chinese_style="auto") == text


def test_normalize_chinese_script_traditional_is_noop():
    text = "我哋嘅會議很長"
    assert normalize_chinese_script(text, chinese_style="traditional") == text


def test_normalize_chinese_script_simplified_swaps_cantonese_particles():
    # These Cantonese particles should be rewritten to Mandarin even without OpenCC.
    text = "我哋嘅老闆話咁樣做係唔啱嘅"
    out = normalize_chinese_script(text, chinese_style="simplified")
    assert "我哋" not in out
    assert "我们" in out
    # 嘅 → 的 after Cantonese swap, or further simplified downstream.
    assert "嘅" not in out


def test_normalize_chinese_script_simplified_passes_mandarin_unchanged():
    # Pure Mandarin Simplified text should survive the normalizer untouched.
    text = "今天开会讨论了 NVIDIA 数据的 labeling spec。"
    assert normalize_chinese_script(text, chinese_style="simplified") == text


def test_build_messages_injects_simplified_rules():
    segs = [(0, AttributedSegment(start=0.0, end=1.0, text="test", speaker="S0"))]
    messages = _build_messages(segs, glossary="", chinese_style="simplified")
    user_content = messages[1]["content"]
    assert "Simplified" in user_content
    assert "Cantonese" in user_content
    assert "我哋" in user_content


def test_build_messages_omits_chinese_rules_when_auto():
    segs = [(0, AttributedSegment(start=0.0, end=1.0, text="test", speaker="S0"))]
    messages = _build_messages(segs, glossary="", chinese_style="auto")
    user_content = messages[1]["content"]
    assert "Simplified" not in user_content
    assert "Cantonese" not in user_content
