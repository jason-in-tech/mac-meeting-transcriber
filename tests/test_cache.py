"""Tests for the on-disk Whisper transcript cache."""

import json
from pathlib import Path

from mac_meeting_transcriber.cache import (
    compute_cache_key,
    load_transcript_cache,
    save_transcript_cache,
)
from mac_meeting_transcriber.transcribe import TranscriptSegment


def _write_fake_audio(path: Path, size: int = 4 * 1024 * 1024) -> Path:
    # Deterministic pseudo-random bytes so head+tail+size hash stays stable.
    payload = bytes((i * 37 + 11) % 256 for i in range(size))
    path.write_bytes(payload)
    return path


def test_cache_key_stable_for_same_inputs(tmp_path: Path):
    audio = _write_fake_audio(tmp_path / "a.wav")
    k1 = compute_cache_key(audio, "whisper-large-v3", "zh")
    k2 = compute_cache_key(audio, "whisper-large-v3", "zh")
    assert k1 == k2


def test_cache_key_changes_with_model(tmp_path: Path):
    audio = _write_fake_audio(tmp_path / "a.wav")
    k1 = compute_cache_key(audio, "whisper-large-v3", "zh")
    k2 = compute_cache_key(audio, "whisper-large-v2", "zh")
    assert k1 != k2


def test_cache_key_changes_with_language(tmp_path: Path):
    audio = _write_fake_audio(tmp_path / "a.wav")
    k_zh = compute_cache_key(audio, "whisper-large-v3", "zh")
    k_en = compute_cache_key(audio, "whisper-large-v3", "en")
    k_none = compute_cache_key(audio, "whisper-large-v3", None)
    assert len({k_zh, k_en, k_none}) == 3


def test_cache_key_changes_with_initial_prompt(tmp_path: Path):
    audio = _write_fake_audio(tmp_path / "a.wav")
    k_none = compute_cache_key(audio, "whisper-large-v3", "zh", None)
    k_a = compute_cache_key(audio, "whisper-large-v3", "zh", "prompt A")
    k_b = compute_cache_key(audio, "whisper-large-v3", "zh", "prompt B")
    k_a_dup = compute_cache_key(audio, "whisper-large-v3", "zh", "prompt A")
    assert len({k_none, k_a, k_b}) == 3
    assert k_a == k_a_dup


def test_cache_key_backcompat_prompt_default_matches_none(tmp_path: Path):
    # Callers that don't pass initial_prompt should hit the same cache
    # entry as callers that pass an explicit None — that preserves old
    # cache files.
    audio = _write_fake_audio(tmp_path / "a.wav")
    k_missing = compute_cache_key(audio, "whisper-large-v3", "zh")
    k_none = compute_cache_key(audio, "whisper-large-v3", "zh", None)
    assert k_missing == k_none


def test_cache_key_changes_with_audio_content(tmp_path: Path):
    a = _write_fake_audio(tmp_path / "a.wav")
    b = tmp_path / "b.wav"
    b.write_bytes(b"\x00" * a.stat().st_size)  # same size, different content
    k_a = compute_cache_key(a, "m", None)
    k_b = compute_cache_key(b, "m", None)
    assert k_a != k_b


def test_cache_key_uses_small_file_path(tmp_path: Path):
    # Files below the 2 MiB threshold still compute a valid key (they
    # just hash head bytes only, no tail seek).
    small = tmp_path / "tiny.wav"
    small.write_bytes(b"tiny audio content")
    key = compute_cache_key(small, "whisper-large-v3", None)
    assert isinstance(key, str) and len(key) == 64  # SHA-256 hex


def test_load_cache_miss_returns_none(tmp_path: Path):
    assert load_transcript_cache("nonexistent-key", cache_dir=tmp_path) is None


def test_save_then_load_round_trip(tmp_path: Path):
    segs = [
        TranscriptSegment(0.0, 1.5, "Hello world"),
        TranscriptSegment(1.5, 3.0, "今天我们聊一下。"),
    ]
    save_transcript_cache("abc123", segs, cache_dir=tmp_path)
    loaded = load_transcript_cache("abc123", cache_dir=tmp_path)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].text == "Hello world"
    assert loaded[1].text == "今天我们聊一下。"
    assert loaded[0].start == 0.0
    assert loaded[1].end == 3.0


def test_cache_file_is_utf8_json(tmp_path: Path):
    segs = [TranscriptSegment(0.0, 1.0, "中文")]
    path = save_transcript_cache("cjk", segs, cache_dir=tmp_path)
    assert path.exists()
    raw = path.read_text(encoding="utf-8")
    # Non-ASCII should be preserved, not escaped.
    assert "中文" in raw
    payload = json.loads(raw)
    assert payload["segments"][0]["text"] == "中文"


def test_load_cache_corrupt_file_returns_none(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{", encoding="utf-8")
    # Key matches the file basename so load looks at `bad.json`.
    assert load_transcript_cache("bad", cache_dir=tmp_path) is None
