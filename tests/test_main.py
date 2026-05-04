"""Tests for the CLI entry-point logic that doesn't require Whisper/Senko.

We import the pure helpers from `mac_meeting_transcriber.__main__` and
exercise them directly. The full `main()` pipeline is integration-tested
manually (it needs real audio + the MLX stack).
"""

from mac_meeting_transcriber.__main__ import (
    DEFAULT_ZH_MEETING_PROMPT,
    resolve_initial_prompt,
)


def _resolve(**overrides):
    base = dict(
        no_initial_prompt=False,
        initial_prompt=None,
        initial_prompt_file=None,
        env_prompt=None,
        language=None,
    )
    base.update(overrides)
    return resolve_initial_prompt(**base)


def test_no_initial_prompt_flag_always_wins():
    # Even with every other hint set, --no-initial-prompt forces None.
    result = _resolve(
        no_initial_prompt=True,
        initial_prompt="explicit",
        env_prompt="env",
        language="zh",
    )
    assert result is None


def test_explicit_prompt_beats_env_and_default():
    result = _resolve(
        initial_prompt="please transcribe verbatim",
        env_prompt="ignored",
        language="zh",
    )
    assert result == "please transcribe verbatim"


def test_env_prompt_beats_language_default():
    result = _resolve(env_prompt="team-wide default", language="zh")
    assert result == "team-wide default"


def test_zh_language_triggers_builtin_default():
    result = _resolve(language="zh")
    assert result == DEFAULT_ZH_MEETING_PROMPT


def test_non_zh_language_has_no_default():
    assert _resolve(language="en") is None
    assert _resolve(language=None) is None


def test_empty_env_prompt_falls_through_to_language_default():
    # Env vars are often exported as empty strings; we treat "" as unset.
    assert _resolve(env_prompt="", language="zh") == DEFAULT_ZH_MEETING_PROMPT
    assert _resolve(env_prompt="", language="en") is None


def test_default_zh_prompt_is_simplified_chinese():
    # Sanity check: the built-in prompt shouldn't accidentally contain
    # Traditional glyphs or Cantonese particles, or it would defeat its
    # own purpose.
    cantonese_particles = ("我哋", "嘅", "咁", "喺", "嚟", "嘢")
    for particle in cantonese_particles:
        assert particle not in DEFAULT_ZH_MEETING_PROMPT
    # A few common Traditional-vs-Simplified anchors.
    assert "訓練" not in DEFAULT_ZH_MEETING_PROMPT  # simplified: 训练
    assert "轉錄" not in DEFAULT_ZH_MEETING_PROMPT  # simplified: 转录
    # Must mention the task hint so Whisper doesn't flip into translate mode.
    assert "不要翻译" in DEFAULT_ZH_MEETING_PROMPT
