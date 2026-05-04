"""Tests for Markdown rendering of attributed transcripts."""

from pathlib import Path

from mac_meeting_transcriber.formatter import (
    _extract_trailing_user_sections,
    _format_timestamp,
    render_markdown,
    save_markdown,
)
from mac_meeting_transcriber.merge import AttributedSegment


def test_format_timestamp_short():
    assert _format_timestamp(0) == "00:00"
    assert _format_timestamp(65) == "01:05"
    assert _format_timestamp(3599) == "59:59"


def test_format_timestamp_crosses_hour_boundary():
    assert _format_timestamp(3600) == "01:00:00"
    assert _format_timestamp(3725) == "01:02:05"
    assert _format_timestamp(7325) == "02:02:05"


def test_format_timestamp_floor_seconds():
    # Fractional seconds truncate, not round.
    assert _format_timestamp(0.9) == "00:00"
    assert _format_timestamp(59.99) == "00:59"


def test_render_markdown_basic_structure():
    segs = [
        AttributedSegment(0.0, 2.0, "Hello", "SPEAKER_00"),
        AttributedSegment(2.0, 4.0, "Hi there", "SPEAKER_01"),
    ]
    md = render_markdown(segs, audio_name="test")

    assert md.startswith("# test\n")
    assert "## Participants" in md
    assert "- **SPEAKER_00**" in md
    assert "- **SPEAKER_01**" in md
    assert "## Transcript" in md
    assert "**[00:00] SPEAKER_00:** Hello" in md
    assert "**[00:02] SPEAKER_01:** Hi there" in md


def test_render_markdown_applies_speaker_aliases():
    segs = [AttributedSegment(0.0, 2.0, "Hi", "SPEAKER_00")]
    md = render_markdown(
        segs, audio_name="t", speaker_aliases={"SPEAKER_00": "Jason"}
    )
    assert "- **Jason**" in md
    assert "[00:00] Jason:" in md
    assert "SPEAKER_00" not in md


def test_render_markdown_participants_ordered_by_first_appearance():
    segs = [
        AttributedSegment(0.0, 1.0, "a", "SPEAKER_02"),
        AttributedSegment(1.0, 2.0, "b", "SPEAKER_00"),
        AttributedSegment(2.0, 3.0, "c", "SPEAKER_02"),
        AttributedSegment(3.0, 4.0, "d", "SPEAKER_01"),
    ]
    md = render_markdown(segs)
    # Only the `## Participants` block; the transcript body also contains
    # speaker labels, so we have to scope the check to the right region.
    participants = md.split("## Participants", 1)[1].split("## Transcript", 1)[0]
    pos2 = participants.index("**SPEAKER_02**")
    pos0 = participants.index("**SPEAKER_00**")
    pos1 = participants.index("**SPEAKER_01**")
    assert pos2 < pos0 < pos1


def test_render_markdown_excludes_unknown_from_participants():
    segs = [
        AttributedSegment(0.0, 1.0, "a", "SPEAKER_00"),
        AttributedSegment(1.0, 2.0, "b", "UNKNOWN"),
    ]
    md = render_markdown(segs)
    participants = md.split("## Participants", 1)[1].split("## Transcript", 1)[0]
    assert "UNKNOWN" not in participants
    assert "**SPEAKER_00**" in participants
    # But UNKNOWN segments still render in the transcript body so the
    # reader doesn't silently lose content.
    transcript = md.split("## Transcript", 1)[1]
    assert "[00:01] UNKNOWN:" in transcript


def test_render_markdown_empty_segments_still_valid():
    md = render_markdown([], audio_name="empty")
    assert "# empty" in md
    # No Participants section when there are no segments.
    assert "## Participants" not in md
    assert "## Transcript" in md
    assert md.endswith("\n")


def test_render_markdown_trailing_newline_once():
    segs = [AttributedSegment(0.0, 1.0, "hi", "SPEAKER_00")]
    md = render_markdown(segs)
    assert md.endswith("\n")
    assert not md.endswith("\n\n")


# ---------------------------------------------------------------------------
# save_markdown — summary preservation across re-runs
# ---------------------------------------------------------------------------


def _sample_rendered_markdown(audio_name: str = "test") -> str:
    segs = [AttributedSegment(0.0, 2.0, "Hello", "SPEAKER_00")]
    return render_markdown(segs, audio_name=audio_name)


def test_extract_trailing_user_sections_finds_summary():
    existing = (
        "# test\n\n"
        "_Transcribed 2026-01-01 00:00 · 1 segments · 1 speakers_\n\n"
        "## Participants\n- **Jason**\n\n"
        "## Transcript\n\n"
        "**[00:00] Jason:** Hello\n\n"
        "## 要点总结\n- 项目 X 进展顺利\n\n"
        "## Action Items\n| Owner | Action |\n"
    )
    tail = _extract_trailing_user_sections(existing)
    assert tail.startswith("## 要点总结")
    assert "## Action Items" in tail
    # Nothing from mmt-owned sections should leak into the tail.
    assert "Transcript" not in tail
    assert "Jason" not in tail.split("## Action Items", 1)[0]


def test_extract_trailing_user_sections_no_tail():
    existing = (
        "# test\n\n"
        "## Participants\n- **Jason**\n\n"
        "## Transcript\n\n"
        "**[00:00] Jason:** Hello\n"
    )
    assert _extract_trailing_user_sections(existing) == ""


def test_extract_trailing_user_sections_empty_file():
    assert _extract_trailing_user_sections("") == ""
    assert _extract_trailing_user_sections("   \n\n") == ""


def test_save_markdown_preserves_existing_summary(tmp_path: Path):
    # Simulate the meeting-summarizer workflow: mmt creates the file,
    # skill appends a summary, user re-runs mmt with a tweak.
    out = tmp_path / "session.md"
    save_markdown(_sample_rendered_markdown("session"), out)

    # Skill appends its content.
    original = out.read_text(encoding="utf-8")
    summary = (
        "\n\n## 要点总结\n"
        "- **Jason** pushed for fair comparison on fixed compute.\n\n"
        "## Action Items\n"
        "| Owner | Action |\n"
        "| Jason | Re-run with merged car+truck class |\n"
    )
    out.write_text(original + summary, encoding="utf-8")

    # Re-run mmt: same rendered markdown (or a tweaked variant, doesn't
    # matter for this test).
    save_markdown(_sample_rendered_markdown("session"), out)

    after = out.read_text(encoding="utf-8")
    # The user-authored tail is intact, byte-for-byte.
    assert "## 要点总结" in after
    assert "pushed for fair comparison" in after
    assert "## Action Items" in after
    assert "Re-run with merged car+truck class" in after
    # And the mmt-owned sections are still in front of it.
    assert after.index("## Transcript") < after.index("## 要点总结")


def test_save_markdown_no_existing_file_is_plain_write(tmp_path: Path):
    out = tmp_path / "new.md"
    save_markdown(_sample_rendered_markdown("new"), out)
    content = out.read_text(encoding="utf-8")
    assert content.startswith("# new\n")
    assert "## 要点总结" not in content
    # Exactly one trailing newline.
    assert content.endswith("\n")
    assert not content.endswith("\n\n")


def test_save_markdown_empty_existing_file_overwrites(tmp_path: Path):
    out = tmp_path / "empty.md"
    out.write_text("", encoding="utf-8")
    save_markdown(_sample_rendered_markdown("empty"), out)
    assert out.read_text(encoding="utf-8").startswith("# empty\n")


def test_save_markdown_preserves_multiple_user_sections(tmp_path: Path):
    out = tmp_path / "multi.md"
    save_markdown(_sample_rendered_markdown("multi"), out)
    original = out.read_text(encoding="utf-8")
    out.write_text(
        original + "\n## Summary\n- x\n\n## Intents\n- y\n\n## Notes\n- z\n",
        encoding="utf-8",
    )
    save_markdown(_sample_rendered_markdown("multi"), out)
    after = out.read_text(encoding="utf-8")
    assert "## Summary" in after
    assert "## Intents" in after
    assert "## Notes" in after
    # Relative order among user sections is preserved.
    assert after.index("## Summary") < after.index("## Intents") < after.index("## Notes")


def test_save_markdown_trailing_newline_after_merge(tmp_path: Path):
    # Tail that ends WITHOUT a newline should still leave file ending in
    # exactly one newline after save.
    out = tmp_path / "endings.md"
    save_markdown(_sample_rendered_markdown("endings"), out)
    original = out.read_text(encoding="utf-8")
    # No trailing newline on the appended tail, one of the subtler bugs
    # to regress on.
    out.write_text(original + "\n## Notes\n- tail without trailing newline", encoding="utf-8")
    save_markdown(_sample_rendered_markdown("endings"), out)
    after = out.read_text(encoding="utf-8")
    assert after.endswith("\n")
    assert not after.endswith("\n\n\n")  # no excessive blank lines either
