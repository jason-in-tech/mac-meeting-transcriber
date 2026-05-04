"""Render attributed transcript segments as a Markdown document."""

from datetime import datetime
from pathlib import Path

from .merge import AttributedSegment


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS depending on length."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def render_markdown(
    segments: list[AttributedSegment],
    audio_name: str | None = None,
    speaker_aliases: dict[str, str] | None = None,
) -> str:
    """Render attributed segments as a human-readable Markdown transcript.

    Args:
        segments: Attributed (and optionally collapsed) transcript segments.
        audio_name: Friendly name for the header. Defaults to "Untitled".
        speaker_aliases: Optional mapping from "SPEAKER_00" → display name
            (e.g. {"SPEAKER_00": "Jason", "SPEAKER_01": "Leo"}).
    """
    aliases = speaker_aliases or {}
    title = audio_name or "Untitled"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    speakers_in_order: list[str] = []
    for seg in segments:
        if seg.speaker not in speakers_in_order:
            speakers_in_order.append(seg.speaker)
    participant_lines = [
        f"- **{aliases.get(sp, sp)}**"
        for sp in speakers_in_order
        if sp != "UNKNOWN"
    ]

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Transcribed {timestamp} · {len(segments)} segments · {len(speakers_in_order)} speakers_")
    lines.append("")

    if participant_lines:
        lines.append("## Participants")
        lines.extend(participant_lines)
        lines.append("")

    lines.append("## Transcript")
    lines.append("")

    for seg in segments:
        speaker = aliases.get(seg.speaker, seg.speaker)
        ts = _format_timestamp(seg.start)
        lines.append(f"**[{ts}] {speaker}:** {seg.text}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# Top-level headings that ``render_markdown`` produces. Anything else at
# the ``## `` level in an existing file is treated as user-authored content
# (a summary section, hand-written notes, action items from the
# meeting-summarizer skill, etc.) and preserved across re-runs.
_MMT_OWNED_SECTIONS = frozenset({"## Participants", "## Transcript"})


def _extract_trailing_user_sections(existing: str) -> str:
    """Return everything at/after the first non-mmt-owned ``## `` heading.

    ``mmt`` renders a fixed set of sections (``## Participants``,
    ``## Transcript``). Users frequently append their own sections
    afterwards — most notably the summary produced by the
    meeting-summarizer skill (``## 要点总结``, ``## Action Items``, …).
    Re-running ``mmt`` on the same audio MUST NOT silently delete that
    downstream content.

    The preserved chunk starts exactly at the first unrecognised ``## ``
    line we find (we assume mmt's own sections always come first in the
    order render_markdown emits them). Anything before that — title,
    transcription metadata, Participants, Transcript body — gets
    overwritten by the freshly rendered content.

    Returns an empty string when the file has no user-authored tail.
    """
    if not existing.strip():
        return ""
    for idx, line in enumerate(existing.splitlines(keepends=True)):
        stripped = line.rstrip("\n")
        if stripped.startswith("## ") and stripped not in _MMT_OWNED_SECTIONS:
            offset = sum(
                len(raw) for raw in existing.splitlines(keepends=True)[:idx]
            )
            return existing[offset:]
    return ""


def save_markdown(markdown: str, output_path: str | Path) -> Path:
    """Write ``markdown`` to ``output_path``, preserving any user-authored
    sections (summary, notes, action items) appended to the previous file.

    A plain overwrite would be fine when a user first runs mmt, but re-runs
    (e.g. after tweaking the model, language hint, or speaker names) would
    silently destroy any summary the meeting-summarizer skill already
    appended. That's unacceptable — the skill's whole value is the
    summary, so we merge instead of stomp.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preserved = ""
    if output_path.exists():
        try:
            preserved = _extract_trailing_user_sections(
                output_path.read_text(encoding="utf-8")
            )
        except OSError:
            # Unreadable existing file: fall back to overwrite rather than
            # failing the whole run.
            preserved = ""

    combined = markdown.rstrip("\n")
    if preserved:
        combined = combined + "\n\n" + preserved.lstrip("\n")
    if not combined.endswith("\n"):
        combined += "\n"

    output_path.write_text(combined, encoding="utf-8")
    return output_path
