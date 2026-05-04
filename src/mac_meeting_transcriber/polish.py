"""LLM-based post-editing of meeting transcripts.

Whisper transcripts have predictable weaknesses that no stronger ASR
model can fix because they require semantic knowledge:

  - Dropped or wrong capitalization on technical terms
    ("fastvit" instead of "FastViT", "nvidia" instead of "NVIDIA").
  - Homophone picks, especially at code-switch boundaries
    ("allhead" instead of "all-hands", "onfreeze" instead of "unfreeze").
  - Long unpunctuated runs that are hard to scan.
  - Occasional character-level drift in Chinese names
    (picking the wrong 同音字 when an English name was meant, or vice versa).

This module asks an LLM to do a light copy-edit pass, optionally
anchored by a user-supplied glossary of project-specific vocabulary.
It preserves every segment's speaker and timestamps — only the text
is rewritten. If a batch fails, its segments pass through unchanged,
so the pipeline never regresses below the raw-Whisper baseline.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from openai import OpenAI

from .identify import (
    FALLBACK_MODELS,
    LLMConfig,
    chat_json_call,
    strip_code_fence,
)
from .merge import AttributedSegment

MAX_SEGMENTS_PER_BATCH = 20
MAX_CHARS_PER_BATCH = 10000
DEFAULT_MAX_WORKERS = 8

DEFAULT_GLOSSARY_PATH = "~/.config/mac-meeting-transcriber/glossary.md"


# Cantonese particles that Whisper sometimes hallucinates when the speaker
# is actually speaking Mandarin. The LLM polish pass handles most of these
# via the prompt, but we apply a deterministic sweep as a safety net so the
# output is clean even when the LLM batch fails or is skipped.
# Order matters: multi-char compounds must fire before single-char ones,
# because Python's str.replace is left-to-right greedy within each pass.
_CANTONESE_TO_MANDARIN: tuple[tuple[str, str], ...] = (
    # Compound forms first — these MUST fire before their single-char parts
    # get swapped, otherwise e.g. 係唔係 → 是不係 before the final 係 maps.
    ("係唔係", "是不是"),
    ("唔係", "不是"),
    ("我哋", "我们"),
    ("你哋", "你们"),
    ("佢哋", "他们"),
    ("點樣", "怎样"),
    ("點解", "为什么"),
    ("呢個", "这个"),
    ("呢啲", "这些"),
    ("嗰個", "那个"),
    ("嗰啲", "那些"),
    ("呢度", "这里"),
    ("嗰度", "那里"),
    # 著 is valid Mandarin (著名, 显著) so we only convert the compound
    # forms where it unambiguously plays the verbal-particle role that in
    # Simplified Mandarin is written 着.
    ("向著", "向着"),
    ("朝著", "朝着"),
    ("沿著", "沿着"),
    ("隨著", "随着"),
    ("接著", "接着"),
    ("跟著", "跟着"),
    ("按著", "按着"),
    ("拿著", "拿着"),
    ("看著", "看着"),
    ("想著", "想着"),
    # Single-character Cantonese particles. 係→是 is critical because OpenCC
    # in "t2s" mode leaves 係 alone (it's valid Traditional Chinese for 系,
    # but in a Mandarin speaker's transcript it almost certainly means 是).
    ("係", "是"),
    ("佢", "他"),
    ("冇", "没"),
    ("嘅", "的"),
    ("咁", "这样"),
    ("喺", "在"),
    ("嚟", "来"),
    ("睇", "看"),
    ("咗", "了"),
    ("噉", "这样"),
    ("嘢", "东西"),
    ("啲", "些"),
    ("咩", "什么"),
    ("唔", "不"),
    ("啱", "对"),
    # 掂 in a Mandarin context almost always means Cantonese 好/掂晒(=OK/fine).
    # 掂量 is legit Mandarin but vanishingly rare in meeting transcripts; we
    # bias toward cleaning up the more common hallucination case.
    ("掂", "好"),
    ("靚", "好"),
    # Sentence-final particles that have no Mandarin equivalent — just drop
    # them. We're conservative here (only the most common ones) to avoid
    # eating legitimate Mandarin characters.
    ("啫", ""),
    ("囉", ""),
    ("咯", ""),
    ("㗎", ""),
    ("喎", ""),
    # Whisper's silence-hallucinated "Gå in!" and its variants come from
    # language mis-identification as Nynorsk; strip these so they don't
    # pollute diarization or polish.
    ("Gå in!", ""),
    ("Gå in", ""),
)


# OpenCC converter, lazily loaded so mmt works even without the dep.
_OPENCC_CONVERTER = None
_OPENCC_TRIED = False


def _get_opencc_converter():
    """Return an OpenCC Traditional→Simplified converter, or None if unavailable.

    We try ``opencc`` (the canonical C++ bindings) first and fall back to
    ``opencc-python-reimplemented`` (pure Python). Either one is fine; the
    conversion is identical.
    """
    global _OPENCC_CONVERTER, _OPENCC_TRIED
    if _OPENCC_TRIED:
        return _OPENCC_CONVERTER
    _OPENCC_TRIED = True
    try:
        import opencc  # type: ignore
        _OPENCC_CONVERTER = opencc.OpenCC("t2s")
    except Exception:
        _OPENCC_CONVERTER = None
    return _OPENCC_CONVERTER


def normalize_chinese_script(
    text: str,
    chinese_style: str = "simplified",
) -> str:
    """Deterministic safety net for Chinese script normalization.

    Applies after the LLM polish pass to catch anything it missed:
      - Cantonese particles (我哋/嘅/咁/喺/嚟/...) → Mandarin equivalents
        (always attempted; very low false-positive rate because these
        glyphs are not used in written Mandarin).
      - Traditional characters → Simplified via OpenCC when available.
        If OpenCC is not installed this step is silently skipped; the
        LLM polish prompt already asks for Simplified output, so the
        bulk of the conversion happens there.

    No-op when ``chinese_style`` is not ``"simplified"``.
    """
    if chinese_style != "simplified":
        return text
    for src, dst in _CANTONESE_TO_MANDARIN:
        if src in text:
            text = text.replace(src, dst)
    converter = _get_opencc_converter()
    if converter is not None:
        text = converter.convert(text)
    return text


@dataclass
class PolishResult:
    """Outcome of a polish pass, with enough info to log and debug."""
    segments: list[AttributedSegment]
    model_used: str
    batches_polished: int
    batches_failed: int


def resolve_glossary(path_arg: str | None = None) -> tuple[str, str | None]:
    """Load glossary text from CLI arg, env var, or default config path.

    Precedence (highest wins):
        1. Explicit path_arg
        2. $MMT_GLOSSARY
        3. ~/.config/mac-meeting-transcriber/glossary.md

    Returns (glossary_text, path_used). If nothing is found, text is ""
    and path_used is None — the polish pass still runs, just without
    glossary anchoring.
    """
    candidates: list[str] = []
    if path_arg:
        candidates.append(path_arg)
    env_path = os.environ.get("MMT_GLOSSARY")
    if env_path:
        candidates.append(env_path)
    candidates.append(DEFAULT_GLOSSARY_PATH)

    for raw_path in candidates:
        p = Path(raw_path).expanduser()
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8"), str(p)
            except OSError:
                continue
    return "", None


def _batch_segments(
    segments: list[AttributedSegment],
    max_segments: int = MAX_SEGMENTS_PER_BATCH,
    max_chars: int = MAX_CHARS_PER_BATCH,
) -> list[list[tuple[int, AttributedSegment]]]:
    """Split segments into index-tagged batches that fit one LLM turn.

    We cap both the segment count and total char budget because a very
    long run of short back-and-forths is fine, but a single 5000-char
    monologue shouldn't be crammed in alongside 14 other turns.
    """
    batches: list[list[tuple[int, AttributedSegment]]] = []
    current: list[tuple[int, AttributedSegment]] = []
    chars = 0
    for i, seg in enumerate(segments):
        seg_chars = len(seg.text)
        if current and (len(current) >= max_segments or chars + seg_chars > max_chars):
            batches.append(current)
            current = []
            chars = 0
        current.append((i, seg))
        chars += seg_chars
    if current:
        batches.append(current)
    return batches


def _build_messages(
    batch: list[tuple[int, AttributedSegment]],
    glossary: str,
    chinese_style: str = "simplified",
) -> list[dict]:
    items = [
        {"id": i, "speaker": seg.speaker, "text": seg.text}
        for i, seg in batch
    ]
    lines = [
        "You are copy-editing a raw meeting transcript produced by "
        "automatic speech recognition (Whisper).",
        "",
        "Rules — follow ALL of them:",
        "  1. Preserve meaning and all content. Do NOT summarize, rewrite, "
        "paraphrase, or add new information. Do NOT delete substantive words.",
        "  2. Do NOT translate. Keep each utterance in its original "
        "language. English-Chinese code-switching inside a single "
        "utterance is normal — keep it as-is.",
        "  3. Fix obvious ASR mistakes: capitalization of proper nouns "
        "and technical terms, missing sentence-level punctuation, "
        "missing spaces, common homophones, and broken code-switch "
        "boundaries (e.g. 'allhead' → 'all-hands', 'onfreeze' → 'unfreeze').",
        "  4. Apply the glossary below verbatim whenever a term is "
        "clearly meant: use the canonical capitalization and spelling "
        "it prescribes. Do not force glossary terms onto unrelated text.",
        "  5. Keep natural disfluencies (uh, um, 嗯, 对, 然后, you know). "
        "A meeting is not prose. Only remove them if they are ASR "
        "stutters like 'y-y-y-yeah' or '呼呼呼'.",
        "  6. Do NOT split or merge utterances across ids. Each input id "
        "becomes exactly one output id with the same speaker.",
        "  7. If an utterance is already correct, return it unchanged.",
    ]
    if chinese_style == "simplified":
        lines.extend([
            "  8. **Chinese script normalization**: convert ALL Chinese "
            "characters to Simplified Chinese (简体中文). Whisper sometimes "
            "outputs Traditional characters (繁體) or a mix; normalize them "
            "to Simplified even when the source glyph is Traditional. "
            "Examples: 繁體 → 繁体, 這個 → 这个, 們 → 们, 說 → 说, 麼 → 么, 發 → 发.",
            "  9. **Mandarin (Putonghua) only — strip Cantonese glyphs**: "
            "speakers are speaking Mandarin. Whisper occasionally hallucinates "
            "Cantonese-specific characters and particles; rewrite them to "
            "their standard Mandarin equivalents while preserving the meaning. "
            "Common substitutions: 我哋→我们, 你哋→你们, 佢哋→他们, 佢→他, "
            "咁→这样/那样, 嘅→的, 喺→在, 嚟→来, 咗→了, 嚟→来, 噉→这样, "
            "冇→没, 嘢→东西, 點解→为什么, 唔→不, 係→是, 啱→对/刚. "
            "Only do this when the character is clearly a Cantonese particle "
            "in a Mandarin sentence — don't touch legitimate Traditional "
            "characters that also exist in Mandarin.",
        ])
    elif chinese_style == "traditional":
        lines.append(
            "  8. **Chinese script**: keep Chinese characters in Traditional "
            "(繁體) form when they are already Traditional; do not convert to Simplified."
        )
    lines.append("")
    if glossary.strip():
        lines.extend([
            "Glossary (project-specific canonical forms):",
            "```",
            glossary.strip(),
            "```",
            "",
        ])
    lines.extend([
        "Input — a JSON array of utterances (id, speaker, text):",
        "```json",
        json.dumps(items, ensure_ascii=False),
        "```",
        "",
        "Output STRICT JSON matching this schema (no markdown, no commentary):",
        '{"polished": [{"id": <int>, "text": "<edited text>"}, ...]}',
        "",
        "Every input id MUST appear in 'polished' exactly once.",
    ])
    return [
        {
            "role": "system",
            "content": "You are a meticulous meeting-transcript copy editor. You output only valid JSON.",
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def _parse_polished(raw: str, expected_ids: list[int]) -> dict[int, str]:
    cleaned = strip_code_fence(raw)
    if not cleaned:
        raise ValueError("empty response")
    payload = json.loads(cleaned)
    items = payload.get("polished")
    if not isinstance(items, list):
        raise ValueError("'polished' is not a list")
    out: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("id")
        text = item.get("text")
        if isinstance(idx, int) and isinstance(text, str):
            out[idx] = text
    missing = [i for i in expected_ids if i not in out]
    if missing:
        raise ValueError(f"missing {len(missing)} ids (first: {missing[:3]})")
    return out


def _run_one_batch(
    client: OpenAI,
    batch: list[tuple[int, AttributedSegment]],
    glossary: str,
    models_to_try: list[str],
    timeout: float,
    log,
    batch_label: str,
    chinese_style: str = "simplified",
) -> tuple[dict[int, str] | None, str]:
    """Polish a single batch, returning (texts_by_id, model_used) or (None, _)."""
    expected_ids = [i for i, _ in batch]
    messages = _build_messages(batch, glossary, chinese_style=chinese_style)
    for model in models_to_try:
        try:
            raw = chat_json_call(client, model, messages, timeout=timeout)
            polished = _parse_polished(raw, expected_ids)
            return polished, model
        except Exception as exc:
            log(f"  {batch_label} with {model} failed: {exc}")
            continue
    return None, models_to_try[0]


def polish_transcript(
    segments: list[AttributedSegment],
    llm: LLMConfig,
    glossary: str = "",
    timeout: float = 90.0,
    max_workers: int = DEFAULT_MAX_WORKERS,
    log=lambda _msg: None,
    chinese_style: str = "simplified",
) -> PolishResult:
    """Clean up ASR artifacts segment-by-segment using an LLM copy edit.

    Preserves segment boundaries, speakers, and timestamps — only
    ``text`` is rewritten. Splits work into bounded batches and runs
    them in parallel against the LLM; any individual batch that fails
    passes through unchanged, so the pipeline never regresses below
    the raw-Whisper baseline.

    Args:
        segments: Speaker-attributed segments to polish.
        llm: Endpoint + api key + primary model to use.
        glossary: Optional free-form glossary text to inject verbatim.
        timeout: Per-request timeout in seconds.
        max_workers: How many batches to polish concurrently. Each batch
            is an independent LLM call, so this is the main knob for
            wall-clock speed.
        log: Optional progress logger (stderr-style).
        chinese_style: How to handle Chinese characters.
            - ``"simplified"`` (default): normalize Traditional → Simplified,
              and rewrite Cantonese particles (我哋/嘅/咁/喺/...) to their
              Mandarin equivalents. Use for Mandarin meetings.
            - ``"traditional"``: keep Traditional glyphs as-is.
            - ``"auto"``: no script normalization at all.
    """
    if not segments:
        return PolishResult(
            segments=[], model_used=llm.model,
            batches_polished=0, batches_failed=0,
        )

    batches = _batch_segments(segments)
    effective_workers = max(1, min(max_workers, len(batches)))
    log(
        f"  polishing {len(segments)} segments across {len(batches)} batches "
        f"(parallel={effective_workers})"
    )

    client = OpenAI(base_url=llm.base_url, api_key=llm.api_key, timeout=timeout)
    models_to_try = [llm.model] + [m for m in FALLBACK_MODELS if m != llm.model]

    new_texts: dict[int, str] = {}
    batches_polished = 0
    batches_failed = 0
    models_used: list[str] = []
    log_lock = threading.Lock()

    def safe_log(msg: str) -> None:
        with log_lock:
            log(msg)

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        future_to_idx = {
            pool.submit(
                _run_one_batch, client, batch, glossary,
                models_to_try, timeout, safe_log,
                f"batch {bi}/{len(batches)}",
                chinese_style,
            ): bi
            for bi, batch in enumerate(batches, 1)
        }
        for fut in as_completed(future_to_idx):
            bi = future_to_idx[fut]
            try:
                texts, model_used = fut.result()
            except Exception as exc:
                batches_failed += 1
                safe_log(f"  batch {bi}/{len(batches)}: unexpected error ({exc}); originals kept")
                continue
            if texts is None:
                batches_failed += 1
                safe_log(f"  batch {bi}/{len(batches)}: all models failed, originals kept")
                continue
            new_texts.update(texts)
            batches_polished += 1
            models_used.append(model_used)
            safe_log(f"  batch {bi}/{len(batches)}: polished ({model_used})")

    out: list[AttributedSegment] = []
    for i, seg in enumerate(segments):
        new_text = new_texts.get(i, seg.text)
        new_text = normalize_chinese_script(new_text, chinese_style=chinese_style)
        if new_text != seg.text:
            out.append(replace(seg, text=new_text))
        else:
            out.append(seg)

    model_used_report = max(set(models_used), key=models_used.count) if models_used else llm.model
    return PolishResult(
        segments=out,
        model_used=model_used_report,
        batches_polished=batches_polished,
        batches_failed=batches_failed,
    )
