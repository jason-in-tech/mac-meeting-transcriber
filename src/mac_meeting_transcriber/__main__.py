"""CLI entry point for mac-meeting-transcriber.

Output path precedence (highest wins):
    1. -o / --output PATH           # explicit file path; use "-" for stdout
    2. --output-dir DIR             # write <DIR>/<input-stem>.md
    3. $MMT_OUTPUT_DIR              # same, via environment
    4. ./transcripts/<stem>.md      # fallback (CWD-relative)

Usage:
    mmt INPUT.m4a                                # writes to ./transcripts/INPUT.md (or $MMT_OUTPUT_DIR)
    mmt INPUT.m4a -o some/other/path.md          # writes to explicit path
    mmt INPUT.m4a -o -                           # prints Markdown to stdout
    mmt INPUT.m4a --output-dir ~/meeting-notes   # writes to ~/meeting-notes/INPUT.md
    mmt INPUT.m4a --speakers "Jason,Leo"         # LLM-based name matching (default)
    mmt INPUT.m4a --speakers "Jason,Leo" --speakers-positional  # old positional mapping
    mmt INPUT.m4a --speakers "Jason,Leo" --no-identify          # skip LLM, keep SPEAKER_XX
    mmt INPUT.m4a --language zh                  # force language (default: auto-detect, or $MMT_DEFAULT_LANGUAGE)
    mmt INPUT.m4a --primary-language zh          # use BELLE Chinese-fine-tuned Whisper (default: auto from --language)
    mmt INPUT.m4a --chinese-script simplified    # normalize Chinese output (default: simplified, or $MMT_CHINESE_SCRIPT)
    mmt INPUT.m4a --initial-prompt "..."         # Whisper decoding bias (default: auto when --language zh)
    mmt INPUT.m4a --no-initial-prompt            # disable default Chinese-meeting prompt
    mmt INPUT.m4a --no-polish                    # skip the LLM copy-edit pass
    mmt INPUT.m4a --glossary ./glossary.md       # anchor polish with a custom glossary

LLM configuration (shared by --speakers identification and polish):
    --llm-model MODEL            # default: gpt-5.4-medium (via Cursor proxy)
    --llm-base-url URL           # OpenAI-compatible endpoint
    --llm-api-key KEY            # API key
    Env vars: MMT_LLM_{MODEL,BASE_URL,API_KEY} or OPENAI_{BASE_URL,API_KEY}
    $MMT_GLOSSARY                # override default glossary path
    $MMT_OUTPUT_DIR              # override default output directory
    $MMT_INITIAL_PROMPT          # default Whisper decoding prompt
    $MMT_PRIMARY_LANGUAGE        # 'en' (default), 'zh' (picks BELLE), or 'auto'
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from .audio import to_canonical_wav
from .cache import (
    compute_cache_key,
    load_transcript_cache,
    save_transcript_cache,
)
from .diarize import diarize_audio
from .formatter import render_markdown, save_markdown
from .identify import (
    LLMConfig,
    identify_speakers,
    parse_candidates,
    resolve_llm_config,
)
from .merge import collapse_consecutive, merge_transcript_with_speakers
from .models import (
    DEFAULT_EN_MODEL,
    DEFAULT_ZH_MODEL,
    ensure_mlx_model,
    resolve_model,
)
from .polish import (
    normalize_chinese_script,
    polish_transcript,
    resolve_glossary,
)
from .transcribe import filter_hallucinations, is_hallucination, raw_transcribe

# Default Whisper decoding bias for Mandarin-primary technical meetings.
# Whisper's critical failure mode on this user's meetings is silent
# mandarin-to-english translation: when language auto-detect is confused
# (long silent pre-roll, code-switching, noisy intro) Whisper flips into
# the `translate` task and renders Chinese speech as English prose. A
# well-crafted initial_prompt biases the decoder away from that by
# demonstrating the target output style: Simplified Chinese with embedded
# English technical terms, preserving code-switching verbatim.
#
# The prompt is surfaced as a regular meeting snippet (not meta-instructions)
# because Whisper treats initial_prompt as "what came before this audio",
# not as a system message. So we give it a paragraph that looks like the
# tail of an earlier Mandarin-English meeting turn using the team's actual
# vocabulary.
DEFAULT_ZH_MEETING_PROMPT = (
    "以下是一段中英文混合的技术会议转录。说话人使用普通话，"
    "夹杂英文技术术语（如 NVIDIA、PyTorch、FastViT、AP、PR curve、"
    "baseline、training、fine-tune、checkpoint、epoch、labeling、"
    "model、dataset 等）。请用简体中文逐字转录说话人的原话，"
    "保留中英文混合，不要翻译。"
)


def resolve_initial_prompt(
    *,
    no_initial_prompt: bool,
    initial_prompt: str | None,
    initial_prompt_file: str | None,
    env_prompt: str | None,
    language: str | None,
) -> str | None:
    """Pure function for resolving the Whisper decoding bias prompt.

    Precedence (first match wins):
      1. ``--no-initial-prompt``    → ``None``
      2. ``--initial-prompt-file``  → file contents (caller reads; we don't
         touch the filesystem here so this function stays pure + testable)
      3. ``--initial-prompt``       → literal CLI string
      4. ``$MMT_INITIAL_PROMPT``    → env default
      5. ``language == "zh"``       → ``DEFAULT_ZH_MEETING_PROMPT``
      6. otherwise                  → ``None``

    ``initial_prompt_file`` is handled by the caller (it requires I/O and
    can fail); by the time we're called, the caller has already turned
    its contents into ``initial_prompt``. We still accept the flag here
    so we can assert mutual-exclusion consistently in tests.
    """
    if no_initial_prompt:
        return None
    if initial_prompt_file is not None:
        # Caller is expected to have loaded the file and passed the
        # contents via ``initial_prompt``. If they didn't, fall through.
        if initial_prompt is not None:
            return initial_prompt
    if initial_prompt is not None:
        return initial_prompt
    if env_prompt:
        return env_prompt
    if language == "zh":
        return DEFAULT_ZH_MEETING_PROMPT
    return None


def _positional_aliases(names: list[str], segments) -> dict[str, str]:
    """Map names to SPEAKER_XX labels by order of first appearance."""
    first_seen: list[str] = []
    for seg in segments:
        if seg.speaker not in first_seen and seg.speaker != "UNKNOWN":
            first_seen.append(seg.speaker)
    return {sp: names[i] for i, sp in enumerate(first_seen) if i < len(names)}


def _resolve_speaker_aliases(
    raw_names: str | None,
    segments,
    use_llm: bool,
    use_positional: bool,
    llm: LLMConfig | None,
    log,
) -> dict[str, str] | None:
    """Figure out the SPEAKER_XX → real-name map.

    Precedence:
      - --no-identify        : skip naming entirely (keep SPEAKER_XX).
      - --speakers-positional: deterministic first-appearance mapping.
      - default              : ask the LLM. On failure, fall back to positional
                               (better than silently returning SPEAKER_XX).
    """
    if not raw_names or not use_llm:
        return None

    candidates = parse_candidates(raw_names)
    if not candidates:
        return None

    names_only = [c.name for c in candidates]

    if use_positional:
        log(f"Speaker naming: positional (first appearance) → {names_only}")
        return _positional_aliases(names_only, segments)

    if llm is None:
        # Shouldn't happen given the main() logic, but guard anyway.
        log("Speaker naming: no LLM config resolved; falling back to positional")
        return _positional_aliases(names_only, segments)

    hint_preview = ", ".join(
        f"{c.name}:{c.hint}" if c.hint else c.name for c in candidates
    )
    log(f"Speaker naming: asking LLM to match [{hint_preview}] by content...")

    try:
        t0 = time.time()
        result = identify_speakers(segments, candidates, llm, log=log)
        log(
            f"  identified in {time.time() - t0:.1f}s "
            f"(model={result.model_used}, confidence={result.confidence:.2f})"
        )
        log(f"  mapping: {result.mapping}")
        if result.reason:
            log(f"  reason: {result.reason}")
        return result.mapping
    except Exception as exc:
        log(f"  LLM identification failed ({exc}); falling back to positional mapping")
        return _positional_aliases(names_only, segments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmt",
        description="Fast Apple-Silicon-native meeting transcription with speaker diarization.",
    )
    parser.add_argument("input", help="Path to audio file (m4a, mp3, wav, mp4, ...)")
    parser.add_argument(
        "-o", "--output",
        help="Explicit output .md file path. Use '-' for stdout. "
             "Overrides --output-dir / $MMT_OUTPUT_DIR. "
             "Default falls back to <output-dir>/<input-stem>.md.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write <input-stem>.md into. "
             "Default: $MMT_OUTPUT_DIR if set, else ./transcripts/. "
             "Ignored when -o is an explicit path or '-'.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g. zh, en) passed to Whisper as a "
             "decoding hint. This is a per-segment decoder flag, NOT a "
             "model selector — see --primary-language for the latter. "
             "Default: $MMT_DEFAULT_LANGUAGE if set, else Whisper "
             "auto-detect. Auto-detect only samples the first 30 s of "
             "audio, so on files with a long silent pre-roll Whisper may "
             "mis-identify the language; setting this explicitly (or "
             "exporting MMT_DEFAULT_LANGUAGE=zh) is the robust fix.",
    )
    parser.add_argument(
        "--primary-language",
        choices=("auto", "en", "zh"),
        default=None,
        help="Which ASR model to load. 'en' (the default; 'auto' is an "
             "alias) uses OpenAI multilingual whisper-large-v3 — our "
             "recommended pick even for Mandarin-primary meetings as "
             "long as there's any code-switched English. 'zh' swaps in "
             "BELLE-whisper-large-v3-zh-punct, a Chinese LoRA fine-tune "
             "that roughly halves CER on pure-Mandarin meeting audio "
             "(WenetSpeech-meeting 10.97 vs 20.15) but transliterates "
             "English technical terms into same-sound Chinese glyphs, "
             "so it only pays off when the audio has no English terms. "
             "BELLE auto-downloads + converts to MLX on first use "
             "(~10 min, ~3 GB cached afterwards). Override default via "
             "$MMT_PRIMARY_LANGUAGE. Use --model to point at an arbitrary "
             "HF repo.",
    )
    parser.add_argument(
        "--chinese-script",
        choices=("simplified", "traditional", "auto"),
        default=None,
        help="Normalize Chinese characters in the polished output. "
             "'simplified' (the default) converts Traditional → Simplified "
             "and rewrites hallucinated Cantonese particles (我哋/嘅/咁/...) "
             "to their Mandarin equivalents. 'traditional' keeps Traditional "
             "glyphs as-is. 'auto' disables script normalization entirely. "
             "Override default via $MMT_CHINESE_SCRIPT.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--initial-prompt",
        default=None,
        help="Text fed to Whisper as a decoding bias (called initial_prompt "
             "in whisper). Strongly recommended for Mandarin-English "
             "technical meetings — it prevents the failure mode where "
             "Whisper silently translates Mandarin speech into English. "
             "When omitted and --language is zh (or MMT_DEFAULT_LANGUAGE=zh), "
             "a built-in Chinese-meeting prompt is used automatically; pass "
             "--no-initial-prompt to disable that default. Override default "
             "via $MMT_INITIAL_PROMPT.",
    )
    prompt_group.add_argument(
        "--initial-prompt-file",
        default=None,
        help="Read --initial-prompt from a UTF-8 text file. Useful for "
             "long domain-specific prompts you want to reuse across runs.",
    )
    prompt_group.add_argument(
        "--no-initial-prompt",
        action="store_true",
        help="Disable the built-in Chinese-meeting prompt that kicks in "
             "when --language is zh. Use when the automatic prompt is "
             "biasing the decoder in an unhelpful way.",
    )
    parser.add_argument(
        "--speakers",
        default=None,
        help=(
            "Comma-separated candidate speaker names, e.g. \"Jason,Leo\". "
            "Optionally attach a role hint after a colon to help the LLM "
            "match names to speakers: \"Jason:IC,Leo:manager\" or "
            "\"Jason:recording the call,Leo:Jason's manager\". "
            "Hints are strongly recommended when nobody says each other's "
            "names in the audio. Built-in shorthands: ic, mgr/manager, "
            "skip, peer, report, me/recorder."
        ),
    )
    parser.add_argument(
        "--speakers-positional",
        action="store_true",
        help="Assign --speakers in strict order of first appearance, skipping "
             "the LLM. Requires --speakers; conflicts with --no-identify.",
    )
    parser.add_argument(
        "--no-identify",
        action="store_true",
        help="Keep anonymous SPEAKER_NN labels as Senko produced them. "
             "Conflicts with --speakers and --speakers-positional.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model for speaker identification (default: gpt-5.4-medium, "
             "or $MMT_LLM_MODEL).",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible endpoint for speaker identification "
             "(default: http://127.0.0.1:8765/v1 Cursor proxy, "
             "or $MMT_LLM_BASE_URL / $OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="API key for the LLM endpoint "
             "(default: 'cursor' for the local Cursor proxy, "
             "or $MMT_LLM_API_KEY / $OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--no-polish",
        action="store_true",
        help="Skip the LLM copy-edit pass that fixes capitalization, "
             "punctuation, and glossary terms. Useful if the LLM "
             "endpoint is unavailable or you want raw Whisper text.",
    )
    parser.add_argument(
        "--glossary",
        default=None,
        help="Path to a glossary file (free-form text/markdown) with "
             "project-specific canonical spellings. "
             "Default: ~/.config/mac-meeting-transcriber/glossary.md, "
             "or $MMT_GLOSSARY if set. Incompatible with --no-polish.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit HuggingFace repo ID or local path for the Whisper "
             "model. Escape hatch — normally you pick via --primary-language. "
             "Accepts MLX-native repos (e.g. mlx-community/whisper-large-v3-mlx) "
             "and HF Transformers repos like BELLE-2/Belle-whisper-large-v3-zh-punct, "
             "which get auto-converted + cached on first use. "
             "Default: derived from --primary-language "
             f"(en → {DEFAULT_EN_MODEL}; zh → {DEFAULT_ZH_MODEL}).",
    )
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        help="Keep Whisper's original segment granularity instead of merging into paragraphs.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the Whisper transcript cache (force fresh transcription).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Stream per-stage progress to stderr.",
    )

    args = parser.parse_args(argv)

    log = (lambda msg: print(f"[mmt] {msg}", file=sys.stderr)) if args.verbose else (lambda _msg: None)

    if args.no_identify and args.speakers:
        print(
            "error: --no-identify and --speakers are mutually exclusive "
            "(pass candidate names OR opt out, not both).",
            file=sys.stderr,
        )
        return 2
    if args.no_identify and args.speakers_positional:
        print(
            "error: --no-identify and --speakers-positional are mutually exclusive.",
            file=sys.stderr,
        )
        return 2
    if args.speakers_positional and not args.speakers:
        print(
            "error: --speakers-positional requires --speakers to specify the names.",
            file=sys.stderr,
        )
        return 2
    if args.no_polish and args.glossary:
        print(
            "error: --no-polish and --glossary are mutually exclusive "
            "(a glossary only matters during the polish pass).",
            file=sys.stderr,
        )
        return 2

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"error: audio not found: {input_path}", file=sys.stderr)
        return 1

    effective_language = args.language or os.environ.get("MMT_DEFAULT_LANGUAGE") or None
    if effective_language and args.language is None:
        log(f"Using MMT_DEFAULT_LANGUAGE={effective_language} (set --language to override)")

    # Resolve --primary-language to a concrete model pick. 'auto' (or
    # unset) derives from --language so a single MMT_DEFAULT_LANGUAGE=zh
    # export picks BELLE without any per-call flag.
    primary_language_arg = (
        args.primary_language
        or os.environ.get("MMT_PRIMARY_LANGUAGE")
        or "auto"
    )
    if primary_language_arg not in ("auto", "en", "zh"):
        print(
            f"error: invalid --primary-language: {primary_language_arg!r} "
            "(expected one of auto, en, zh)",
            file=sys.stderr,
        )
        return 2
    # 'auto' == 'en'. We deliberately do NOT derive 'zh' from --language
    # because BELLE's Chinese fine-tune delivers its big CER wins only on
    # pure-Mandarin audio; on code-switched technical meetings (the
    # common case here) it *loses* to multilingual whisper-large-v3 — it
    # transliterates English terms into same-sound Chinese characters
    # ("baseline" → "贝斯兰", "FastViT" → "发生VT") and degenerates into
    # token-repetition loops on silent lead-ins. Users who genuinely
    # want BELLE can pass --primary-language zh explicitly (or export
    # MMT_PRIMARY_LANGUAGE=zh).
    if primary_language_arg == "auto":
        effective_primary_language = "en"
    else:
        effective_primary_language = primary_language_arg

    # resolve_model returns an HF repo ID or local path. ensure_mlx_model
    # then makes sure that path is in MLX format — it's a no-op for
    # mlx-community/* repos and runs the HF→MLX conversion once (cached)
    # for BELLE / other HF Transformers checkpoints.
    effective_model_logical = resolve_model(
        effective_primary_language, args.model,
    )
    log(
        f"Model: {effective_model_logical} "
        f"(primary-language={effective_primary_language}"
        f"{', via --model override' if args.model else ''})"
    )

    effective_chinese_script = (
        args.chinese_script
        or os.environ.get("MMT_CHINESE_SCRIPT")
        or "simplified"
    )
    if effective_chinese_script not in ("simplified", "traditional", "auto"):
        print(
            f"error: invalid chinese-script value: {effective_chinese_script!r} "
            "(expected one of simplified, traditional, auto)",
            file=sys.stderr,
        )
        return 2

    # Turn --initial-prompt-file into a string up front so resolve_initial_prompt
    # stays pure (easier to unit-test, no filesystem).
    cli_initial_prompt = args.initial_prompt
    if args.initial_prompt_file:
        try:
            cli_initial_prompt = (
                Path(args.initial_prompt_file).expanduser().read_text(encoding="utf-8").strip()
            )
        except OSError as exc:
            print(f"error: cannot read --initial-prompt-file: {exc}", file=sys.stderr)
            return 1

    effective_initial_prompt = resolve_initial_prompt(
        no_initial_prompt=args.no_initial_prompt,
        initial_prompt=cli_initial_prompt,
        initial_prompt_file=args.initial_prompt_file,
        env_prompt=os.environ.get("MMT_INITIAL_PROMPT"),
        language=effective_language,
    )

    if args.no_initial_prompt:
        log("Initial prompt: disabled by --no-initial-prompt")
    elif args.initial_prompt_file:
        log(
            f"Initial prompt: loaded from {args.initial_prompt_file} "
            f"({len(effective_initial_prompt or '')} chars)"
        )
    elif args.initial_prompt is not None:
        log(f"Initial prompt: explicit ({len(args.initial_prompt)} chars)")
    elif os.environ.get("MMT_INITIAL_PROMPT"):
        log(
            f"Initial prompt: from MMT_INITIAL_PROMPT "
            f"({len(os.environ['MMT_INITIAL_PROMPT'])} chars)"
        )
    elif effective_initial_prompt is not None:
        log(
            "Initial prompt: using built-in Chinese-meeting default "
            "(pass --no-initial-prompt to disable)"
        )

    # Senko needs 16kHz mono 16-bit WAV, so we make one canonical copy
    # and feed it to both Whisper and Senko for perfectly aligned timing.
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        log("Converting audio to 16 kHz mono WAV...")
        t0 = time.time()
        to_canonical_wav(input_path, wav_path)
        log(f"  done in {time.time() - t0:.1f}s")

        cache_key = None
        raw_segments = None
        if not args.no_cache:
            # Key on the *logical* model (HF repo ID) rather than the
            # physical MLX cache path so a re-converted BELLE doesn't
            # invalidate the transcript cache.
            cache_key = compute_cache_key(
                input_path,
                effective_model_logical,
                effective_language,
                effective_initial_prompt,
            )
            raw_segments = load_transcript_cache(cache_key)
            if raw_segments is not None:
                log(f"Loaded {len(raw_segments)} cached Whisper segments (key: {cache_key[:12]}...)")

        if raw_segments is None:
            # This may kick off a one-time ~3-5 min conversion if the
            # user picked a BELLE variant and it's not cached yet.
            effective_model_path = ensure_mlx_model(effective_model_logical, log=log)
            log(f"Transcribing with mlx-whisper (model: {effective_model_logical})...")
            t0 = time.time()
            raw_segments = raw_transcribe(
                wav_path,
                model=effective_model_path,
                language=effective_language,
                initial_prompt=effective_initial_prompt,
                verbose=False,
            )
            log(f"  transcribed {len(raw_segments)} raw segments in {time.time() - t0:.1f}s")
            if cache_key is not None:
                save_transcript_cache(cache_key, raw_segments)

        transcript = filter_hallucinations(raw_segments)
        log(f"  {len(transcript)} segments after hallucination filter")

        log("Diarizing with Senko (CoreML)...")
        t0 = time.time()
        speakers = diarize_audio(wav_path, warmup=False, quiet=True)
        log(f"  found {len({s.speaker for s in speakers})} speakers in {time.time() - t0:.1f}s")

        log("Merging transcript with speakers...")
        attributed = merge_transcript_with_speakers(transcript, speakers)
        if not args.no_collapse:
            attributed = collapse_consecutive(attributed)
        log(f"  {len(attributed)} final paragraphs")

    # Resolve the LLM config once and share it between polish + identify,
    # so both stages log the same endpoint/model and we don't re-read env
    # vars mid-run.
    needs_polish = not args.no_polish
    needs_identify = bool(args.speakers) and not args.no_identify and not args.speakers_positional
    llm_config: LLMConfig | None = None
    if needs_polish or needs_identify:
        llm_config = resolve_llm_config(args.llm_base_url, args.llm_api_key, args.llm_model)
        log(f"LLM endpoint {llm_config.base_url} with model {llm_config.model}")

    if needs_polish:
        glossary_text, glossary_path = resolve_glossary(args.glossary)
        if glossary_path:
            log(f"Polishing with glossary: {glossary_path}")
        else:
            log("Polishing (no glossary found — fixing caps/punct/homophones only)")
        if effective_chinese_script == "simplified":
            log("  Chinese script: simplified (Traditional → Simplified + Cantonese particles → Mandarin)")
        elif effective_chinese_script == "traditional":
            log("  Chinese script: traditional (kept as-is)")
        else:
            log("  Chinese script: auto (no normalization)")
        t0 = time.time()
        try:
            polish_result = polish_transcript(
                attributed, llm_config, glossary=glossary_text, log=log,
                chinese_style=effective_chinese_script,
            )
            attributed = polish_result.segments
            log(
                f"  polished {polish_result.batches_polished} batches "
                f"({polish_result.batches_failed} failed) "
                f"in {time.time() - t0:.1f}s using {polish_result.model_used}"
            )
        except Exception as exc:
            log(f"  polish failed ({exc}); keeping unpolished transcript")
    elif effective_chinese_script == "simplified":
        log("Chinese script: applying deterministic simplified-Mandarin sweep (polish skipped)")
        from dataclasses import replace as _replace
        attributed = [
            _replace(seg, text=normalize_chinese_script(seg.text, chinese_style="simplified"))
            for seg in attributed
        ]

    # Re-filter after polish/normalization: the Cantonese / Gå-in sweep
    # can empty out short hallucinated segments, and the LLM sometimes
    # produces trivial output we want to drop too.
    before = len(attributed)
    attributed = [seg for seg in attributed if not is_hallucination(seg.text)]
    if len(attributed) != before:
        log(f"Post-polish filter: dropped {before - len(attributed)} empty/hallucinated segment(s)")

    aliases = _resolve_speaker_aliases(
        raw_names=args.speakers,
        segments=attributed,
        use_llm=not args.no_identify,
        use_positional=args.speakers_positional,
        llm=llm_config,
        log=log,
    )
    markdown = render_markdown(
        attributed,
        audio_name=input_path.stem,
        speaker_aliases=aliases,
    )

    if args.output == "-":
        sys.stdout.write(markdown)
    else:
        if args.output:
            out_path_arg = Path(args.output).expanduser()
        else:
            out_dir_raw = args.output_dir or os.environ.get("MMT_OUTPUT_DIR")
            base_dir = Path(out_dir_raw).expanduser() if out_dir_raw else Path.cwd() / "transcripts"
            out_path_arg = base_dir / f"{input_path.stem}.md"
        out_path = save_markdown(markdown, out_path_arg)
        log(f"Wrote {out_path}")
        print(str(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
