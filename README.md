# mac-meeting-transcriber

[![CI](https://github.com/jason-in-tech/mac-meeting-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/jason-in-tech/mac-meeting-transcriber/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%E2%80%93M4-success.svg)](#requirements)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Fast, accurate, on-device meeting transcription for Apple Silicon — with speaker diarization and optional LLM polish, all in one `mmt` command.**

A 60-minute Mandarin-English meeting on an M4 Pro becomes a speaker-attributed Markdown file in **~12 minutes** (cold) or **~3 minutes** (warm cache). ASR + diarization run 100% locally; LLM polish and speaker identification are optional and use any OpenAI-compatible endpoint.

```bash
mmt recording.m4a --speakers "Alex:engineer,Pat:manager"
# → ./transcripts/recording.md
```

| Stage                         | Library                                                                                          | Runs on                |
| ----------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------- |
| Speech-to-text                | [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) large-v3 / BELLE   | Apple GPU (Metal/MLX)  |
| Diarization                   | [`senko`](https://github.com/narcotic-sh/senko) (pyannote seg-3 + CAM++)                         | Neural Engine (CoreML) |
| Polish *(optional)*           | OpenAI-compatible LLM                                                                            | Network                |
| Speaker identification *(opt)* | same LLM                                                                                         | Network                |

`--no-polish --no-identify` for a fully offline run.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Highlights](#highlights)
- [Requirements](#requirements)
- [Install](#install)
- [Quickstart](#quickstart)
- [Common flags](#common-flags)
- [Language detection trap](#language-detection-trap)
- [ASR model selection (`large-v3` vs BELLE)](#asr-model-selection-large-v3-vs-belle)
- [Output directory](#output-directory)
- [Polish pass](#polish-pass)
- [Speaker identification](#speaker-identification)
- [LLM endpoint](#llm-endpoint)
- [Performance](#performance)
- [Project layout](#project-layout)
- [Privacy & telemetry](#privacy--telemetry)
- [FAQ](#faq)
- [Known limits](#known-limits)
- [Development](#development)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Why this exists

Most meeting tools either:

- send your audio to a third-party server (Otter.ai, Fireflies, Granola, ...) and you trust them with your audio, or
- run Whisper locally but stop at "raw text", with no speaker info, no punctuation polish, and no English/Chinese term capitalization.

`mac-meeting-transcriber` runs the entire ASR + diarization pipeline **locally on the Apple GPU + Neural Engine**, with a thoughtful polish layer on top that fixes the things even better ASR can't:

- code-switched English technical terms get correctly capitalized (`fastvit` → `FastViT`, `nvidia` → `NVIDIA`)
- Cantonese particles in pure-Mandarin recordings get rewritten to Mandarin (`我哋` → `我们`, `嘅` → `的`)
- Whisper's "translate everything to English when in doubt" mis-detection is suppressed via initial-prompt biasing
- speakers `SPEAKER_00`, `SPEAKER_01`, ... are mapped to real names by an LLM that reads the transcript and uses your role hints

Everything is one CLI command, with sensible defaults and a thorough cache so re-running with different polish/glossary settings is ~15 seconds, not 9 minutes.

## Highlights

- **100% local ASR + diarization.** No audio leaves your machine for transcription/diarization.
- **Apple Silicon native.** MLX Whisper on the GPU, Senko diarization on the Neural Engine.
- **Robust language handling.** Built-in Chinese decoding bias, Cantonese-particle sweep, OpenCC Traditional → Simplified, configurable script targets.
- **Smart caching.** Transcript is cached by `(audio + model + language + initial-prompt)`. Re-running with new polish/glossary/speaker settings reuses the cached transcript and skips Whisper entirely.
- **Optional LLM polish + speaker ID.** Use any OpenAI-compatible endpoint; defaults to the local Cursor proxy. Both stages can be disabled for fully offline runs.
- **Glossary-aware.** Inject project-specific vocabulary into the polish prompt via flag, env var, or `~/.config/mac-meeting-transcriber/glossary.md`.
- **Streamable.** Output to a file, a directory, or stdout (`-o -`).
- **113 unit tests, ruff-clean, runs in CI.**

## Requirements

- macOS 14+ on Apple Silicon (M1 / M2 / M3 / M4)
- Xcode command-line tools: `xcode-select --install`
- [`uv`](https://docs.astral.sh/uv/) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Python 3.13 — `uv` will fetch it for you

For LLM polish and speaker identification (optional): an OpenAI-compatible endpoint. Tested with the local Cursor proxy and OpenAI directly; should work with any provider that exposes an OpenAI-shaped Chat Completions API (Together, Groq, OpenRouter, Ollama, vLLM, ...).

## Install

```bash
git clone https://github.com/jason-in-tech/mac-meeting-transcriber.git
cd mac-meeting-transcriber
uv sync
```

`uv sync` creates `.venv`, installs all deps, and lazy-pulls the `whisper-large-v3-mlx` weights on the first transcription run (~3 GB).

## Quickstart

```bash
# Basic run — writes to $MMT_OUTPUT_DIR/<stem>.md, defaulting to ./transcripts/<stem>.md
mmt recording.m4a

# Recommended — name speakers with role hints
mmt recording.m4a --speakers "Alex:IC,Pat:manager" -v

# Fully offline (no LLM polish, no LLM speaker ID)
mmt recording.m4a --no-polish --no-identify

# Output anywhere
mmt recording.m4a -o ~/Docs/meeting.md      # explicit file
mmt recording.m4a --output-dir ~/notes      # dir + auto-named file
mmt recording.m4a -o -                      # stdout (great for piping into grep/sed/jq)

# Force Mandarin (recommended for any meeting that's mostly Chinese)
mmt recording.m4a --language zh

# Use a project-specific glossary
mmt recording.m4a --glossary ~/configs/glossary.md
```

`mmt --help` prints the full flag list.

## Common flags

```text
--no-polish                          # skip LLM copy-edit (fully offline)
--no-identify                        # keep anonymous SPEAKER_XX labels
--no-cache                           # force a fresh transcription
--glossary PATH                      # project-specific vocabulary
--language zh|en                     # force language (default: $MMT_DEFAULT_LANGUAGE or auto)
--primary-language en|zh             # pick ASR model (default: en → whisper-large-v3)
--chinese-script simplified|traditional|auto   # normalize Chinese output (default: simplified)
--initial-prompt "..."               # bias Whisper decoding (auto-applied when language=zh)
--initial-prompt-file PATH           # load a long/shared prompt from disk
--no-initial-prompt                  # disable the built-in Chinese-meeting default
--llm-model MODEL                    # override LLM model (default: $MMT_LLM_MODEL)
--llm-base-url URL                   # override LLM endpoint
--llm-api-key KEY                    # override LLM API key
-v / --verbose                       # show timings and pipeline stage logs
```

## Language detection trap

Whisper auto-detects language from the **first 30 seconds**, including silence. Files with a long silent pre-roll (Voice Memos share flows, recordings that start with "uh, hello?, ... 大家好") regularly mis-detect as Nynorsk / Vietnamese / German and then **silently flip into the `translate` task — rendering Mandarin speech as fluent English prose, with no warning whatsoever.**

Two mitigations, in order:

```bash
# Best: pin the language explicitly per run
mmt recording.m4a --language zh

# Or: set a global default in your shell rc
export MMT_DEFAULT_LANGUAGE="zh"
export MMT_CHINESE_SCRIPT="simplified"
export MMT_INITIAL_PROMPT="..."   # optional: override the built-in Chinese-meeting prompt
```

When `--language zh` is active (explicitly or via env), `mmt` automatically attaches a built-in `initial_prompt` that demonstrates "Mandarin-English code-switched technical meeting", which anchors the decoder to the right language and keeps technical terms verbatim. Disable with `--no-initial-prompt` or override with `--initial-prompt "..."` / `--initial-prompt-file`.

The prompt is part of the cache key, so different prompts never share a cached transcript.

## ASR model selection (`large-v3` vs BELLE)

The default `--primary-language en` loads **`mlx-community/whisper-large-v3-mlx`** (OpenAI multilingual). Counter-intuitively this is also our recommended pick for **Mandarin-primary meetings that contain any English technical terms** (Chinese-speaking engineering teams, ML research chats, mixed cross-functional meetings). v3's multilingual encoder preserves code-switched English terms verbatim (`FastViT`, `NVIDIA`, `PR curve`, `baseline`, `checkpoint`, ...).

`--primary-language zh` swaps in **`BELLE-2/Belle-whisper-large-v3-zh-punct`**, a Chinese LoRA fine-tune of whisper-large-v3.

| Recording type                                     | Recommended            |
| -------------------------------------------------- | ---------------------- |
| Mandarin + English code-switching (most ML/eng meetings) | **`--primary-language en`** (default — v3) |
| Pure Mandarin, no technical terms                  | `--primary-language zh` (BELLE)   |
| Mandarin variants (Cantonese, Taiwanese, HK)       | `--primary-language en` + `--chinese-script traditional` |
| English-only                                       | default                |

On pure Chinese, BELLE roughly halves CER on meeting-style audio (WenetSpeech-meeting CER 10.97 vs 20.15 for base v3). On code-switched recordings it *regresses* — BELLE transliterates English terms into same-sound Chinese glyphs (`baseline` → 贝斯兰, `FastViT` → 发生VT, `labeling spec` → 雷布林斯的) and is more prone to token-repetition loops on silent lead-ins. The fine-tune data is 100% Mandarin, so its multilingual head is weaker than v3's.

BELLE is not pre-converted to MLX. On first use `mmt` downloads the HuggingFace Transformers checkpoint (~3 GB safetensors) and runs a one-shot HF → MLX conversion; the result is cached under `~/.cache/mac-meeting-transcriber/mlx-models/<repo>/` (~3 GB fp16). Subsequent runs load instantly. Use `--model <repo>` to point at any other HF Whisper checkpoint; pre-converted `mlx-community/*` repos skip conversion.

## Output directory

Resolution order: `-o PATH` → `--output-dir DIR` → `$MMT_OUTPUT_DIR` → `./transcripts/`.

Recommended shell-rc line so every run lands in one place:

```bash
export MMT_OUTPUT_DIR="$HOME/Documents/meetings"
```

## Polish pass

Runs by default. Fixes what no stronger ASR can:

- **Capitalization of technical terms** (`fastvit` → `FastViT`, `nvidia` → `NVIDIA`)
- **Code-switch boundaries** (`allhead` → `all-hands`, `onfreeze` → `unfreeze`)
- **Punctuation and homophones**
- **Preserves every timestamp and speaker label**

Batched 8-way in parallel (~2 min for a 60-min meeting); failed batches pass through unchanged. Results are deterministic when paired with the same `--llm-model` + temperature.

**Chinese script normalization** piggybacks on polish. When `--chinese-script simplified` (default) is active:

1. The polish prompt explicitly asks the LLM to output Simplified Chinese and rewrite any Cantonese particles (`我哋`, `嘅`, `咁`, `喺`, `嚟`, `睇`, `係`, ...) into Mandarin equivalents.
2. A deterministic post-polish sweep applies the same Cantonese-to-Mandarin mappings and runs [`opencc`](https://github.com/BYVoid/OpenCC) `t2s` for thorough Traditional → Simplified conversion. OpenCC is a soft dependency — if not installed, the prompt + hand-rules still catch the common cases.

Pass `--chinese-script traditional` for true Cantonese / Taiwan / Hong Kong Mandarin recordings, or `--chinese-script auto` to disable normalization entirely.

**Glossary** (optional, dramatically improves term accuracy) — any free-form markdown injected into the prompt. Lookup order: `--glossary PATH` → `$MMT_GLOSSARY` → `~/.config/mac-meeting-transcriber/glossary.md`.

```markdown
# Glossary
- FastViT, MobileViT, EfficientNet (uppercase, hyphenated as shown)
- People: Alex (engineer), Pat (PM), Sam (designer)
- "all-hands" not "allhead"; "unfreeze" not "onfreeze"
- "code review" not "code reveal"
```

## Speaker identification

Senko returns `SPEAKER_00`, `SPEAKER_01`, ... `--speakers "Name:hint,..."` asks the LLM to match them by transcript content.

- **Hints recommended** — especially when speakers don't say each other's names.
  Built-in shorthands: `ic`, `mgr` / `manager`, `skip`, `peer`, `report`, `me` / `recorder`.
  Or free-form: `"Alex:Pat's manager giving direction"`.
- `--speakers-positional` — deterministic first-appearance mapping (fragile).
- `--no-identify` — keep anonymous labels.

On LLM failure, falls back to positional mapping automatically.

## LLM endpoint

Both polish and speaker-ID share one OpenAI-compatible endpoint.

|              | CLI flag         | Env var                                   | Default                     |
| ------------ | ---------------- | ----------------------------------------- | --------------------------- |
| Model        | `--llm-model`    | `MMT_LLM_MODEL`                           | `gpt-5.4-medium`            |
| Base URL     | `--llm-base-url` | `MMT_LLM_BASE_URL`, `OPENAI_BASE_URL`     | `http://127.0.0.1:8765/v1`  |
| API key      | `--llm-api-key`  | `MMT_LLM_API_KEY`, `OPENAI_API_KEY`       | `cursor`                    |

Examples:

```bash
# OpenAI directly
export OPENAI_API_KEY="sk-..."
export MMT_LLM_BASE_URL="https://api.openai.com/v1"
export MMT_LLM_MODEL="gpt-4o-mini"
mmt recording.m4a

# Local Ollama
export MMT_LLM_BASE_URL="http://localhost:11434/v1"
export MMT_LLM_API_KEY="ollama"
export MMT_LLM_MODEL="qwen2.5:14b"
mmt recording.m4a
```

## Performance

66-minute Chinese-English recording on an M4 Pro (16-core GPU):

| Config                              | Cold      | Warm (cache hit) |
| ----------------------------------- | --------- | ---------------- |
| Default (polish + identify)         | ~12 min   | ~3 min           |
| `--no-polish --no-identify`         | ~9 min    | ~15 s            |

Whisper output is cached in `~/.cache/mac-meeting-transcriber/` keyed by `audio + model + language + initial_prompt`. Changing any of those produces a fresh transcription. The Markdown file itself is always regenerated, so polish, speaker ID, and glossary changes take effect without `--no-cache`.

Switching to BELLE for the first time adds a one-shot ~10 min HF download + HF→MLX conversion (~3 GB cached under `~/.cache/mac-meeting-transcriber/mlx-models/`); subsequent BELLE runs transcribe at the same speed as v3.

## Project layout

```
mac-meeting-transcriber/
├── src/mac_meeting_transcriber/
│   ├── __main__.py        # CLI entry point + argparse
│   ├── audio.py           # ffmpeg → canonical 16 kHz mono WAV
│   ├── transcribe.py      # MLX Whisper wrapper, hallucination filter
│   ├── diarize.py         # Senko wrapper (pyannote seg-3 + CAM++)
│   ├── merge.py           # transcript + speakers → AttributedSegments
│   ├── polish.py          # LLM copy-edit + Chinese script normalization
│   ├── identify.py        # LLM speaker-name resolution
│   ├── formatter.py       # AttributedSegments → Markdown
│   ├── cache.py           # on-disk transcript cache (audio+model+lang+prompt)
│   └── models.py          # HF → MLX one-shot Whisper conversion
├── tests/                 # 113 unit tests, no network, run in <2s
├── examples/basic_usage.py
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Privacy & telemetry

- **No telemetry.** The tool never phones home or reports usage.
- **Audio stays local for ASR + diarization.** No upload, no third-party processing.
- **Polish + speaker-ID send transcript text to your configured LLM endpoint.** This is your choice — point it at a local Ollama / vLLM / llama.cpp server, or run with `--no-polish --no-identify` for a fully offline pipeline.
- **No glossary or recording metadata is committed back to anywhere.** The cache is purely on-disk in `~/.cache/`.

If your meeting content is sensitive, the recommended setup is a local LLM:

```bash
export MMT_LLM_BASE_URL="http://localhost:11434/v1"   # Ollama
export MMT_LLM_API_KEY="ollama"
export MMT_LLM_MODEL="qwen2.5:14b"
mmt recording.m4a
```

## FAQ

**Why is the default LLM endpoint `http://127.0.0.1:8765/v1`?**
That's the Cursor desktop app's local proxy. If you're not running Cursor, override `MMT_LLM_BASE_URL` to point at OpenAI / Ollama / etc. The pipeline works fine with any OpenAI-compatible API.

**Why Python 3.13?**
Pure preference (cleaner `from datetime import UTC` etc.). Dropping back to 3.11 is straightforward — most of the type hints already use the modern union syntax, and the only 3.13-specific stdlib usage is in tests.

**Will it work on Intel Macs / Linux / Windows?**
ASR uses `mlx-whisper`, which is **Apple-Silicon only**. Diarization (`senko`) is also tuned for Apple's Neural Engine. The polish, identify, formatter, and cache layers are pure Python and would work cross-platform if you swap in a non-MLX Whisper backend (e.g. `faster-whisper`). PRs welcome.

**Why is my Mandarin output coming out in English?**
You probably hit the language-detection trap (see [the dedicated section above](#language-detection-trap)). Either pass `--language zh` explicitly, or `export MMT_DEFAULT_LANGUAGE=zh` in your shell rc.

**Why is my code-switched English getting transliterated to Chinese?**
You're on `--primary-language zh` (BELLE). Switch back to `--primary-language en` (the default) — v3's multilingual head preserves English terms verbatim.

**The cache is huge — how do I clear it?**
```bash
rm -rf ~/.cache/mac-meeting-transcriber/
```
You'll redownload the model on the next run.

**Can I batch-transcribe a folder?**
There's no built-in batch flag, but it's a one-liner:
```bash
for f in ~/Recordings/*.m4a; do mmt "$f"; done
```

## Known limits

- **Apple-Silicon only at runtime.** No CUDA / Intel-Mac / Windows support today.
- **Best on English + Mandarin.** Other languages work but the polish prompt and Cantonese-particle sweep are tuned for Mandarin.
- **Overlapping speech is attributed to a single dominant speaker** — that's a Senko / CAM++ limitation, not something we can paper over downstream.
- **Low-quality audio degrades diarization more than transcription.** A noisy recording with crisp speech will still transcribe well but mis-cluster speakers.

## Development

```bash
uv sync --extra dev

# Tests — 113 unit tests, all pure Python, ~1.5 s, no network
uv run pytest -v

# Lint
uv run ruff check src tests examples

# Run the CLI from the repo
uv run mmt path/to/audio.m4a -v
```

CI runs lint + tests on every push / PR — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

PRs especially welcome for:

- Linux / Intel-Mac backends (faster-whisper + a non-Senko diarizer)
- Additional polish languages (Spanish, Japanese, ...) — the polish prompt is the only piece that needs a per-language variant
- Better overlapping-speech handling
- Whisper.cpp / faster-whisper benchmarks alongside the MLX numbers

## Acknowledgments

- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — the Apple MLX team's MLX port of Whisper
- [Senko](https://github.com/narcotic-sh/senko) — Hamza Qayum's modern macOS-native pyannote-style diarizer
- [FluidAudio](https://github.com/FluidInference/FluidAudio) — CoreML seg-3 conversion used by Senko
- [OpenAI Whisper](https://github.com/openai/whisper) — the original
- [BELLE-2](https://huggingface.co/BELLE-2) — Mandarin LoRA fine-tune of whisper-large-v3
- [OpenCC](https://github.com/BYVoid/OpenCC) — Traditional ↔ Simplified conversion

## License

MIT — see [LICENSE](LICENSE).
