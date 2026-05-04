"""LLM-based speaker identification.

Senko returns anonymous labels like SPEAKER_00 / SPEAKER_01. Rather than
mapping those to real names positionally (which breaks when the labeled
person isn't the first one to speak), we give an LLM a handful of
representative utterances per speaker and ask it to match against a list
of candidate names.

Designed to talk to any OpenAI-compatible endpoint, including the local
Cursor proxy at http://127.0.0.1:8765/v1.
"""

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

from openai import OpenAI

from .merge import AttributedSegment


@dataclass
class Candidate:
    """A candidate speaker name plus an optional free-form role hint.

    The hint (e.g. "manager", "IC", "Jason's direct report") is passed
    through to the LLM to help disambiguate names when the transcript
    alone can't tell them apart.
    """
    name: str
    hint: str = ""


def parse_candidates(raw: str) -> list[Candidate]:
    """Parse '--speakers' input like "Jason:IC,Leo:manager".

    Each comma-separated entry is either "Name" or "Name:role hint".
    Whitespace around each part is stripped. Empty entries are ignored.
    """
    out: list[Candidate] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, sep, hint = chunk.partition(":")
        name = name.strip()
        if not name:
            continue
        out.append(Candidate(name=name, hint=hint.strip() if sep else ""))
    return out


DEFAULT_BASE_URL = "http://127.0.0.1:8765/v1"
DEFAULT_API_KEY = "cursor"
DEFAULT_MODEL = "gpt-5.4-medium"
FALLBACK_MODELS = ["claude-4.6-sonnet-medium", "kimi-k2.5"]


@dataclass
class LLMConfig:
    """Resolved LLM connection settings."""
    base_url: str
    api_key: str
    model: str


@dataclass
class IdentificationResult:
    """What the LLM decided, plus enough context to debug."""
    mapping: dict[str, str]
    confidence: float
    reason: str
    model_used: str
    raw_response: str


def resolve_llm_config(
    base_url_arg: str | None = None,
    api_key_arg: str | None = None,
    model_arg: str | None = None,
) -> LLMConfig:
    """Resolve endpoint + key + model with a clear precedence order.

    Precedence (highest wins):
        1. Explicit CLI args (base_url_arg, api_key_arg, model_arg)
        2. MMT_LLM_* env vars (app-specific override)
        3. OPENAI_BASE_URL / OPENAI_API_KEY (standard env vars)
        4. Cursor proxy at 127.0.0.1:8765 (auto-detected default)
    """
    base_url = (
        base_url_arg
        or os.environ.get("MMT_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    api_key = (
        api_key_arg
        or os.environ.get("MMT_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or DEFAULT_API_KEY
    )
    model = (
        model_arg
        or os.environ.get("MMT_LLM_MODEL")
        or DEFAULT_MODEL
    )
    return LLMConfig(base_url=base_url, api_key=api_key, model=model)


def _distinct_speakers(segments: Iterable[AttributedSegment]) -> list[str]:
    seen: list[str] = []
    for seg in segments:
        if seg.speaker not in seen and seg.speaker != "UNKNOWN":
            seen.append(seg.speaker)
    return seen


def _score_utterance(text: str) -> int:
    """Rough "information density" score for picking good samples.

    Favors medium-length utterances with real content words. Penalizes
    filler like "yeah" / "right" / "mm-hmm".
    """
    text = text.strip()
    length = len(text)
    if length < 20:
        return 0
    if length > 800:
        length = 400
    filler_re = re.compile(r"^(?:yeah|right|ok(?:ay)?|sure|mm+-?hmm|嗯+|对+|好+|是的|哦+)[\s.,!?]*$", re.I)
    if filler_re.match(text):
        return 0
    word_count = len(re.findall(r"\w+", text))
    return length + word_count * 5


def sample_speaker_utterances(
    segments: list[AttributedSegment],
    max_per_speaker: int = 6,
    max_chars_per_utterance: int = 350,
) -> dict[str, list[str]]:
    """Pick a diverse, information-dense sample of utterances per speaker.

    We grab the top-scoring utterances spread across the timeline so the
    LLM sees how each speaker talks at the start, middle, and end.
    """
    per_speaker: dict[str, list[tuple[float, int, str]]] = {}
    for seg in segments:
        if seg.speaker == "UNKNOWN":
            continue
        score = _score_utterance(seg.text)
        if score == 0:
            continue
        text = seg.text.strip()
        if len(text) > max_chars_per_utterance:
            text = text[:max_chars_per_utterance].rstrip() + "..."
        per_speaker.setdefault(seg.speaker, []).append((seg.start, score, text))

    out: dict[str, list[str]] = {}
    for speaker, items in per_speaker.items():
        items.sort(key=lambda x: x[1], reverse=True)
        top = items[: max_per_speaker * 3]
        top.sort(key=lambda x: x[0])
        step = max(1, len(top) // max_per_speaker)
        chosen = top[::step][:max_per_speaker]
        out[speaker] = [text for _, _, text in chosen]
    return out


_ROLE_HINT_SHORTHANDS = {
    "ic": "individual contributor, reports progress and owns execution of technical work",
    "mgr": "manager who sets direction, gives feedback, and reviews work",
    "manager": "manager who sets direction, gives feedback, and reviews work",
    "skip": "skip-level manager (manager's manager)",
    "peer": "peer-level collaborator",
    "report": "direct report (junior on the team)",
    "recorder": "the person who recorded this meeting (the user of this tool)",
    "me": "the person who recorded this meeting (the user of this tool)",
}


def _expand_hint(hint: str) -> str:
    """Expand shorthand role tags like 'ic' / 'mgr' to a descriptive phrase."""
    hint = hint.strip()
    if not hint:
        return ""
    key = hint.lower()
    if key in _ROLE_HINT_SHORTHANDS:
        return _ROLE_HINT_SHORTHANDS[key]
    return hint


def _build_prompt(
    samples: dict[str, list[str]],
    candidates: list[Candidate],
) -> list[dict]:
    has_hints = any(c.hint for c in candidates)

    lines = [
        "You are identifying which anonymous speaker label in a meeting "
        "transcript corresponds to which real person, based on what each "
        "speaker says (content, role signals, tone, vocabulary).",
        "",
        "Candidates:",
    ]
    for c in candidates:
        if c.hint:
            lines.append(f'  - {c.name}  — role: "{_expand_hint(c.hint)}"')
        else:
            lines.append(f"  - {c.name}")
    lines.extend(["", "Sampled utterances per anonymous speaker:", ""])

    for speaker in sorted(samples.keys()):
        lines.append(f"### {speaker}")
        for i, text in enumerate(samples[speaker], 1):
            lines.append(f"{i}. \"{text}\"")
        lines.append("")

    if has_hints:
        lines.append(
            "Use the candidate role descriptions above as the PRIMARY anchor: "
            "decide which speaker best matches each role (who gives direction "
            "vs. reports progress, who probes vs. defends, who sets agenda vs. "
            "executes), then assign the matching candidate name."
        )
    else:
        lines.append(
            "Use role signals in the content (who gives direction vs. reports, "
            "who asks probing questions vs. defends work, who sets agenda vs. "
            "executes, who owns which topic) to match candidates to speakers."
        )
        lines.append(
            "IMPORTANT: If the transcript gives no reliable signal to "
            "distinguish two candidates (e.g. no names mentioned, similar "
            "roles), say so by returning a LOW confidence value so the caller "
            "can ask the user to disambiguate."
        )

    lines.extend([
        "",
        "Return STRICT JSON (no markdown, no commentary) matching this schema:",
        "{",
        '  "mapping": { "SPEAKER_00": "<name>", "SPEAKER_01": "<name>", ... },',
        '  "confidence": <float between 0 and 1>,',
        '  "reason": "<2-4 sentences citing concrete quotes and role signals>"',
        "}",
        "",
        "Every SPEAKER_XX in the transcript MUST appear in mapping. Each "
        "candidate name must appear at most once.",
    ])
    return [
        {
            "role": "system",
            "content": "You are a careful meeting-transcript analyst. You only output JSON.",
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def strip_code_fence(text: str) -> str:
    """Drop a leading ```lang / trailing ``` wrapper if the LLM added one."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def chat_json_call(
    client: OpenAI,
    model: str,
    messages: list[dict],
    timeout: float,
    temperature: float = 0.1,
) -> str:
    """Call chat completions asking for JSON, gracefully downgrading.

    Some OpenAI-compatible endpoints / models reject
    ``response_format={"type": "json_object"}``. We try with it first
    (which strongly biases the model toward valid JSON), and if the
    endpoint rejects it we retry once without — the model is still
    instructed to output JSON via the prompt.

    Shared by ``identify`` (speaker matching) and ``polish`` (copy-edit
    pass), both of which need JSON-shaped responses from the same LLM.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
    return resp.choices[0].message.content or ""


def _validate_mapping(
    raw_mapping: dict,
    expected_speakers: list[str],
    candidate_names: list[str],
) -> dict[str, str]:
    """Accept only mappings that cover every speaker with distinct candidate names."""
    if not isinstance(raw_mapping, dict):
        raise ValueError("mapping is not a dict")
    mapping: dict[str, str] = {}
    for speaker in expected_speakers:
        name = raw_mapping.get(speaker)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"no name assigned to {speaker}")
        if name not in candidate_names:
            raise ValueError(
                f"name '{name}' for {speaker} not in candidates {candidate_names}"
            )
        mapping[speaker] = name
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"duplicate candidate assigned: {mapping}")
    return mapping


def identify_speakers(
    segments: list[AttributedSegment],
    candidates: list[Candidate] | list[str],
    llm: LLMConfig,
    timeout: float = 60.0,
    log=lambda _msg: None,
) -> IdentificationResult:
    """Ask an LLM to match SPEAKER_XX labels to candidate names.

    Each candidate can carry a free-form role hint (e.g. "manager",
    "IC", "Jason's direct report"), which becomes the primary anchor the
    LLM uses to decide which speaker is which. Without hints, the LLM
    falls back to guessing from content, which is unreliable when no one
    says each other's names.

    Raises RuntimeError if no model succeeds, so the caller can fall
    back to positional mapping or leave labels anonymous.
    """
    normalized: list[Candidate] = [
        c if isinstance(c, Candidate) else Candidate(name=c) for c in candidates
    ]
    candidate_names = [c.name for c in normalized]

    speakers = _distinct_speakers(segments)
    if not speakers:
        raise RuntimeError("no speakers present in transcript")
    if len(normalized) < len(speakers):
        raise RuntimeError(
            f"got {len(speakers)} speakers but only {len(normalized)} "
            f"candidate names ({candidate_names}); need at least as many "
            f"names as speakers"
        )

    samples = sample_speaker_utterances(segments)
    if len(samples) < len(speakers):
        log(
            f"  warning: only {len(samples)}/{len(speakers)} speakers had "
            f"scorable utterances; others will fall back to first candidate"
        )

    messages = _build_prompt(samples, normalized)
    client = OpenAI(base_url=llm.base_url, api_key=llm.api_key, timeout=timeout)

    models_to_try = [llm.model] + [m for m in FALLBACK_MODELS if m != llm.model]
    last_err: Exception | None = None

    for model in models_to_try:
        log(f"  asking {model} to identify speakers...")
        try:
            raw = chat_json_call(client, model, messages, timeout=timeout)
            cleaned = strip_code_fence(raw)
            payload = json.loads(cleaned) if cleaned else {}
            raw_mapping = payload.get("mapping") or {}
            mapping = _validate_mapping(raw_mapping, speakers, candidate_names)
            confidence = float(payload.get("confidence") or 0.0)
            reason = str(payload.get("reason") or "").strip()
            return IdentificationResult(
                mapping=mapping,
                confidence=confidence,
                reason=reason,
                model_used=model,
                raw_response=raw,
            )
        except Exception as exc:
            last_err = exc
            log(f"  {model} failed: {exc}")
            continue

    raise RuntimeError(f"all LLM identification attempts failed; last error: {last_err}")
