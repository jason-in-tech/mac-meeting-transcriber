"""Model selection + lazy HuggingFace-to-MLX conversion for mmt.

There are two coordinate systems at play:

1. **Logical model pick.** Users don't want to think about HF repo IDs;
   they want to say "this is a Chinese meeting" and get a
   Mandarin-specialized ASR. ``resolve_model()`` maps the
   ``--primary-language`` knob (plus a tiny language→model table) to an
   actual HF repo ID — or respects an explicit ``--model`` override.

2. **Filesystem path mlx-whisper can actually load.** ``mlx_whisper``
   only accepts an MLX-native checkpoint directory containing
   ``config.json`` (with OpenAI-style ``n_audio_state`` etc.) plus
   ``weights.safetensors`` / ``weights.npz``. Most pre-converted models
   on HF live under ``mlx-community/*``. For everything else
   (``BELLE-2/Belle-whisper-large-v3-zh-punct`` and friends),
   ``ensure_mlx_model()`` downloads the HF Transformers-format repo,
   runs a one-shot conversion, caches the result under
   ``~/.cache/mac-meeting-transcriber/mlx-models/<repo>/``, and returns
   the local directory.

The conversion logic is a trimmed port of Apple's
``mlx-examples/whisper/convert.py`` (copyright Apple, Apache-2.0):
weight-key remapping + dtype coercion + config field renaming. We
piggyback on ``mlx_whisper.torch_whisper.ModelDimensions`` and
``mlx_whisper.whisper.Whisper`` so the resulting checkpoint is byte-for-byte
compatible with the ``mlx_whisper.load_model`` reader used upstream.
"""

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

# Default MLX-native English model (pre-converted, fast cold-start).
DEFAULT_EN_MODEL = "mlx-community/whisper-large-v3-mlx"

# Chinese-specialized model (HF Transformers format; auto-converted on first use).
# Belle-whisper-large-v3-zh-punct beats base whisper-large-v3 by 45–65 % on
# Chinese ASR benchmarks (CER) and is explicitly tuned for meeting audio
# (WenetSpeech-meeting CER 10.97 vs base 20.15). It preserves base v3's
# multilingual encoder so code-switched English technical terms still come
# through unchanged.
DEFAULT_ZH_MODEL = "BELLE-2/Belle-whisper-large-v3-zh-punct"

# Known good MLX-native prefixes — we skip conversion entirely for these.
_MLX_NATIVE_PREFIXES = ("mlx-community/",)

# Where we stash converted checkpoints. One subdir per repo, named by
# repo_id with slashes replaced by double-underscore so we can round-trip
# it back out of the filename if needed for debugging.
_DEFAULT_CONVERT_CACHE = Path.home() / ".cache" / "mac-meeting-transcriber" / "mlx-models"


def resolve_model(
    primary_language: str | None,
    model_override: str | None,
) -> str:
    """Pick the HF repo ID to feed into the conversion/loading pipeline.

    Precedence:
      1. ``model_override`` (``--model``) always wins. Use this for
         experiments with other fine-tunes (`BELLE-2/...`, `openai/whisper-*`,
         or any pre-converted MLX repo).
      2. ``primary_language == "zh"`` → ``DEFAULT_ZH_MODEL`` (BELLE).
      3. Everything else → ``DEFAULT_EN_MODEL`` (OpenAI via mlx-community).

    We keep the language list deliberately tiny (en, zh) — a real
    language-router isn't the value add; pinning the right model per
    recording is.
    """
    if model_override:
        return model_override
    if primary_language == "zh":
        return DEFAULT_ZH_MODEL
    return DEFAULT_EN_MODEL


def is_mlx_native_repo(repo_id_or_path: str) -> bool:
    """Does ``mlx_whisper.load_model`` accept this path as-is?

    True if it's:
      - an ``mlx-community/*`` repo (these are always pre-converted), or
      - a local directory that already looks MLX-native (has a
        weights.safetensors or weights.npz next to config.json).

    False for HF Transformers-format repos like ``BELLE-2/*`` or
    ``openai/whisper-*`` — those need conversion.
    """
    if any(repo_id_or_path.startswith(p) for p in _MLX_NATIVE_PREFIXES):
        return True
    p = Path(repo_id_or_path)
    if p.is_dir():
        has_config = (p / "config.json").is_file()
        has_mlx_weights = (p / "weights.safetensors").is_file() or (p / "weights.npz").is_file()
        return has_config and has_mlx_weights
    return False


def ensure_mlx_model(
    repo_id_or_path: str,
    cache_dir: Path | None = None,
    dtype: str = "float16",
    log: Callable[[str], None] = print,
) -> str:
    """Return a path mlx-whisper can load.

    - If ``repo_id_or_path`` is already MLX-native, returns it unchanged.
    - Otherwise downloads the HF repo + converts it, caching the result
      under ``cache_dir/<repo_slug>/``. Idempotent: second call is a
      no-op.

    ``dtype`` is the on-disk weight precision. ``float16`` matches what
    ``mlx-community`` ships and halves memory with no measurable quality
    hit on Whisper-large.

    This is the slow path (~3–5 min on first BELLE run: download 3 GB
    safetensors + CPU-bound weight remap). ``log`` is called with
    human-readable progress messages so callers can surface it through
    their own verbosity gate.
    """
    if is_mlx_native_repo(repo_id_or_path):
        return repo_id_or_path

    cache_dir = cache_dir or _DEFAULT_CONVERT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    slug = repo_id_or_path.replace("/", "__")
    dest = cache_dir / slug
    marker = dest / ".mmt-converted"

    if marker.is_file():
        log(f"Using cached MLX checkpoint for {repo_id_or_path} at {dest}")
        return str(dest)

    log(
        f"First-time setup: converting {repo_id_or_path} to MLX format "
        f"(one-time ~3–5 min; cached at {dest})"
    )

    import mlx.core as mx
    from mlx.utils import tree_flatten

    dtype_mx = getattr(mx, dtype)

    model = _convert_hf_to_mlx(repo_id_or_path, dtype_mx, log=log)

    tmp = dest.with_suffix(".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    try:
        config = asdict(model.dims)
        config["model_type"] = "whisper"
        (tmp / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # mlx_whisper.load_model looks for weights.safetensors first,
        # then weights.npz. We pick safetensors because mx.save_safetensors
        # produces a deterministic, memory-mappable file that loads
        # noticeably faster on repeated runs.
        weights = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(tmp / "weights.safetensors"), weights)

        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
        marker.write_text(
            f"source={repo_id_or_path}\ndtype={dtype}\n", encoding="utf-8"
        )
    except BaseException:
        # On failure (KeyboardInterrupt, disk full, conversion bug, ...)
        # remove the half-written dir so the next run re-starts cleanly.
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    log(f"Conversion complete: {dest}")
    return str(dest)


# ---------------------------------------------------------------------------
# HuggingFace-format → MLX Whisper conversion.
#
# Lifted almost verbatim from Apple's mlx-examples/whisper/convert.py
# (Apache-2.0). We stay faithful to their key-remap table so the output
# is bit-for-bit compatible with pre-converted mlx-community checkpoints.
# ---------------------------------------------------------------------------


# OpenAI whisper config field ← list of HF whisper config fields to read.
# Using a list-of-sources instead of a flat dict because HF's ``d_model``
# is duplicated into both ``n_audio_state`` and ``n_text_state`` on the
# OpenAI side (whisper shares encoder/decoder d_model), which a plain
# forward dict can't express. Mirrors mlx-examples/whisper/convert.py::hf_to_pt
# (Apache-2.0).
_PT_TO_HF_CONFIG = {
    "n_mels": "num_mel_bins",
    "n_audio_ctx": "max_source_positions",
    "n_audio_state": "d_model",
    "n_audio_head": "encoder_attention_heads",
    "n_audio_layer": "encoder_layers",
    "n_vocab": "vocab_size",
    "n_text_ctx": "max_target_positions",
    "n_text_state": "d_model",
    "n_text_head": "decoder_attention_heads",
    "n_text_layer": "decoder_layers",
}


def _remap_hf_weight_key(k: str) -> str:
    """Translate a HuggingFace Whisper state-dict key to the OpenAI naming.

    ``mlx_whisper.whisper.Whisper`` expects OpenAI Whisper parameter
    names (``encoder.blocks.0.attn.query.weight`` etc.). HF Transformers
    uses a different convention (``model.encoder.layers.0.self_attn.q_proj.weight``).
    The mapping is mechanical string-substitution and identical to what
    Apple's convert.py does.
    """
    k = k.replace("model.", "")
    k = k.replace(".layers", ".blocks")
    k = k.replace(".self_attn", ".attn")
    k = k.replace(".attn_layer_norm", ".attn_ln")
    k = k.replace(".encoder_attn.", ".cross_attn.")
    k = k.replace(".encoder_attn_layer_norm", ".cross_attn_ln")
    k = k.replace(".final_layer_norm", ".mlp_ln")
    k = k.replace(".q_proj", ".query")
    k = k.replace(".k_proj", ".key")
    k = k.replace(".v_proj", ".value")
    k = k.replace(".out_proj", ".out")
    k = k.replace(".fc1", ".mlp1")
    k = k.replace(".fc2", ".mlp2")
    k = k.replace("embed_positions.weight", "positional_embedding")
    k = k.replace("decoder.embed_tokens", "decoder.token_embedding")
    k = k.replace("encoder.layer_norm", "encoder.ln_post")
    k = k.replace("decoder.layer_norm", "decoder.ln")
    return k


def _post_remap(k: str, v, dtype):
    """Apply the second-pass fixups that ``mlx-examples`` does post-hf_to_pt.

    After the HF→OpenAI key remap, OpenAI-style weights still have two
    quirks relative to MLX: (a) the FFN is named ``mlp.0`` / ``mlp.2``
    because of nn.Sequential, and (b) Conv1d weights are stored as
    (out_channels, in_channels, kernel) but MLX expects
    (out_channels, kernel, in_channels). Both are fixed here.
    """
    import mlx.core as mx

    k = k.replace("mlp.0", "mlp1")
    k = k.replace("mlp.2", "mlp2")
    if "conv" in k and v.ndim == 3:
        v = v.swapaxes(1, 2)
    if not isinstance(v, mx.array):
        # HF safetensors loaded via ``mx.load`` are already mx.array, but
        # the pytorch_model.bin fallback path returns torch tensors; coerce.
        import torch

        if isinstance(v, torch.Tensor):
            v = mx.array(v.detach().numpy())
    return k, v.astype(dtype)


def _convert_hf_to_mlx(repo_id: str, dtype, log):
    """Download an HF-format Whisper model and return an in-memory MLX ``Whisper``.

    Accepts either a repo ID (downloaded via huggingface_hub) or a local
    directory already holding a Transformers-format checkpoint.
    """
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_whisper.whisper import ModelDimensions, Whisper

    p = Path(repo_id)
    if not p.exists():
        log(f"Downloading {repo_id} from HuggingFace ...")
        local = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                "*.json",
                "model.safetensors",
                "pytorch_model.bin",
                "*.txt",
            ],
        )
        p = Path(local)

    log("Loading HF config and weights ...")
    with open(p / "config.json") as f:
        hf_config = json.load(f)

    safetensors_path = p / "model.safetensors"
    if safetensors_path.is_file():
        weights = mx.load(str(safetensors_path))
    else:
        pt_path = p / "pytorch_model.bin"
        if not pt_path.is_file():
            raise FileNotFoundError(
                f"{repo_id} has neither model.safetensors nor pytorch_model.bin; "
                "can't convert."
            )
        import torch

        weights = torch.load(pt_path, map_location="cpu")

    # HF → OpenAI config-key remap.
    config = {pt_key: hf_config[hf_key] for pt_key, hf_key in _PT_TO_HF_CONFIG.items()}
    # Token embeddings are shared with the output projection in whisper,
    # but HF stores a separate ``proj_out`` head. Drop it.
    weights.pop("proj_out.weight", None)
    weights = {_remap_hf_weight_key(k): v for k, v in weights.items()}

    log(f"Remapping {len(weights)} tensors to MLX layout ...")
    # Drop the learnable encoder positional embedding — Whisper uses a
    # sinusoidal one reconstructed at model-init time, and including the
    # HF copy here produces a strict-mode load failure.
    weights.pop("encoder.positional_embedding", None)
    weights = dict(_post_remap(k, v, dtype) for k, v in weights.items())

    dims = ModelDimensions(**config)
    model = Whisper(dims, dtype)
    # strict=False because a handful of optional buffers (e.g. proj_out)
    # are intentionally missing; tolerated by the upstream convert script too.
    model.load_weights(list(weights.items()), strict=False)
    return model
