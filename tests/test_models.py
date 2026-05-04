"""Tests for model selection and MLX-native-path detection.

We intentionally do NOT test the full HF→MLX conversion path here —
that requires a 3 GB download + MLX compute and is better exercised
manually (or in an offline integration suite). The pure logic around
repo selection and cache layout is fully covered.
"""

from pathlib import Path

from mac_meeting_transcriber.models import (
    DEFAULT_EN_MODEL,
    DEFAULT_ZH_MODEL,
    ensure_mlx_model,
    is_mlx_native_repo,
    resolve_model,
)

# ---------------------------------------------------------------------------
# resolve_model: pure, no IO.
# ---------------------------------------------------------------------------


def test_resolve_model_default_is_english():
    assert resolve_model(None, None) == DEFAULT_EN_MODEL
    assert resolve_model("en", None) == DEFAULT_EN_MODEL


def test_resolve_model_zh_picks_belle():
    assert resolve_model("zh", None) == DEFAULT_ZH_MODEL


def test_resolve_model_override_wins_over_language():
    assert resolve_model("zh", "foo/bar") == "foo/bar"
    assert resolve_model("en", "mlx-community/whisper-tiny") == "mlx-community/whisper-tiny"
    assert resolve_model(None, "openai/whisper-small") == "openai/whisper-small"


def test_resolve_model_empty_override_treated_as_missing():
    # argparse gives us None when --model is omitted; a literal empty
    # string from some other source should also fall through.
    assert resolve_model("zh", "") == DEFAULT_ZH_MODEL


# ---------------------------------------------------------------------------
# is_mlx_native_repo: pure except for optional filesystem introspection.
# ---------------------------------------------------------------------------


def test_mlx_community_repo_is_native():
    assert is_mlx_native_repo("mlx-community/whisper-large-v3-mlx")
    assert is_mlx_native_repo("mlx-community/whisper-tiny-mlx")


def test_belle_and_openai_repos_are_not_native():
    assert not is_mlx_native_repo("BELLE-2/Belle-whisper-large-v3-zh-punct")
    assert not is_mlx_native_repo("openai/whisper-large-v3")


def test_local_dir_native_detection(tmp_path: Path):
    # An MLX-format directory must have config.json + weights.safetensors
    # (or weights.npz) side by side.
    mlx = tmp_path / "mlx-ckpt"
    mlx.mkdir()
    (mlx / "config.json").write_text("{}", encoding="utf-8")
    (mlx / "weights.safetensors").write_bytes(b"fake")
    assert is_mlx_native_repo(str(mlx))

    # npz layout is also acceptable.
    mlx_npz = tmp_path / "mlx-npz"
    mlx_npz.mkdir()
    (mlx_npz / "config.json").write_text("{}", encoding="utf-8")
    (mlx_npz / "weights.npz").write_bytes(b"fake")
    assert is_mlx_native_repo(str(mlx_npz))

    # HF-format dir (model.safetensors, not weights.safetensors) is NOT
    # MLX-native.
    hf = tmp_path / "hf-ckpt"
    hf.mkdir()
    (hf / "config.json").write_text("{}", encoding="utf-8")
    (hf / "model.safetensors").write_bytes(b"fake")
    assert not is_mlx_native_repo(str(hf))


# ---------------------------------------------------------------------------
# ensure_mlx_model: fast path short-circuits for native inputs.
# The slow HF→MLX conversion path is exercised manually.
# ---------------------------------------------------------------------------


def test_ensure_mlx_model_noop_for_native_repo(tmp_path: Path):
    # mlx-community repos should pass through without touching the
    # cache_dir at all. We verify by pointing cache_dir at a tmp path
    # and making sure nothing is written there.
    result = ensure_mlx_model(
        "mlx-community/whisper-large-v3-mlx",
        cache_dir=tmp_path,
        log=lambda _: None,
    )
    assert result == "mlx-community/whisper-large-v3-mlx"
    assert list(tmp_path.iterdir()) == []


def test_ensure_mlx_model_returns_cached_path_without_reconverting(tmp_path: Path):
    # Pre-populate the cache dir with what a successful conversion
    # would have left behind (marker file + config.json + weights).
    slug = "BELLE-2__Belle-whisper-large-v3-zh-punct"
    dest = tmp_path / slug
    dest.mkdir(parents=True)
    (dest / "config.json").write_text("{}", encoding="utf-8")
    (dest / "weights.safetensors").write_bytes(b"fake")
    (dest / ".mmt-converted").write_text("source=...\n", encoding="utf-8")

    log_messages: list[str] = []
    result = ensure_mlx_model(
        "BELLE-2/Belle-whisper-large-v3-zh-punct",
        cache_dir=tmp_path,
        log=log_messages.append,
    )
    assert result == str(dest)
    # Should NOT log the "First-time setup" message — that would mean
    # we blew past the cache check.
    assert not any("First-time setup" in msg for msg in log_messages)
    # But should log that we used the cache.
    assert any("cached" in msg.lower() for msg in log_messages)
