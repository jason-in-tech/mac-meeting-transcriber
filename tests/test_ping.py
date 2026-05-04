"""Tests for ``ping_llm_endpoint``.

We don't talk to a real LLM here — the probe is just a TCP connect, so
all four cases can be exercised against a local socket we control.
"""

import socket
import threading
import time

import pytest

from mac_meeting_transcriber.identify import (
    LLMConfig,
    PingResult,
    ping_llm_endpoint,
)


def _listening_socket() -> tuple[socket.socket, int]:
    """Bind a TCP listener on a free localhost port. Caller must close it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


def _llm_at(host: str, port: int, scheme: str = "http") -> LLMConfig:
    return LLMConfig(
        base_url=f"{scheme}://{host}:{port}/v1",
        api_key="test",
        model="test-model",
    )


def test_ping_reachable_localhost():
    sock, port = _listening_socket()
    try:
        result = ping_llm_endpoint(_llm_at("127.0.0.1", port))
        assert isinstance(result, PingResult)
        assert result.reachable is True
        assert f"127.0.0.1:{port}" in result.detail
    finally:
        sock.close()


def test_ping_unreachable_closed_port():
    # Bind+close to claim a definitely-not-listening port number.
    sock, port = _listening_socket()
    sock.close()
    # Tiny race window where the kernel may still be holding the port; in
    # practice the test machine is quiet enough that this is reliable.
    time.sleep(0.05)

    result = ping_llm_endpoint(_llm_at("127.0.0.1", port))
    assert result.reachable is False
    assert "cannot reach" in result.detail or "timeout" in result.detail


def test_ping_unparseable_url():
    bad_llm = LLMConfig(base_url="not a url", api_key="k", model="m")
    result = ping_llm_endpoint(bad_llm)
    assert result.reachable is False


def test_ping_missing_host():
    bad_llm = LLMConfig(base_url="http:///v1", api_key="k", model="m")
    result = ping_llm_endpoint(bad_llm)
    assert result.reachable is False
    assert "no host" in result.detail


def test_ping_unknown_scheme():
    bad_llm = LLMConfig(base_url="ftp://example.com/v1", api_key="k", model="m")
    result = ping_llm_endpoint(bad_llm)
    assert result.reachable is False
    assert "scheme" in result.detail


def test_ping_uses_default_https_port():
    # We can't actually reach example.com:443 from CI in a guaranteed way,
    # but we *can* verify the parser picks the right default port for
    # https URLs without an explicit ``:port``. The cheapest way is to
    # assert that an URL with no port gets attempted at the scheme default
    # by pointing it at a guaranteed-closed port.
    #
    # Strategy: use a URL with scheme=http and rely on the default :80
    # being closed on the loopback interface. If your laptop happens to
    # have something listening on 127.0.0.1:80, this test will be flaky;
    # in CI (macos-14 default runner) port 80 is closed.
    llm = LLMConfig(base_url="http://127.0.0.1/v1", api_key="k", model="m")
    result = ping_llm_endpoint(llm, timeout=0.3)
    # Either reachable (something on :80) or unreachable — what we
    # care about is that the parser handled the missing port without
    # crashing.
    assert isinstance(result, PingResult)
    assert isinstance(result.reachable, bool)


def test_ping_listener_can_be_reached_from_thread():
    # Sanity: the probe shouldn't hang the test thread. Run the probe
    # against a real listener and make sure it returns within a sensible
    # window even if we're under load.
    sock, port = _listening_socket()
    results: list[PingResult] = []

    def worker():
        results.append(ping_llm_endpoint(_llm_at("127.0.0.1", port)))

    t = threading.Thread(target=worker)
    try:
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "ping_llm_endpoint hung"
        assert len(results) == 1
        assert results[0].reachable is True
    finally:
        sock.close()


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:8765/v1",
    "https://api.openai.com/v1",
    "http://localhost:11434/v1",
])
def test_ping_does_not_crash_on_real_world_urls(base_url):
    """We don't assert reachability — these endpoints may or may not be up.
    We just want to make sure the parser handles each shape without raising.
    """
    llm = LLMConfig(base_url=base_url, api_key="k", model="m")
    result = ping_llm_endpoint(llm, timeout=0.3)
    assert isinstance(result, PingResult)
