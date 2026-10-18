import json

from oaiprobe import __version__
from oaiprobe.cli import main
from oaiprobe.mock import DEFAULT_KEY, MockServer


def test_mock_text(capsys):
    assert main(["--mock"]) == 0
    out = capsys.readouterr().out
    assert "fully compatible" in out and "tool_roundtrip" in out


def test_mock_with_quirks_exit_codes(capsys):
    assert main(["--mock", "no_done", "--quick"]) == 0
    assert main(["--mock", "no_done", "--quick", "--strict"]) == 1
    assert main(["--mock", "ignore_stop", "--only", "stop"]) == 1
    capsys.readouterr()


def test_json_output(capsys):
    assert main(["--mock", "reasoning", "-f", "json", "--quick"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["version"] == __version__
    assert data["summary"]["fail"] == 0
    assert {r["id"] for r in data["results"]} >= {"chat", "stream", "tools"}


def test_markdown_output(capsys):
    assert main(["--mock", "no_auth", "-f", "markdown", "--only", "auth,chat"]) == 0
    out = capsys.readouterr().out
    assert "| `auth` | ⚠️ WARN |" in out


def test_real_url_and_model_discovery(capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OAIPROBE_API_KEY", DEFAULT_KEY)
    with MockServer() as url:
        assert main(["-u", url, "--quick", "-f", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model"] == "mock-1"
    assert "picked from GET /models" in data["note"]


def test_api_key_never_printed(capsys):
    with MockServer(api_key="sk-super-secret-123") as url:
        main(["-u", url, "-k", "sk-super-secret-123", "-m", "mock-1"])
        main(["-u", url, "-k", "sk-super-secret-123", "-m", "mock-1", "-f", "json"])
        main(["-u", url, "-k", "sk-wrong-secret-456", "-m", "mock-1", "-f", "markdown"])
    cap = capsys.readouterr()
    assert "sk-super-secret-123" not in cap.out + cap.err
    assert "sk-wrong-secret-456" not in cap.out + cap.err


def test_usage_errors(capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert main([]) == 2
    assert main(["--mock", "nonsense_quirk"]) == 2
    assert main(["--mock", "--only", "nope"]) == 2
    assert main(["-u", "http://127.0.0.1:9", "-m", "x", "--timeout", "2"]) == 2
    err = capsys.readouterr().err
    assert "cannot reach" in err


def test_lists(capsys):
    assert main(["--list-checks"]) == 0
    assert main(["--list-quirks"]) == 0
    out = capsys.readouterr().out
    assert "stream_tools" in out and "cumulative_args" in out


def test_param_is_sent(capsys):
    seen = []
    with MockServer() as url:
        import oaiprobe.http as h

        orig = h.Client.post

        def spy(self, path, payload, auth=True):
            seen.append(payload)
            return orig(self, path, payload, auth)

        h.Client.post = spy
        try:
            assert main(["-u", url, "-k", DEFAULT_KEY, "-m", "mock-1", "--only", "chat,max_tokens", "-p", "temperature=0", "-p", "tag=abc"]) == 0
        finally:
            h.Client.post = orig
    capsys.readouterr()
    assert all(p["temperature"] == 0 and p["tag"] == "abc" for p in seen)
    assert seen[1]["max_tokens"] == 8
    assert main(["--mock", "-p", "novalue"]) == 2
