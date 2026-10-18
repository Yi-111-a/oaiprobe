import pytest

from oaiprobe.http import Client, ConnectError, Response, parse_sse_lines


def test_sse_parser_variants():
    lines = [
        ": keep-alive\n",
        'data: {"a": 1}\r\n',
        "\r\n",
        'data:{"b": 2}\n',
        "\n",
        "event: message\n",
        'data: {"c":\n',
        "data: 3}\n",
        "\n",
        "data: not-json\n",
        "\n",
        "data: [DONE]\n",
        "\n",
    ]
    out = parse_sse_lines(lines)
    assert out["events"] == [{"a": 1}, {"b": 2}, {"c": 3}]
    assert out["bad"] == ["not-json"]
    assert out["done"] is True


def test_sse_parser_without_trailing_blank_line():
    out = parse_sse_lines(['data: {"a": 1}'])
    assert out["events"] == [{"a": 1}] and out["done"] is False


def test_error_message_shapes():
    assert Response(400, {}, b'{"error": {"message": "bad thing"}}', 0).error_message() == "bad thing"
    assert Response(400, {}, b'{"error": "flat"}', 0).error_message() == "flat"
    assert Response(422, {}, b'{"detail": "pydantic"}', 0).error_message() == "pydantic"
    assert "oops" in Response(500, {}, b"<h1>oops</h1>", 0).error_message()


def test_auth_header_handling():
    c = Client("http://x/v1/", "sk-1", {"X-Extra": "1"})
    assert c.base_url == "http://x/v1"
    assert c._headers()["Authorization"] == "Bearer sk-1"
    assert "Authorization" not in c._headers(auth=False)
    c2 = Client("http://x", None, {"api-key": "azure"})
    assert "api-key" not in c2._headers(auth=False)
    assert c2._headers()["api-key"] == "azure"


def test_connect_error():
    with pytest.raises(ConnectError):
        Client("http://127.0.0.1:9", timeout=2).get("/models")


def test_silent_server_counts_as_unreachable():
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)  # accepts connections but never answers
    port = srv.getsockname()[1]
    try:
        with pytest.raises(ConnectError, match="no response"):
            Client("http://127.0.0.1:%d/v1" % port, timeout=0.5).get("/models")
    finally:
        srv.close()
