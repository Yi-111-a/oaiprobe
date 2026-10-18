"""Tiny zero-dependency HTTP client for OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import __version__

USER_AGENT = "oaiprobe/" + __version__


class ProbeError(Exception):
    """A request could not be completed (timeout, bad JSON, ...)."""


class ConnectError(ProbeError):
    """The endpoint could not be reached at all."""


@dataclass
class Response:
    status: int
    headers: Dict[str, str]
    body: bytes
    elapsed: float

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProbeError("response is not valid JSON: %s" % excerpt(self.text)) from exc

    def error_message(self) -> str:
        """Best-effort human readable error from a non-2xx body."""
        try:
            data = json.loads(self.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return excerpt(self.text) or "(empty body)"
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return excerpt(str(err["message"]))
            if isinstance(err, str):
                return excerpt(err)
            if data.get("message"):
                return excerpt(str(data["message"]))
            if data.get("detail"):
                return excerpt(str(data["detail"]))
        return excerpt(json.dumps(data))


@dataclass
class Stream:
    status: int
    headers: Dict[str, str]
    events: List[Any] = field(default_factory=list)
    bad_lines: List[str] = field(default_factory=list)
    done: bool = False
    first_event: Optional[float] = None
    first_content: Optional[float] = None
    elapsed: float = 0.0
    body: bytes = b""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def as_response(self) -> Response:
        return Response(self.status, self.headers, self.body, self.elapsed)


def excerpt(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_sse_lines(lines) -> Dict[str, Any]:
    """Parse SSE ``data:`` lines. Returns events, bad lines and [DONE] flag.

    Accepts ``data:`` with or without a following space, CRLF endings,
    comments (``: ping``) and multi-line data fields.
    """
    events: List[Any] = []
    bad: List[str] = []
    done = False
    buf: List[str] = []

    def flush():
        nonlocal done
        if not buf:
            return
        data = "\n".join(buf)
        buf.clear()
        if data.strip() == "[DONE]":
            done = True
            return
        try:
            events.append(json.loads(data))
        except ValueError:
            bad.append(excerpt(data, 80))

    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        line = raw.rstrip("\r\n")
        if line == "":
            flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            buf.append(value)
        # event:, id:, retry: are ignored
    flush()
    return {"events": events, "bad": bad, "done": done}


class Client:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.extra_headers = dict(headers or {})
        self.timeout = timeout
        self.seen_response = False  # has the endpoint ever answered at all?

    def _headers(self, auth: bool = True, stream: bool = False) -> Dict[str, str]:
        h = {"User-Agent": USER_AGENT, "Accept": "text/event-stream" if stream else "application/json"}
        h.update(self.extra_headers)
        if not auth:
            for k in list(h):
                if k.lower() in ("authorization", "api-key", "x-api-key"):
                    del h[k]
        elif self.api_key and not any(k.lower() == "authorization" for k in h):
            h["Authorization"] = "Bearer " + self.api_key
        return h

    def _open(self, method: str, path: str, payload: Any, auth: bool, stream: bool):
        url = self.base_url + path
        data = None
        headers = self._headers(auth=auth, stream=stream)
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            resp = exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise self._timeout(url) from exc
            raise ConnectError("cannot reach %s: %s" % (url, reason)) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise self._timeout(url) from exc
        except (ConnectionError, OSError) as exc:
            raise ConnectError("cannot reach %s: %s" % (url, exc)) from exc
        self.seen_response = True
        return resp

    def _timeout(self, url: str) -> ProbeError:
        if not self.seen_response:
            # nothing has ever come back: treat as unreachable instead of timing out every check
            return ConnectError("no response from %s within %gs" % (url, self.timeout))
        return ProbeError("timed out after %gs" % self.timeout)

    def request(self, method: str, path: str, payload: Any = None, auth: bool = True) -> Response:
        start = time.perf_counter()
        resp = self._open(method, path, payload, auth, stream=False)
        try:
            body = resp.read()
        except (socket.timeout, TimeoutError) as exc:
            raise ProbeError("timed out reading body after %gs" % self.timeout) from exc
        finally:
            resp.close()
        status = getattr(resp, "status", None) or resp.getcode()
        return Response(status, {k.lower(): v for k, v in resp.headers.items()}, body, time.perf_counter() - start)

    def get(self, path: str, auth: bool = True) -> Response:
        return self.request("GET", path, None, auth)

    def post(self, path: str, payload: Any, auth: bool = True) -> Response:
        return self.request("POST", path, payload, auth)

    def stream(self, path: str, payload: Any) -> Stream:
        start = time.perf_counter()
        resp = self._open("POST", path, payload, True, stream=True)
        status = getattr(resp, "status", None) or resp.getcode()
        headers = {k.lower(): v for k, v in resp.headers.items()}
        st = Stream(status, headers)
        try:
            if not (200 <= status < 300):
                st.body = resp.read()
                return st
            raw_lines: List[str] = []
            collected = []

            def lines():
                for raw in resp:
                    now = time.perf_counter() - start
                    line = raw.decode("utf-8", "replace")
                    raw_lines.append(line)
                    if line.startswith("data:") and "[DONE]" not in line:
                        if st.first_event is None:
                            st.first_event = now
                        if st.first_content is None and _has_content(line):
                            st.first_content = now
                    yield line

            parsed = parse_sse_lines(lines())
            collected = parsed
            st.events = collected["events"]
            st.bad_lines = collected["bad"]
            st.done = collected["done"]
            st.body = "".join(raw_lines).encode("utf-8")
        except (socket.timeout, TimeoutError) as exc:
            raise ProbeError("stream timed out after %gs" % self.timeout) from exc
        finally:
            resp.close()
            st.elapsed = time.perf_counter() - start
        return st


def _has_content(line: str) -> bool:
    try:
        obj = json.loads(line[5:].strip())
    except ValueError:
        return False
    for ch in (obj.get("choices") or []) if isinstance(obj, dict) else []:
        delta = ch.get("delta") or {}
        if delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content") or delta.get("reasoning"):
            return True
    return False
