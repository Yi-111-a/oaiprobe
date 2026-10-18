"""A small, deterministic OpenAI-compatible mock server.

It answers the prompts oaiprobe sends (and reasonable others) without any
model, and can imitate common gateway quirks so you can see what each
failure looks like, or test your own client against them.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Optional, Tuple

QUIRKS = {
    "no_done": "stream ends without 'data: [DONE]'",
    "no_stream_usage": "ignores stream_options.include_usage",
    "reject_stream_options": "returns 400 when stream_options is sent",
    "ignore_max_tokens": "ignores max_tokens / max_completion_tokens",
    "reject_max_completion_tokens": "returns 400 for max_completion_tokens",
    "ignore_stop": "ignores stop sequences",
    "no_tools": "ignores tools and answers in text",
    "args_object": "returns function.arguments as a JSON object, not a string",
    "cumulative_args": "streams the full arguments so far in every chunk",
    "reject_tool_required": "returns 400 for tool_choice='required'",
    "single_tool_call": "never makes more than one tool call",
    "substitute_model": "answers every request as 'cheap-mini' and accepts unknown models",
    "accept_any_model": "accepts unknown model names",
    "no_auth": "does not check the API key",
    "no_json_mode": "returns 400 for response_format",
    "fenced_json": "wraps JSON mode output in a ```json fence",
    "ignore_schema": "accepts json_schema but returns unrelated JSON",
    "reasoning": "adds DeepSeek-style reasoning_content and cache usage fields",
    "no_logprobs": "ignores logprobs",
    "ignore_n": "ignores n",
    "reject_developer": "returns 400 for the 'developer' role",
    "html_errors": "returns HTML error pages instead of JSON errors",
    "bad_usage": "reports total_tokens that do not add up",
}

DEFAULT_MODEL = "mock-1"
DEFAULT_KEY = "sk-oaiprobe-mock"
CITIES = ["Paris", "Tokyo", "London", "Berlin", "Beijing", "New York", "Sydney"]
STORY = (
    "The lighthouse keeper climbed the spiral stairs every evening, counting each of the one hundred and twelve "
    "steps while the sea threw itself against the rocks below. "
)


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _text_of(msg: Dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


class MockError(Exception):
    def __init__(self, status: int, message: str, code: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


class Engine:
    def __init__(self, quirks: Iterable[str] = (), model: str = DEFAULT_MODEL, api_key: Optional[str] = DEFAULT_KEY):
        self.quirks = set(quirks)
        unknown = self.quirks - set(QUIRKS)
        if unknown:
            raise ValueError("unknown quirk(s): %s" % ", ".join(sorted(unknown)))
        self.model = model
        self.api_key = api_key
        self.counter = 0

    def q(self, name: str) -> bool:
        return name in self.quirks

    # -------------------------------------------------- request handling
    def check_auth(self, headers: Dict[str, str]) -> None:
        if self.api_key is None or self.q("no_auth"):
            return
        if headers.get("authorization", "") != "Bearer " + self.api_key:
            raise MockError(401, "Incorrect API key provided.", "invalid_api_key")

    def models(self) -> Dict[str, Any]:
        return {"object": "list", "data": [{"id": self.model, "object": "model", "created": 0, "owned_by": "oaiprobe"}]}

    def validate(self, req: Dict[str, Any]) -> None:
        if not isinstance(req.get("messages"), list) or not req["messages"]:
            raise MockError(400, "'messages' must be a non-empty array")
        model = req.get("model")
        if model != self.model and not (self.q("accept_any_model") or self.q("substitute_model")):
            raise MockError(404, "The model `%s` does not exist." % model, "model_not_found")
        roles = {m.get("role") for m in req["messages"] if isinstance(m, dict)}
        if "developer" in roles and self.q("reject_developer"):
            raise MockError(400, "Invalid role: developer")
        if req.get("stream_options") is not None and self.q("reject_stream_options"):
            raise MockError(400, "Unrecognized request argument supplied: stream_options")
        if "max_completion_tokens" in req and self.q("reject_max_completion_tokens"):
            raise MockError(400, "Unrecognized request argument supplied: max_completion_tokens")
        if req.get("tool_choice") == "required" and self.q("reject_tool_required"):
            raise MockError(400, "tool_choice 'required' is not supported")
        if req.get("response_format") and self.q("no_json_mode"):
            raise MockError(400, "response_format is not supported")

    def answer(self, req: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
        """Return (content, tool_calls, finish_reason)."""
        msgs = [m for m in req["messages"] if isinstance(m, dict)]
        last = msgs[-1]
        last_user = next((_text_of(m) for m in reversed(msgs) if m.get("role") == "user"), "")
        tools = req.get("tools") if not self.q("no_tools") else None
        choice = req.get("tool_choice", "auto")
        if tools and choice != "none" and last.get("role") != "tool":
            cities = [c for c in CITIES if c.lower() in last_user.lower()]
            forced = choice == "required" or isinstance(choice, dict)
            if cities or forced:
                cities = cities or ["Paris"]
                if forced and isinstance(choice, dict) or self.q("single_tool_call"):
                    cities = cities[:1]
                name = (choice.get("function") or {}).get("name") if isinstance(choice, dict) else None
                name = name or tools[0]["function"]["name"]
                calls = []
                for i, city in enumerate(cities):
                    self.counter += 1
                    args: Any = {"city": city}
                    calls.append(
                        {
                            "id": "call_mock_%d" % self.counter,
                            "type": "function",
                            "function": {"name": name, "arguments": args if self.q("args_object") else json.dumps(args)},
                        }
                    )
                return None, calls, ("stop" if isinstance(choice, dict) else "tool_calls")
        if last.get("role") == "tool":
            try:
                data = json.loads(_text_of(last))
            except ValueError:
                data = {}
            text = "It is %s°C and %s in %s." % (data.get("temp_c", "?"), data.get("condition", "unknown"), data.get("city", "the city"))
        else:
            text = self._text(req, last_user)
        system = " ".join(_text_of(m) for m in msgs if m.get("role") in ("system", "developer")).lower()
        if "uppercase" in system:
            text = text.upper()
        return text, [], "stop"

    def _text(self, req: Dict[str, Any], prompt: str) -> str:
        rf = req.get("response_format") or {}
        low = prompt.lower()
        if rf.get("type") == "json_object":
            text = json.dumps({"answer": "pong"})
            return "```json\n%s\n```" % text if self.q("fenced_json") else text
        if rf.get("type") == "json_schema":
            if self.q("ignore_schema"):
                return json.dumps({"city": "Paris"})
            schema = (rf.get("json_schema") or {}).get("schema") or {}
            obj = {}
            samples = {"name": "Paris", "country": "France", "population": 2102650}
            for key, spec in (schema.get("properties") or {}).items():
                t = spec.get("type")
                obj[key] = samples.get(key) if key in samples else {"string": "example", "integer": 1, "number": 1.0, "boolean": True, "array": [], "object": {}}.get(t)
            text = json.dumps(obj)
            return "```json\n%s\n```" % text if self.q("fenced_json") else text
        if "pong" in low:
            return "pong"
        if "count from 1 to 10" in low:
            return "1 2 3 4 5 6 7 8 9 10"
        if "story" in low:
            return (STORY * 12).strip()
        if "color" in low:
            return "Blue"
        if "hello" in low:
            return "Hello!"
        return "This is a reply from the oaiprobe mock server."

    def complete(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Build a full completion (also used as the source for streaming)."""
        self.validate(req)
        content, calls, finish = self.answer(req)
        if content is not None:
            stops = req.get("stop")
            stops = [stops] if isinstance(stops, str) else (stops or [])
            if stops and not self.q("ignore_stop"):
                cut = min((content.find(s) for s in stops if s and s in content), default=-1)
                if cut >= 0:
                    content = content[:cut].rstrip()
            limit = req.get("max_completion_tokens", req.get("max_tokens"))
            if isinstance(limit, int) and not self.q("ignore_max_tokens"):
                words = re.findall(r"\S+\s*", content)
                if len(words) > limit:
                    content = "".join(words[:limit]).rstrip()
                    finish = "length"
        prompt_tokens = sum(_words(_text_of(m)) for m in req["messages"] if isinstance(m, dict)) + 3
        if req.get("tools"):
            prompt_tokens += len(json.dumps(req["tools"])) // 4
        completion_tokens = _words(content) if content is not None else sum(
            len(json.dumps(c["function"]["arguments"])) // 4 + 1 for c in calls
        )
        n = req.get("n") or 1
        if self.q("ignore_n"):
            n = 1
        choices = []
        for i in range(n):
            msg: Dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = calls
            if self.q("reasoning"):
                msg["reasoning_content"] = "The user wants a short answer."
            ch: Dict[str, Any] = {"index": i, "message": msg, "finish_reason": finish}
            if req.get("logprobs") and not self.q("no_logprobs") and content:
                top = int(req.get("top_logprobs") or 0)
                toks = re.findall(r"\S+", content)
                ch["logprobs"] = {
                    "content": [
                        {
                            "token": t,
                            "logprob": -0.01,
                            "bytes": list(t.encode()),
                            "top_logprobs": [{"token": t, "logprob": -0.01, "bytes": list(t.encode())}][:top]
                            + [{"token": "x", "logprob": -5.0, "bytes": [120]}] * max(0, top - 1),
                        }
                        for t in toks
                    ]
                }
            choices.append(ch)
        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens * n,
            "total_tokens": prompt_tokens + completion_tokens * n + (7 if self.q("bad_usage") else 0),
        }
        if self.q("reasoning"):
            usage["prompt_cache_hit_tokens"] = 0
            usage["prompt_cache_miss_tokens"] = prompt_tokens
            usage["completion_tokens_details"] = {"reasoning_tokens": 6}
        self.counter += 1
        return {
            "id": "chatcmpl-mock-%d" % self.counter,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "cheap-mini" if self.q("substitute_model") else req.get("model"),
            "choices": choices,
            "usage": usage,
        }

    def chunks(self, req: Dict[str, Any]) -> List[Dict[str, Any]]:
        full = self.complete(req)
        base = {"id": full["id"], "object": "chat.completion.chunk", "created": full["created"], "model": full["model"]}
        out = []

        def chunk(delta, finish=None):
            d = dict(base)
            d["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
            out.append(d)

        msg = full["choices"][0]["message"]
        finish = full["choices"][0]["finish_reason"]
        chunk({"role": "assistant", "content": ""})
        if msg.get("reasoning_content"):
            chunk({"reasoning_content": msg["reasoning_content"]})
        for piece in re.findall(r"\S+\s*", msg.get("content") or ""):
            chunk({"content": piece})
        for i, tc in enumerate(msg.get("tool_calls") or []):
            args = tc["function"]["arguments"]
            chunk({"tool_calls": [{"index": i, "id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": ""}}]})
            if not isinstance(args, str):
                chunk({"tool_calls": [{"index": i, "function": {"arguments": args}}]})
                continue
            so_far = ""
            for j in range(0, len(args), 5):
                so_far += args[j : j + 5]
                piece = so_far if self.q("cumulative_args") else args[j : j + 5]
                chunk({"tool_calls": [{"index": i, "function": {"arguments": piece}}]})
        chunk({}, finish)
        so = req.get("stream_options") or {}
        if so.get("include_usage") and not self.q("no_stream_usage"):
            d = dict(base)
            d["choices"] = []
            d["usage"] = full["usage"]
            out.append(d)
        return out


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "oaiprobe-mock"
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):  # keep test output quiet
            pass

        def _path(self) -> str:
            path = self.path.split("?", 1)[0].rstrip("/")
            return path[3:] if path.startswith("/v1") else path

        def _send_json(self, status: int, obj: Any):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, err: MockError):
            if engine.q("html_errors"):
                body = ("<html><body><h1>%d</h1><p>%s</p></body></html>" % (err.status, err.message)).encode()
                self.send_response(err.status)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send_json(err.status, {"error": {"message": err.message, "type": "invalid_request_error", "param": None, "code": err.code}})

        def _headers(self) -> Dict[str, str]:
            return {k.lower(): v for k, v in self.headers.items()}

        def do_GET(self):
            try:
                engine.check_auth(self._headers())
                if self._path() == "/models":
                    return self._send_json(200, engine.models())
                raise MockError(404, "Not found: %s" % self.path, "not_found")
            except MockError as err:
                self._send_error(err)

        def do_POST(self):
            try:
                engine.check_auth(self._headers())
                if self._path() != "/chat/completions":
                    raise MockError(404, "Not found: %s" % self.path, "not_found")
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    req = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    raise MockError(400, "request body is not valid JSON")
                if not isinstance(req, dict):
                    raise MockError(400, "request body must be a JSON object")
                if not req.get("stream"):
                    return self._send_json(200, engine.complete(req))
                chunks = engine.chunks(req)
            except MockError as err:
                return self._send_error(err)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for ch in chunks:
                self.wfile.write(("data: %s\n\n" % json.dumps(ch)).encode())
                self.wfile.flush()
            if not engine.q("no_done"):
                self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


class MockServer:
    """Run the mock in a background thread: ``with MockServer() as url: ...``."""

    def __init__(self, quirks: Iterable[str] = (), host: str = "127.0.0.1", port: int = 0, model: str = DEFAULT_MODEL, api_key: Optional[str] = DEFAULT_KEY):
        self.engine = Engine(quirks, model=model, api_key=api_key)
        self.httpd = ThreadingHTTPServer((host, port), make_handler(self.engine))
        self.httpd.daemon_threads = True
        self.thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return "http://%s:%d/v1" % (host, port)

    def start(self) -> "MockServer":
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def serve_forever(self) -> None:
        try:
            self.httpd.serve_forever()
        finally:
            self.httpd.server_close()

    def __enter__(self) -> str:
        self.start()
        return self.url

    def __exit__(self, *exc) -> None:
        self.stop()
